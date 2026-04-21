"""Async GRPO training runner.

Orchestrates the full async training loop:
  Environment <-> Sampler <-> Trainer

Usage:
    python -m ppt_rl.atropos.runner --config configs/async_grpo_lora.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import EnvConfig, HtmlFormattingEnv
from .sampler import build_sampler
from .trainer import AsyncGRPOTrainer, LoRAConfig, TrainerConfig

logger = logging.getLogger(__name__)


def _validate_run_config(config: RunConfig) -> None:
    if not config.env.tasks_path:
        raise RuntimeError("env.tasks_path is required for async GRPO runs")
    if not Path(config.env.tasks_path).exists():
        raise RuntimeError(f"env.tasks_path does not exist: {config.env.tasks_path}")
    if not config.trainer.model_path.strip():
        raise RuntimeError("trainer.model_path is required for async GRPO runs")


@dataclass
class RunConfig:
    """Top-level run configuration."""

    env: EnvConfig
    trainer: TrainerConfig
    sampler_type: str = "vllm"
    sampler_kwargs: dict[str, Any] | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 768
    batch_size: int = 4
    log_interval_seconds: float = 30.0


def load_run_config(path: str | Path) -> RunConfig:
    """Load configuration from YAML file."""
    import yaml

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid async GRPO config: {path}")

    env_cfg = EnvConfig(**raw.get("env", {}))

    trainer_raw = raw.get("trainer", {})
    lora_raw = trainer_raw.pop("lora", {})
    lora_cfg = LoRAConfig(**lora_raw) if lora_raw else LoRAConfig()
    trainer_cfg = TrainerConfig(**trainer_raw, lora=lora_cfg)

    config = RunConfig(
        env=env_cfg,
        trainer=trainer_cfg,
        sampler_type=raw.get("sampler_type", "vllm"),
        sampler_kwargs=raw.get("sampler", {}),
        temperature=raw.get("temperature", 0.8),
        top_p=raw.get("top_p", 0.95),
        max_tokens=raw.get("max_tokens", 768),
        batch_size=raw.get("batch_size", 4),
        log_interval_seconds=raw.get("log_interval_seconds", 30.0),
    )
    _validate_run_config(config)
    return config


async def run_async_grpo(config: RunConfig) -> dict[str, Any]:
    """Execute the full async GRPO training loop.

    Flow:
    1. Environment generates problems
    2. Sampler generates HTML responses (async, via vLLM or local)
    3. Environment scores responses (structure + fidelity + visual)
    4. Trainer ingests rollouts and updates LoRA weights (GRPO)
    5. Repeat until max_steps
    """
    env = HtmlFormattingEnv(config.env)
    trainer = AsyncGRPOTrainer(config.trainer)
    sampler = build_sampler(config.sampler_type, **(config.sampler_kwargs or {}))

    await env.setup()
    await trainer.setup()

    logger.info("=== Async GRPO Training Started ===")
    logger.info(
        "Tasks: %d, Batch: %d, Max steps: %d",
        len(env.tasks),
        config.batch_size,
        config.trainer.max_steps,
    )

    last_log_time = time.time()
    total_rollouts = 0
    iteration = 0

    try:
        while not trainer.is_done:
            iteration += 1

            # Step 1: Generate problems
            problems = await env.generate_problems(batch_size=config.batch_size)

            # Step 2: Sample responses (async)
            outputs = await sampler.generate(
                problems,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )

            # Step 3: Score responses
            responses = [o.text for o in outputs]
            scores = await env.score_batch(problems, responses)

            # Step 4: Compute group advantages
            groups = env.compute_group_advantages(problems, scores)

            # Step 5: Ingest into trainer
            added = trainer.ingest_rollouts(groups, outputs)
            total_rollouts += added

            # Step 6: Train on buffer
            await trainer.train_on_buffer()

            # Periodic logging
            now = time.time()
            if now - last_log_time > config.log_interval_seconds:
                logger.info(
                    "Iteration %d | step %d/%d | rollouts=%d | buffer=%d | "
                    "reward=%.3f | env_scored=%d",
                    iteration,
                    trainer.stats["step"],
                    config.trainer.max_steps,
                    total_rollouts,
                    trainer.stats["buffer_size"],
                    trainer.stats["mean_reward"],
                    env.stats["total_scored"],
                )
                last_log_time = now

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        await sampler.close()
        await trainer.cleanup()
        await env.cleanup()

    summary = {
        "status": "completed" if trainer.is_done else "interrupted",
        "total_iterations": iteration,
        "total_rollouts": total_rollouts,
        "trainer": trainer.stats,
        "env": env.stats,
    }
    logger.info("=== Training Summary ===\n%s", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Async GRPO training with LoRA for HTML document generation"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_run_config(args.config)
    summary = asyncio.run(run_async_grpo(config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
