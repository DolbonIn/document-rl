"""Async GRPO trainer with LoRA adapter support.

Implements the Tinker side of the Tinker-Atropos framework:
  - Asynchronous training loop decoupled from environment
  - LoRA adapter for lightweight fine-tuning (prevents overfitting on small data)
  - GRPO policy gradient with clipped ratio and KL penalty
  - Rollout buffer for async rollout consumption
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .environment import RolloutGroup
from .sampler import SamplerOutput

logger = logging.getLogger(__name__)


def _load_causal_lm(transformers: Any, model_path: str, dtype: Any, device: str) -> Any:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    return model.to(device)


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    def to_peft_config(self) -> Any:
        peft = importlib.import_module("peft")
        return peft.LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
            target_modules=self.target_modules,
        )


@dataclass
class TrainerConfig:
    """Configuration for AsyncGRPOTrainer."""

    model_path: str = ""
    output_dir: str = "/raid/html_rl_output"
    device: str = "cuda"

    # LoRA
    use_lora: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    lora_checkpoint: str | None = None

    # GRPO hyperparameters
    learning_rate: float = 1e-5
    beta_kl: float = 0.02
    clip_range: float = 0.2
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 10

    # Training loop
    max_steps: int = 500
    save_every: int = 50
    log_every: int = 10
    rollout_buffer_size: int = 64

    # Precision
    dtype: str = "bfloat16"


@dataclass
class TrainStep:
    """A single training record from a rollout."""

    prompt: str
    completion: str
    old_logprobs: list[float]
    ref_logprobs: list[float]
    reward: float
    advantage: float
    group_id: str


class AsyncGRPOTrainer:
    """Asynchronous GRPO trainer with LoRA.

    The trainer maintains a rollout buffer. Training steps consume
    rollouts asynchronously while the environment keeps generating.
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._optimizer = None
        self._scheduler = None
        self._ref_model = None
        self._rollout_buffer: deque[TrainStep] = deque(
            maxlen=config.rollout_buffer_size
        )
        self._step = 0
        self._accum_loss = 0.0
        self._accum_count = 0
        self._losses: list[float] = []
        self._rewards: list[float] = []
        self._start_time: float = 0.0

    async def setup(self) -> None:
        """Initialize model, LoRA adapter, optimizer, and reference model."""
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        if not self.config.model_path.strip():
            raise RuntimeError("trainer.model_path is required for async GRPO runs")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        logger.info("Loading base model: %s", self.config.model_path)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_path
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = _load_causal_lm(
            transformers,
            self.config.model_path,
            dtype,
            self.config.device,
        )

        # Apply LoRA adapter
        if self.config.use_lora:
            peft = importlib.import_module("peft")
            if self.config.lora_checkpoint:
                logger.info("Loading LoRA checkpoint: %s", self.config.lora_checkpoint)
                self._model = peft.PeftModel.from_pretrained(
                    self._model, self.config.lora_checkpoint
                )
            else:
                lora_config = self.config.lora.to_peft_config()
                self._model = peft.get_peft_model(self._model, lora_config)

            trainable = sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self._model.parameters())
            logger.info(
                "LoRA applied: %d trainable / %d total params (%.2f%%)",
                trainable,
                total,
                100.0 * trainable / total,
            )

        # Reference model (frozen copy for KL computation)
        self._ref_model = _load_causal_lm(
            transformers,
            self.config.model_path,
            dtype,
            self.config.device,
        )
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad = False

        # Optimizer
        self._optimizer = torch.optim.AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Linear warmup scheduler
        self._scheduler = torch.optim.lr_scheduler.LinearLR(
            self._optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

        logger.info(
            "Trainer ready: device=%s, dtype=%s, lora=%s, lr=%.2e",
            self.config.device,
            self.config.dtype,
            self.config.use_lora,
            self.config.learning_rate,
        )

    def ingest_rollouts(
        self,
        groups: list[RolloutGroup],
        sampler_outputs: list[SamplerOutput],
    ) -> int:
        """Ingest scored rollouts into the training buffer."""
        output_map = {o.problem_id: o for o in sampler_outputs}
        added = 0

        for group in groups:
            if group.skipped:
                continue
            for problem, score, advantage in zip(
                group.problems, group.scores, group.advantages
            ):
                output = output_map.get(problem.problem_id)
                if output is None or not output.text:
                    continue

                self._rollout_buffer.append(
                    TrainStep(
                        prompt=f"{problem.system_prompt}\n\n{problem.user_prompt}",
                        completion=output.text,
                        old_logprobs=output.logprobs,
                        ref_logprobs=[],
                        reward=score.reward,
                        advantage=advantage,
                        group_id=group.group_id,
                    )
                )
                self._rewards.append(score.reward)
                added += 1

        return added

    async def train_step(self) -> dict[str, float] | None:
        """Execute one GRPO training step from the rollout buffer."""
        if not self._rollout_buffer:
            return None
        if (
            self._model is None
            or self._tokenizer is None
            or self._ref_model is None
            or self._optimizer is None
            or self._scheduler is None
        ):
            raise RuntimeError(
                "AsyncGRPOTrainer.setup() must be called before train_step"
            )

        torch = importlib.import_module("torch")
        record = self._rollout_buffer.popleft()

        self._model.train()

        # Encode prompt + completion
        full_text = record.prompt + record.completion
        encoded = self._tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        )
        encoded = {k: v.to(self.config.device) for k, v in encoded.items()}

        # Forward pass - current policy
        outputs = self._model(**encoded)
        logits = outputs.logits[:, :-1, :]
        labels = encoded["input_ids"][:, 1:]
        log_probs = (
            torch.log_softmax(logits, dim=-1)
            .gather(-1, labels.unsqueeze(-1))
            .squeeze(-1)
        )

        # Compute prompt length to mask completion tokens only
        prompt_encoded = self._tokenizer(record.prompt, return_tensors="pt")
        prompt_len = prompt_encoded["input_ids"].shape[1] - 1

        completion_mask = torch.zeros_like(log_probs)
        completion_mask[:, prompt_len:] = 1.0
        masked_log_probs = (
            log_probs * completion_mask
        ).sum() / completion_mask.sum().clamp(min=1)

        # Reference model logprobs for KL
        with torch.no_grad():
            ref_outputs = self._ref_model(**encoded)
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs = (
                torch.log_softmax(ref_logits, dim=-1)
                .gather(-1, labels.unsqueeze(-1))
                .squeeze(-1)
            )
            masked_ref_log_probs = (
                ref_log_probs * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1)

        # Old policy logprobs
        if record.old_logprobs:
            old_mean = torch.tensor(
                sum(record.old_logprobs) / max(1, len(record.old_logprobs)),
                device=self.config.device,
                dtype=masked_log_probs.dtype,
            )
        else:
            old_mean = masked_log_probs.detach()

        # GRPO loss: clipped policy gradient + KL penalty
        advantage = torch.tensor(
            record.advantage, device=self.config.device, dtype=masked_log_probs.dtype
        )
        ratio = torch.exp(masked_log_probs - old_mean)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range,
        )
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

        # KL divergence penalty
        kl = torch.abs(masked_log_probs - masked_ref_log_probs)
        loss = policy_loss + self.config.beta_kl * kl

        # Gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()
        self._accum_loss += float(loss.detach().cpu())
        self._accum_count += 1

        metrics: dict[str, float] = {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "ratio": float(ratio.detach().cpu()),
            "advantage": record.advantage,
            "reward": record.reward,
        }

        # Optimizer step every N accumulation steps
        if self._accum_count >= self.config.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), self.config.max_grad_norm
            )
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()

            avg_loss = self._accum_loss / self._accum_count
            self._losses.append(avg_loss)
            self._accum_loss = 0.0
            self._accum_count = 0
            self._step += 1

            metrics["step"] = self._step
            metrics["avg_loss"] = avg_loss
            metrics["lr"] = self._scheduler.get_last_lr()[0]

            if self._step % self.config.log_every == 0:
                recent_rewards = self._rewards[-100:]
                logger.info(
                    "Step %d | loss=%.4f | reward=%.3f | kl=%.4f | lr=%.2e | buffer=%d",
                    self._step,
                    avg_loss,
                    sum(recent_rewards) / max(1, len(recent_rewards)),
                    metrics["kl"],
                    metrics["lr"],
                    len(self._rollout_buffer),
                )

            if self._step % self.config.save_every == 0:
                await self.save_checkpoint()

        return metrics

    async def train_on_buffer(self) -> list[dict[str, float]]:
        """Drain the rollout buffer, executing training steps."""
        all_metrics: list[dict[str, float]] = []
        while self._rollout_buffer and self._step < self.config.max_steps:
            metrics = await self.train_step()
            if metrics:
                all_metrics.append(metrics)
        return all_metrics

    async def save_checkpoint(self, tag: str | None = None) -> str:
        """Save LoRA adapter (or full model) checkpoint."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "AsyncGRPOTrainer.setup() must be called before save_checkpoint"
            )

        suffix = tag or f"step_{self._step}"
        save_path = Path(self.config.output_dir) / f"checkpoint-{suffix}"
        save_path.mkdir(parents=True, exist_ok=True)

        if self.config.use_lora:
            self._model.save_pretrained(str(save_path))
            logger.info("LoRA adapter saved: %s", save_path)
        else:
            self._model.save_pretrained(str(save_path))
            self._tokenizer.save_pretrained(str(save_path))
            logger.info("Full model saved: %s", save_path)

        state = {
            "step": self._step,
            "losses": self._losses[-100:],
            "mean_reward": (
                sum(self._rewards[-100:]) / max(1, len(self._rewards[-100:]))
            ),
            "total_rollouts": len(self._rewards),
            "elapsed_seconds": time.time() - self._start_time,
            "config": {
                "model_path": self.config.model_path,
                "use_lora": self.config.use_lora,
                "learning_rate": self.config.learning_rate,
                "beta_kl": self.config.beta_kl,
                "clip_range": self.config.clip_range,
            },
        }
        state_path = save_path / "training_state.json"
        state_path.write_text(json.dumps(state, indent=2))

        return str(save_path)

    async def cleanup(self) -> None:
        """Final save and cleanup."""
        if self._step > 0:
            await self.save_checkpoint(tag="final")
        self._model = None
        self._ref_model = None
        self._tokenizer = None
        logger.info(
            "Training complete: %d steps, mean reward %.3f",
            self._step,
            sum(self._rewards) / max(1, len(self._rewards)) if self._rewards else 0,
        )

    @property
    def is_done(self) -> bool:
        return self._step >= self.config.max_steps

    @property
    def stats(self) -> dict[str, Any]:
        recent = self._rewards[-100:]
        return {
            "step": self._step,
            "buffer_size": len(self._rollout_buffer),
            "total_rollouts": len(self._rewards),
            "mean_reward": sum(recent) / max(1, len(recent)) if recent else 0.0,
            "mean_loss": (
                sum(self._losses[-10:]) / max(1, len(self._losses[-10:]))
                if self._losses
                else 0.0
            ),
            "elapsed_seconds": time.time() - self._start_time
            if self._start_time
            else 0,
        }
