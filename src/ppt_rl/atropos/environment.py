"""Atropos-compatible environment for HTML document formatting tasks.

Implements the Tinker-Atropos environment protocol:
  setup() -> generate_problem() -> score(response) -> cleanup()

Reward = structure_accuracy + visual_fidelity + prompt_compliance
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..rewards import (
    aggregate_reward,
    compute_advantages,
    compute_fidelity,
)
from ..schemas import (
    ArtifactRecord,
    RewardRecord,
    SamplingParams,
    TaskSpec,
    TrajectoryGroup,
)
from ..text_utils import extract_visible_text
from ..validation import validate_candidate, ValidatorResult
from ..renderers import FallbackRenderer

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """A single HTML formatting problem for the model to solve."""

    problem_id: str
    task: TaskSpec
    system_prompt: str
    user_prompt: str
    group_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "task_id": self.task.task_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "group_id": self.group_id,
            "metadata": self.metadata,
        }


@dataclass
class Score:
    """Scoring result for a model response."""

    problem_id: str
    candidate_id: str
    reward: float
    metrics: dict[str, float]
    is_valid: bool
    feedback: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "candidate_id": self.candidate_id,
            "reward": self.reward,
            "metrics": self.metrics,
            "is_valid": self.is_valid,
            "feedback": self.feedback,
        }


@dataclass
class RolloutGroup:
    """A group of rollouts for GRPO advantage computation."""

    group_id: str
    task: TaskSpec
    problems: list[Problem]
    scores: list[Score]
    advantages: list[float]
    skipped: bool = False
    skip_reason: str | None = None


# --- Reward weight presets ---------------------------------------------------

REWARD_PRESETS = {
    "default": {
        "structure_weight": 0.30,
        "visual_weight": 0.45,
        "fidelity_weight": 0.25,
    },
    "structure_heavy": {
        "structure_weight": 0.50,
        "visual_weight": 0.30,
        "fidelity_weight": 0.20,
    },
    "fidelity_heavy": {
        "structure_weight": 0.25,
        "visual_weight": 0.30,
        "fidelity_weight": 0.45,
    },
}


@dataclass
class EnvConfig:
    """Configuration for HtmlFormattingEnv."""

    tasks_path: str = ""
    artifacts_dir: str = "/tmp/html_rl_artifacts"
    num_candidates_per_task: int = 4
    renderer: str = "fallback"
    reward_preset: str = "default"
    seed: int = 42
    max_new_tokens: int = 768
    # Reward weight overrides (None = use preset)
    structure_weight: float | None = None
    visual_weight: float | None = None
    fidelity_weight: float | None = None

    def effective_weights(self) -> dict[str, float]:
        preset = REWARD_PRESETS.get(self.reward_preset, REWARD_PRESETS["default"])
        return {
            "structure_weight": (
                self.structure_weight
                if self.structure_weight is not None
                else preset["structure_weight"]
            ),
            "visual_weight": (
                self.visual_weight
                if self.visual_weight is not None
                else preset["visual_weight"]
            ),
            "fidelity_weight": (
                self.fidelity_weight
                if self.fidelity_weight is not None
                else preset["fidelity_weight"]
            ),
        }


SYSTEM_PROMPT = (
    "You are an expert HTML/CSS document designer. Generate a single self-contained "
    "HTML page with inline CSS only. No external assets, no JavaScript. The output "
    'must use <main class="slide"> as the root container. Follow the task constraints '
    "exactly."
)


def _build_user_prompt(task: TaskSpec) -> str:
    parts = [
        f"Task: {task.prompt}",
        f"Document type: {task.doc_type}",
        f"Viewport: {task.constraints.viewport.width}x{task.constraints.viewport.height}",
        f"Aspect ratio: {task.constraints.aspect_ratio}",
        f"Theme: {task.constraints.theme}",
    ]
    if task.constraints.must_include:
        parts.append(f"Must include: {', '.join(task.constraints.must_include)}")
    if task.constraints.must_not_include:
        parts.append(
            f"Must NOT include: {', '.join(task.constraints.must_not_include)}"
        )
    parts.append("Return only the complete HTML document.")
    return "\n".join(parts)


class HtmlFormattingEnv:
    """Atropos environment for HTML formatting RL.

    Lifecycle:
        env = HtmlFormattingEnv(config)
        await env.setup()
        while True:
            problems = await env.generate_problems(batch_size)
            # ... model generates responses ...
            scores = await env.score_batch(problems, responses)
            groups = env.compute_group_advantages(problems, scores)
            # ... trainer consumes groups ...
        await env.cleanup()
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.tasks: list[TaskSpec] = []
        self._task_index = 0
        self._renderer = FallbackRenderer()
        self._artifacts_dir = Path(config.artifacts_dir)
        self._rng = random.Random(config.seed)
        self._weights = config.effective_weights()
        self._epoch = 0
        self._total_scored = 0

    async def setup(self) -> None:
        """Load tasks and initialize renderer."""
        from ..task_sampler import load_tasks

        if not self.config.tasks_path:
            raise RuntimeError("env.tasks_path is required for async GRPO runs")

        self.tasks = load_tasks(Path(self.config.tasks_path))
        if not self.tasks:
            raise RuntimeError(
                f"No tasks found for async GRPO run: {self.config.tasks_path}"
            )

        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self.config.renderer != "fallback":
            from ..renderers import build_renderer

            self._renderer = build_renderer(self.config.renderer)

        logger.info(
            "HtmlFormattingEnv ready: %d tasks, renderer=%s, weights=%s",
            len(self.tasks),
            self.config.renderer,
            self._weights,
        )

    async def generate_problems(self, batch_size: int = 1) -> list[Problem]:
        """Generate a batch of problems from the task pool.

        Each task produces `num_candidates_per_task` problems in one group,
        so the model generates K completions per task for GRPO.
        """
        problems: list[Problem] = []
        tasks_needed = max(1, batch_size // self.config.num_candidates_per_task)

        for _ in range(tasks_needed):
            if self._task_index >= len(self.tasks):
                self._task_index = 0
                self._epoch += 1
                self._rng.shuffle(self.tasks)

            task = self.tasks[self._task_index]
            self._task_index += 1
            group_id = f"grp_{task.task_id}_{self._epoch}_{uuid.uuid4().hex[:8]}"
            user_prompt = _build_user_prompt(task)

            for k in range(self.config.num_candidates_per_task):
                problems.append(
                    Problem(
                        problem_id=f"{group_id}_k{k}",
                        task=task,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        group_id=group_id,
                        metadata={
                            "candidate_index": k,
                            "epoch": self._epoch,
                            "seed": self.config.seed + self._task_index + k,
                        },
                    )
                )

        return problems

    async def score(self, problem: Problem, response: str) -> Score:
        """Score a single HTML response against its task.

        Returns a Score with:
          - structure reward (render success + diagnostics)
          - visual reward (heuristic pairwise, set to 0.5 for single scoring)
          - fidelity reward (prompt compliance)
        """
        from ..schemas import HtmlCandidate

        candidate = HtmlCandidate(
            candidate_id=problem.problem_id,
            task_id=problem.task.task_id,
            group_id=problem.group_id,
            model_version="async_grpo",
            output_format="constrained_html",
            html=response,
            sampling_params=SamplingParams(
                seed=problem.metadata.get("seed", 42),
                max_new_tokens=self.config.max_new_tokens,
            ),
            prompt=problem.user_prompt,
        )

        # Validate HTML structure
        val_result = validate_candidate(candidate)
        if val_result.failed:
            return Score(
                problem_id=problem.problem_id,
                candidate_id=problem.problem_id,
                reward=-1.0,
                metrics={
                    "structure_reward": -1.0,
                    "visual_reward": 0.0,
                    "fidelity_reward": 0.0,
                    "diagnostic_penalty": 0.0,
                    "hacking_penalty": 0.0,
                },
                is_valid=False,
                feedback=f"Validation failed: {val_result.error_type}",
            )

        # Render and compute diagnostics
        task_dir = self._artifacts_dir / problem.task.task_id
        render_result, artifact = self._renderer.render(
            problem.task, candidate, task_dir
        )

        if render_result.failed:
            return Score(
                problem_id=problem.problem_id,
                candidate_id=problem.problem_id,
                reward=-0.8,
                metrics={
                    "structure_reward": -0.8,
                    "visual_reward": 0.0,
                    "fidelity_reward": 0.0,
                    "diagnostic_penalty": 0.0,
                    "hacking_penalty": 0.0,
                },
                is_valid=False,
                feedback=f"Render failed: {render_result.error_type}",
            )

        # Compute individual reward components
        fidelity = compute_fidelity(problem.task, artifact)
        reward_record = aggregate_reward(
            candidate_id=problem.problem_id,
            failure_type=render_result.error_type,
            visual_reward=0.5,  # Neutral for single-candidate scoring
            fidelity_reward=fidelity,
            overflow_score=render_result.diagnostics.overflow_score,
            tiny_text_count=render_result.diagnostics.tiny_text_count,
            contrast_failure_count=render_result.diagnostics.contrast_failure_count,
            clipped_element_count=render_result.diagnostics.clipped_element_count,
            artifact=artifact,
        )

        # Re-weight using configured weights
        w = self._weights
        weighted_reward = (
            w["structure_weight"] * (1.0 if not render_result.failed else -1.0)
            + w["visual_weight"] * reward_record.visual_reward
            + w["fidelity_weight"] * fidelity
            - reward_record.diagnostic_penalty
            - reward_record.hacking_penalty
        )
        weighted_reward = max(-1.0, min(1.0, weighted_reward))

        self._total_scored += 1

        return Score(
            problem_id=problem.problem_id,
            candidate_id=problem.problem_id,
            reward=round(weighted_reward, 4),
            metrics={
                "structure_reward": reward_record.render_reward,
                "visual_reward": reward_record.visual_reward,
                "fidelity_reward": round(fidelity, 4),
                "diagnostic_penalty": reward_record.diagnostic_penalty,
                "hacking_penalty": reward_record.hacking_penalty,
                "overflow_score": render_result.diagnostics.overflow_score,
            },
            is_valid=True,
            feedback="OK",
        )

    async def score_batch(
        self, problems: list[Problem], responses: list[str]
    ) -> list[Score]:
        """Score a batch of responses concurrently."""
        tasks = [self.score(p, r) for p, r in zip(problems, responses)]
        return await asyncio.gather(*tasks)

    def compute_group_advantages(
        self, problems: list[Problem], scores: list[Score]
    ) -> list[RolloutGroup]:
        """Group scores by group_id and compute GRPO advantages."""
        groups: dict[str, tuple[list[Problem], list[Score]]] = {}
        for p, s in zip(problems, scores):
            gid = p.group_id
            if gid not in groups:
                groups[gid] = ([], [])
            groups[gid][0].append(p)
            groups[gid][1].append(s)

        result: list[RolloutGroup] = []
        for gid, (probs, scrs) in groups.items():
            rewards = [s.reward for s in scrs]
            advantages, skipped, skip_reason = compute_advantages(rewards)
            result.append(
                RolloutGroup(
                    group_id=gid,
                    task=probs[0].task,
                    problems=probs,
                    scores=scrs,
                    advantages=advantages,
                    skipped=skipped,
                    skip_reason=skip_reason,
                )
            )
        return result

    async def cleanup(self) -> None:
        """Cleanup resources."""
        close = getattr(self._renderer, "close", None)
        if callable(close):
            close()
        logger.info(
            "HtmlFormattingEnv cleanup: scored %d candidates across %d epochs",
            self._total_scored,
            self._epoch + 1,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_tasks": len(self.tasks),
            "total_scored": self._total_scored,
            "current_epoch": self._epoch,
            "task_index": self._task_index,
            "reward_weights": self._weights,
        }
