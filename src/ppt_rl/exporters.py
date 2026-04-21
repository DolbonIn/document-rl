from __future__ import annotations

from pathlib import Path

from .io_utils import write_jsonl
from .schemas import (
    ArtifactRecord,
    HtmlCandidate,
    JudgeResult,
    RenderResult,
    RewardRecord,
    TaskSpec,
    TrainerRecord,
    TrajectoryGroup,
    ValidatorResult,
)


def write_run_outputs(
    output_dir: Path,
    tasks: list[TaskSpec],
    candidates: list[HtmlCandidate],
    validator_results: list[ValidatorResult],
    render_results: list[RenderResult],
    artifacts: list[ArtifactRecord],
    judge_results: list[JudgeResult],
    rewards: list[RewardRecord],
    groups: list[TrajectoryGroup],
    trainer_records: list[TrainerRecord],
) -> None:
    write_jsonl(output_dir / "tasks.jsonl", [task.to_dict() for task in tasks])
    write_jsonl(
        output_dir / "candidates.jsonl", [item.to_dict() for item in candidates]
    )
    write_jsonl(
        output_dir / "validator_results.jsonl",
        [item.to_dict() for item in validator_results],
    )
    write_jsonl(
        output_dir / "render_results.jsonl", [item.to_dict() for item in render_results]
    )
    write_jsonl(output_dir / "artifacts.jsonl", [item.to_dict() for item in artifacts])
    write_jsonl(
        output_dir / "judge_results.jsonl", [item.to_dict() for item in judge_results]
    )
    write_jsonl(output_dir / "rewards.jsonl", [item.to_dict() for item in rewards])
    write_jsonl(
        output_dir / "trajectory_groups.jsonl", [item.to_dict() for item in groups]
    )
    write_jsonl(
        output_dir / "trainer_records.jsonl",
        [item.to_dict() for item in trainer_records],
    )
