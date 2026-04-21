from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .exporters import write_run_outputs
from .judge import pairwise_visual_rewards
from .policy_sampler import build_sampler
from .renderers import build_renderer
from .rewards import aggregate_reward, compute_advantages, compute_fidelity
from .schemas import (
    ArtifactRecord,
    HtmlCandidate,
    RewardRecord,
    SamplingParams,
    TaskSpec,
    TrainerRecord,
    TrajectoryGroup,
    ValidatorResult,
)
from .task_sampler import load_tasks
from .validation import validate_candidate


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    sampler: str = "stub"
    model_path: str | None = None
    num_candidates: int = 4
    seed: int = 42
    renderer: str = "fallback"
    device: str = "cpu"
    max_new_tokens: int = 768


def run_pipeline(config: PipelineConfig) -> dict:
    tasks = load_tasks(config.input_path)
    sampler = build_sampler(config.sampler, config.model_path, config.device)
    renderer = build_renderer(config.renderer)
    all_candidates: list[HtmlCandidate] = []
    all_validators: list[ValidatorResult] = []
    all_renders = []
    all_artifacts: list[ArtifactRecord] = []
    all_judges = []
    all_rewards: list[RewardRecord] = []
    all_groups: list[TrajectoryGroup] = []
    all_trainer_records: list[TrainerRecord] = []

    artifacts_dir = config.output_dir / "artifacts"

    for task_index, task in enumerate(tasks, start=1):
        group_id = f"group_{task.task_id}"
        sampling_params = SamplingParams(
            seed=config.seed + task_index, max_new_tokens=config.max_new_tokens
        )
        candidates = sampler.sample(
            task,
            group_id=group_id,
            num_candidates=config.num_candidates,
            sampling_params=sampling_params,
        )
        all_candidates.extend(candidates)
        render_map = {}
        artifact_map = {}
        fidelity_map = {}
        reward_map = {}
        successful_ids: list[str] = []

        for candidate in candidates:
            validator_result = validate_candidate(candidate)
            all_validators.append(validator_result)
            if validator_result.failed:
                reward = aggregate_reward(
                    candidate_id=candidate.candidate_id,
                    failure_type="validator_failure",
                    visual_reward=0.0,
                    fidelity_reward=0.0,
                    overflow_score=0.0,
                    tiny_text_count=0,
                    contrast_failure_count=0,
                    clipped_element_count=0,
                    artifact=None,
                )
                reward_map[candidate.candidate_id] = reward
                all_rewards.append(reward)
                continue

            task_artifact_dir = artifacts_dir / task.task_id
            render_result, artifact = renderer.render(
                task, candidate, task_artifact_dir
            )
            all_renders.append(render_result)
            all_artifacts.append(artifact)
            render_map[candidate.candidate_id] = render_result
            artifact_map[candidate.candidate_id] = artifact
            fidelity = compute_fidelity(task, artifact)
            fidelity_map[candidate.candidate_id] = fidelity
            if not render_result.failed:
                successful_ids.append(candidate.candidate_id)

        judges, visual_rewards = pairwise_visual_rewards(
            task, successful_ids, render_map, artifact_map, fidelity_map
        )
        all_judges.extend(judges)

        for candidate in candidates:
            if candidate.candidate_id in reward_map:
                continue
            render_result = render_map[candidate.candidate_id]
            artifact = artifact_map[candidate.candidate_id]
            reward = aggregate_reward(
                candidate_id=candidate.candidate_id,
                failure_type=render_result.error_type,
                visual_reward=visual_rewards.get(candidate.candidate_id, 0.0),
                fidelity_reward=fidelity_map.get(candidate.candidate_id, 0.0),
                overflow_score=render_result.diagnostics.overflow_score,
                tiny_text_count=render_result.diagnostics.tiny_text_count,
                contrast_failure_count=render_result.diagnostics.contrast_failure_count,
                clipped_element_count=render_result.diagnostics.clipped_element_count,
                artifact=artifact,
            )
            reward_map[candidate.candidate_id] = reward
            all_rewards.append(reward)

        ordered_rewards = [
            reward_map[candidate.candidate_id].total_reward for candidate in candidates
        ]
        advantages, skipped, skip_reason = compute_advantages(ordered_rewards)
        group = TrajectoryGroup(
            group_id=group_id,
            task_id=task.task_id,
            candidate_ids=[candidate.candidate_id for candidate in candidates],
            rewards=ordered_rewards,
            advantages=advantages,
            skipped=skipped,
            skip_reason=skip_reason,
        )
        all_groups.append(group)
        for candidate, advantage in zip(candidates, advantages):
            reward = reward_map[candidate.candidate_id]
            all_trainer_records.append(
                TrainerRecord(
                    record_id=f"rollout_{candidate.candidate_id}",
                    group_id=group_id,
                    task_id=task.task_id,
                    prompt=candidate.prompt,
                    completion=candidate.html,
                    prompt_tokens=[],
                    completion_tokens=candidate.completion_tokens,
                    completion_mask=candidate.completion_mask,
                    old_logprobs=candidate.old_logprobs,
                    ref_logprobs=candidate.ref_logprobs,
                    reward=reward.total_reward,
                    advantage=advantage,
                    kl=0.0,
                    model_version=candidate.model_version,
                    sampling_params=candidate.sampling_params.to_dict(),
                    reward_config_version="reward_v1.0",
                    renderer_version=getattr(
                        renderer, "renderer_version", "renderer_unknown"
                    ),
                    browser_version="chromium_pinned"
                    if config.renderer == "playwright"
                    else "fallback",
                    font_pack_version="default",
                    judge_prompt_version="heuristic_judge_v1",
                )
            )

    write_run_outputs(
        config.output_dir,
        tasks,
        all_candidates,
        all_validators,
        all_renders,
        all_artifacts,
        all_judges,
        all_rewards,
        all_groups,
        all_trainer_records,
    )
    return {
        "tasks": len(tasks),
        "candidates": len(all_candidates),
        "successful_renders": sum(
            1 for reward in all_rewards if reward.failure_type is None
        ),
        "groups": len(all_groups),
        "trainer_records": len(all_trainer_records),
        "output_dir": str(config.output_dir),
    }
