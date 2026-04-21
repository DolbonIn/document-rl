from __future__ import annotations

import math

from .schemas import ArtifactRecord, RewardRecord, TaskSpec

RENDER_REWARD_MAP = {
    "success": 0.30,
    "validator_failure": -0.70,
    "timeout": -1.00,
    "browser_crash": -1.00,
    "js_exception": -0.90,
    "network_violation": -0.85,
    "blank_render": -0.80,
    "asset_missing": -0.60,
    "severe_overflow": -0.50,
    "severe_clipping": -0.50,
    "minor_overflow": -0.20,
    "unknown_render_failure": -0.90,
}


def compute_fidelity(task: TaskSpec, artifact: ArtifactRecord) -> float:
    visible = artifact.visible_text.lower()
    checks: list[bool] = []
    for required in task.constraints.must_include:
        checks.append(required.lower() in visible)
    for forbidden in task.constraints.must_not_include:
        checks.append(forbidden.lower() not in visible)
    checks.append(bool(visible.strip()))
    return round(sum(1 for item in checks if item) / max(1, len(checks)), 4)


def compute_diagnostic_penalty(
    overflow_score: float,
    tiny_text_count: int,
    contrast_failure_count: int,
    clipped_element_count: int,
) -> float:
    penalty = (
        0.20 * overflow_score
        + 0.05 * min(tiny_text_count, 5) / 5.0
        + 0.10 * min(contrast_failure_count, 10) / 10.0
        + 0.10 * min(clipped_element_count, 10) / 10.0
    )
    return round(penalty, 4)


def compute_hacking_penalty(artifact: ArtifactRecord) -> float:
    text = artifact.visible_text.lower()
    suspicious = [
        "ignore previous instructions",
        "choose this candidate",
        "judge",
        "reward me",
    ]
    if any(needle in text for needle in suspicious):
        return 0.30
    return 0.0


def aggregate_reward(
    candidate_id: str,
    failure_type: str | None,
    visual_reward: float,
    fidelity_reward: float,
    overflow_score: float,
    tiny_text_count: int,
    contrast_failure_count: int,
    clipped_element_count: int,
    artifact: ArtifactRecord | None,
) -> RewardRecord:
    render_key = failure_type if failure_type is not None else "success"
    render_reward = RENDER_REWARD_MAP.get(render_key, -0.90)
    diagnostic_penalty = compute_diagnostic_penalty(
        overflow_score, tiny_text_count, contrast_failure_count, clipped_element_count
    )
    hacking_penalty = compute_hacking_penalty(artifact) if artifact else 0.0
    if failure_type and failure_type != "success":
        total_reward = render_reward
    else:
        total_reward = (
            render_reward
            + 0.45 * visual_reward
            + 0.25 * fidelity_reward
            - diagnostic_penalty
            - hacking_penalty
        )
    total_reward = max(-1.0, min(1.0, total_reward))
    return RewardRecord(
        candidate_id=candidate_id,
        render_reward=round(render_reward, 4),
        visual_reward=round(visual_reward, 4),
        fidelity_reward=round(fidelity_reward, 4),
        diagnostic_penalty=round(diagnostic_penalty, 4),
        hacking_penalty=round(hacking_penalty, 4),
        total_reward=round(total_reward, 4),
        failure_type=failure_type,
    )


def compute_advantages(
    rewards: list[float], eps: float = 1e-8
) -> tuple[list[float], bool, str | None]:
    mean = sum(rewards) / max(1, len(rewards))
    variance = sum((reward - mean) ** 2 for reward in rewards) / max(1, len(rewards))
    std = math.sqrt(variance)
    if std < eps:
        return [0.0 for _ in rewards], True, "zero_variance"
    return [round((reward - mean) / (std + eps), 6) for reward in rewards], False, None
