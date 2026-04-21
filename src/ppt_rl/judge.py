from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .schemas import ArtifactRecord, JudgeResult, RenderResult, RewardRecord, TaskSpec


@dataclass
class HeuristicPairwiseJudge:
    judge_prompt_version: str = "heuristic_judge_v1"

    def compare(
        self,
        task: TaskSpec,
        pair_index: int,
        candidate_a: str,
        candidate_b: str,
        render_a: RenderResult,
        render_b: RenderResult,
        artifact_a: ArtifactRecord,
        artifact_b: ArtifactRecord,
        fidelity_a: float,
        fidelity_b: float,
    ) -> JudgeResult:
        score_a = (
            (1.0 - render_a.diagnostics.overflow_score)
            + fidelity_a
            + min(render_a.diagnostics.visible_text_length, 400) / 400.0
        )
        score_b = (
            (1.0 - render_b.diagnostics.overflow_score)
            + fidelity_b
            + min(render_b.diagnostics.visible_text_length, 400) / 400.0
        )
        if score_a == score_b:
            winner = None
            confidence = 0.5
            reason = "Both candidates are equivalent under the heuristic judge."
        else:
            winner = candidate_a if score_a > score_b else candidate_b
            delta = abs(score_a - score_b)
            confidence = max(0.51, min(0.99, 0.5 + delta / 4.0))
            reason = f"{winner} has better readability-density balance and lower overflow under the heuristic judge."
        return JudgeResult(
            judge_id=f"judge_{task.task_id}_{pair_index}",
            task_id=task.task_id,
            mode="pairwise",
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            winner=winner,
            confidence=round(confidence, 4),
            rubric_scores={
                "readability": {
                    "a": round((1.0 - render_a.diagnostics.overflow_score) * 10, 2),
                    "b": round((1.0 - render_b.diagnostics.overflow_score) * 10, 2),
                },
                "prompt_fidelity": {
                    "a": round(fidelity_a * 10, 2),
                    "b": round(fidelity_b * 10, 2),
                },
            },
            reason=reason,
        )


def pairwise_visual_rewards(
    task: TaskSpec,
    successful_ids: list[str],
    render_map: dict[str, RenderResult],
    artifact_map: dict[str, ArtifactRecord],
    fidelity_map: dict[str, float],
) -> tuple[list[JudgeResult], dict[str, float]]:
    if len(successful_ids) < 2:
        return [], {candidate_id: 0.5 for candidate_id in successful_ids}
    judge = HeuristicPairwiseJudge()
    wins = {candidate_id: 0.0 for candidate_id in successful_ids}
    comparisons = {candidate_id: 0.0 for candidate_id in successful_ids}
    results: list[JudgeResult] = []
    for idx, (a, b) in enumerate(combinations(successful_ids, 2), start=1):
        result = judge.compare(
            task,
            idx,
            a,
            b,
            render_map[a],
            render_map[b],
            artifact_map[a],
            artifact_map[b],
            fidelity_map[a],
            fidelity_map[b],
        )
        results.append(result)
        comparisons[a] += 1
        comparisons[b] += 1
        if result.winner is None:
            wins[a] += 0.5 * result.confidence
            wins[b] += 0.5 * result.confidence
        else:
            wins[result.winner] += result.confidence
    visual = {
        candidate_id: round(wins[candidate_id] / comparisons[candidate_id], 4)
        for candidate_id in successful_ids
    }
    return results, visual
