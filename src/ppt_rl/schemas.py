from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def to_json_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_json_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_json_dict(item) for key, item in value.items()}
    return value


@dataclass
class Viewport:
    width: int = 1920
    height: int = 1080

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskConstraints:
    page_count: int = 1
    aspect_ratio: str = "16:9"
    viewport: Viewport = field(default_factory=Viewport)
    theme: str = "minimal"
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskConstraints":
        viewport = payload.get("viewport") or {}
        return cls(
            page_count=payload.get("page_count", 1),
            aspect_ratio=payload.get("aspect_ratio", "16:9"),
            viewport=Viewport(**viewport) if viewport else Viewport(),
            theme=payload.get("theme", "minimal"),
            must_include=list(payload.get("must_include", [])),
            must_not_include=list(payload.get("must_not_include", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["viewport"] = self.viewport.to_dict()
        return data


@dataclass
class TaskSpec:
    task_id: str
    prompt: str
    doc_type: str
    constraints: TaskConstraints
    difficulty: str = "easy"
    split: str = "train"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        return cls(
            task_id=payload["task_id"],
            prompt=payload["prompt"],
            doc_type=payload.get("doc_type", "slide"),
            constraints=TaskConstraints.from_dict(payload.get("constraints", {})),
            difficulty=payload.get("difficulty", "easy"),
            split=payload.get("split", "train"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["constraints"] = self.constraints.to_dict()
        return data


@dataclass
class SamplingParams:
    seed: int
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 768

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HtmlCandidate:
    candidate_id: str
    task_id: str
    group_id: str
    model_version: str
    output_format: str
    html: str
    sampling_params: SamplingParams
    prompt: str
    completion_tokens: list[int] = field(default_factory=list)
    completion_mask: list[int] = field(default_factory=list)
    old_logprobs: list[float] = field(default_factory=list)
    ref_logprobs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sampling_params"] = self.sampling_params.to_dict()
        return data


@dataclass
class ValidationIssue:
    type: str
    message: str
    selector: str | None = None
    attribute: str | None = None
    value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidatorResult:
    candidate_id: str
    validator_status: str
    error_type: str | None
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.validator_status == "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "validator_status": self.validator_status,
            "error_type": self.error_type,
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
        }


@dataclass
class RenderDiagnostics:
    blank_page: bool = False
    horizontal_overflow_px: int = 0
    vertical_overflow_px: int = 0
    overflow_score: float = 0.0
    clipped_element_count: int = 0
    tiny_text_count: int = 0
    contrast_failure_count: int = 0
    dom_node_count: int = 0
    html_byte_size: int = 0
    visible_text_length: int = 0
    non_white_pixel_ratio: float = 0.0
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RenderResult:
    candidate_id: str
    render_status: str
    error_type: str | None
    traceback: str | None = None
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    failed_requests: list[str] = field(default_factory=list)
    network_violations: list[str] = field(default_factory=list)
    render_time_ms: int = 0
    screenshot_ready: bool = False
    diagnostics: RenderDiagnostics = field(default_factory=RenderDiagnostics)
    repair_hint: str | None = None

    @property
    def failed(self) -> bool:
        return self.render_status != "success"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["diagnostics"] = self.diagnostics.to_dict()
        return data


@dataclass
class ArtifactRecord:
    candidate_id: str
    screenshot_path: str | None
    fullpage_screenshot_path: str | None
    dom_path: str | None
    visible_text: str
    artifact_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeResult:
    judge_id: str
    task_id: str
    mode: str
    candidate_a: str
    candidate_b: str
    winner: str | None
    confidence: float
    rubric_scores: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RewardRecord:
    candidate_id: str
    render_reward: float
    visual_reward: float
    fidelity_reward: float
    diagnostic_penalty: float
    hacking_penalty: float
    total_reward: float
    failure_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryGroup:
    group_id: str
    task_id: str
    candidate_ids: list[str]
    rewards: list[float]
    advantages: list[float]
    skipped: bool
    skip_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainerRecord:
    record_id: str
    group_id: str
    task_id: str
    prompt: str
    completion: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_mask: list[int]
    old_logprobs: list[float]
    ref_logprobs: list[float]
    reward: float
    advantage: float
    kl: float
    model_version: str
    sampling_params: dict[str, Any]
    reward_config_version: str
    renderer_version: str
    browser_version: str
    font_pack_version: str
    judge_prompt_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
