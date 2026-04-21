from __future__ import annotations

from pathlib import Path

from .io_utils import read_jsonl
from .schemas import TaskSpec


def load_tasks(path: Path) -> list[TaskSpec]:
    return [TaskSpec.from_dict(item) for item in read_jsonl(path)]
