from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from .io_utils import ensure_dir
from .schemas import ArtifactRecord

PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAQAAAAAYLlVAAAAVUlEQVR4Ae3PAQ0AAAgDIN8/9K3hHFQgCjRt2rRp06ZNmzZt2rRp06ZNmzZt2rRp06ZNmzZt2rRp06ZNmzZt2rRp06ZNmzZt2rRp06ZNWwGUEwH/o0AelQAAAABJRU5ErkJggg=="
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_candidate_artifacts(
    output_dir: Path,
    candidate_id: str,
    html: str,
    visible_text: str,
    screenshot_bytes: bytes | None,
) -> ArtifactRecord:
    ensure_dir(output_dir)
    dom_path = output_dir / f"{candidate_id}.dom.html"
    screenshot_path = output_dir / f"{candidate_id}.viewport.png"
    dom_path.write_text(html, encoding="utf-8")
    payload = screenshot_bytes or PLACEHOLDER_PNG
    screenshot_path.write_bytes(payload)
    return ArtifactRecord(
        candidate_id=candidate_id,
        screenshot_path=str(screenshot_path),
        fullpage_screenshot_path=None,
        dom_path=str(dom_path),
        visible_text=visible_text,
        artifact_hashes={
            "html_sha256": sha256_bytes(html.encode("utf-8")),
            "screenshot_sha256": sha256_bytes(payload),
            "dom_sha256": sha256_bytes(html.encode("utf-8")),
        },
    )
