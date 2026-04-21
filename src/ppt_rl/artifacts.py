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
    fullpage_screenshot_bytes: bytes | None = None,
    page_screenshots: list[tuple[str, bytes]] | None = None,
) -> ArtifactRecord:
    ensure_dir(output_dir)
    dom_path = output_dir / f"{candidate_id}.dom.html"
    screenshot_path = output_dir / f"{candidate_id}.viewport.png"
    fullpage_screenshot_path = output_dir / f"{candidate_id}.fullpage.png"
    dom_path.write_text(html, encoding="utf-8")
    payload = screenshot_bytes or PLACEHOLDER_PNG
    screenshot_path.write_bytes(payload)
    if fullpage_screenshot_bytes is not None:
        fullpage_screenshot_path.write_bytes(fullpage_screenshot_bytes)

    page_screenshot_paths: list[str] = []
    for filename, image_bytes in page_screenshots or []:
        page_path = output_dir / filename
        page_path.write_bytes(image_bytes)
        page_screenshot_paths.append(str(page_path))

    artifact_hashes = {
        "html_sha256": sha256_bytes(html.encode("utf-8")),
        "screenshot_sha256": sha256_bytes(payload),
        "dom_sha256": sha256_bytes(html.encode("utf-8")),
    }
    if fullpage_screenshot_bytes is not None:
        artifact_hashes["fullpage_screenshot_sha256"] = sha256_bytes(
            fullpage_screenshot_bytes
        )
    for index, (_, image_bytes) in enumerate(page_screenshots or [], start=1):
        artifact_hashes[f"page_{index:03d}_screenshot_sha256"] = sha256_bytes(
            image_bytes
        )

    return ArtifactRecord(
        candidate_id=candidate_id,
        screenshot_path=str(screenshot_path),
        fullpage_screenshot_path=(
            str(fullpage_screenshot_path)
            if fullpage_screenshot_bytes is not None
            else None
        ),
        page_screenshot_paths=page_screenshot_paths,
        dom_path=str(dom_path),
        visible_text=visible_text,
        artifact_hashes=artifact_hashes,
    )
