from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path

from .artifacts import write_candidate_artifacts
from .schemas import (
    ArtifactRecord,
    HtmlCandidate,
    RenderDiagnostics,
    RenderResult,
    TaskSpec,
)
from .text_utils import count_tag_occurrences, extract_visible_text


def _heuristic_diagnostics(
    html: str, visible_text: str, fallback_used: bool
) -> RenderDiagnostics:
    html_bytes = len(html.encode("utf-8"))
    word_count = len(visible_text.split())
    dom_nodes = sum(
        count_tag_occurrences(html, tag)
        for tag in ["div", "section", "p", "li", "svg", "table", "tr", "td"]
    )
    overflow_score = min(
        1.0,
        max(
            (word_count - 120) / 300.0,
            (dom_nodes - 80) / 300.0,
            (html_bytes - 30000) / 120000.0,
            0.0,
        ),
    )
    tiny_text_count = 3 if "font-size:12px" in html or "font-size: 12px" in html else 0
    blank = len(visible_text.strip()) < 8 and "<svg" not in html.lower()
    return RenderDiagnostics(
        blank_page=blank,
        horizontal_overflow_px=int(1920 * overflow_score * 0.6),
        vertical_overflow_px=int(1080 * overflow_score),
        overflow_score=round(overflow_score, 4),
        clipped_element_count=2 if overflow_score > 0.18 else 0,
        tiny_text_count=tiny_text_count,
        contrast_failure_count=0,
        dom_node_count=dom_nodes,
        html_byte_size=html_bytes,
        visible_text_length=len(visible_text),
        non_white_pixel_ratio=min(1.0, 0.1 + len(visible_text) / 1200.0),
        fallback_used=fallback_used,
    )


def _repair_hint(error_type: str | None) -> str | None:
    hints = {
        "blank_render": "Add visible content inside the slide root and avoid empty placeholders.",
        "severe_overflow": "Reduce content density, font count, or card count to fit the viewport.",
        "validator_failure": "Remove forbidden constructs and keep the HTML inside the constrained subset.",
    }
    if error_type is None:
        return None
    return hints.get(error_type)


@dataclass
class FallbackRenderer:
    renderer_version: str = "fallback_renderer_v1"

    def render(
        self, task: TaskSpec, candidate: HtmlCandidate, output_dir: Path
    ) -> tuple[RenderResult, ArtifactRecord]:
        started = time.time()
        visible_text = extract_visible_text(candidate.html)
        diagnostics = _heuristic_diagnostics(
            candidate.html, visible_text, fallback_used=True
        )
        error_type = None
        status = "success"
        if diagnostics.blank_page:
            status = "failure"
            error_type = "blank_render"
        elif diagnostics.overflow_score >= 0.22:
            status = "failure"
            error_type = "severe_overflow"
        artifact = write_candidate_artifacts(
            output_dir,
            candidate.candidate_id,
            candidate.html,
            visible_text,
            screenshot_bytes=None,
        )
        result = RenderResult(
            candidate_id=candidate.candidate_id,
            render_status=status,
            error_type=error_type,
            render_time_ms=int((time.time() - started) * 1000),
            screenshot_ready=True,
            diagnostics=diagnostics,
            repair_hint=_repair_hint(error_type),
        )
        return result, artifact


@dataclass
class PlaywrightRenderer:
    renderer_version: str = "playwright_renderer_v1"

    def render(
        self, task: TaskSpec, candidate: HtmlCandidate, output_dir: Path
    ) -> tuple[RenderResult, ArtifactRecord]:
        try:
            sync_api = importlib.import_module("playwright.sync_api")
            PlaywrightTimeoutError = getattr(sync_api, "TimeoutError")
            sync_playwright = getattr(sync_api, "sync_playwright")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Playwright renderer requires the playwright package"
            ) from exc

        started = time.time()
        screenshot_bytes: bytes | None = None
        visible_text = ""
        console_errors: list[str] = []
        page_errors: list[str] = []
        failed_requests: list[str] = []
        network_violations: list[str] = []
        error_type: str | None = None
        status = "success"
        diagnostics = RenderDiagnostics()

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={
                        "width": task.constraints.viewport.width,
                        "height": task.constraints.viewport.height,
                    },
                    locale="en-US",
                    timezone_id="UTC",
                    java_script_enabled=False,
                )
                page = context.new_page()

                def on_console(msg):
                    if msg.type == "error":
                        console_errors.append(msg.text)

                def on_page_error(err):
                    page_errors.append(str(err))

                page.on("console", on_console)
                page.on("pageerror", on_page_error)
                page.route("**/*", lambda route: route.abort())
                page.set_content(
                    candidate.html, wait_until="domcontentloaded", timeout=5000
                )
                page.wait_for_timeout(300)
                screenshot_bytes = page.screenshot(type="png")
                visible_text = page.locator("body").inner_text(timeout=1000)
                metrics = page.evaluate(
                    """
                    () => {
                      const body = document.body;
                      const root = document.documentElement;
                      return {
                        scrollWidth: Math.max(body.scrollWidth, root.scrollWidth),
                        clientWidth: root.clientWidth,
                        scrollHeight: Math.max(body.scrollHeight, root.scrollHeight),
                        clientHeight: root.clientHeight,
                        domNodeCount: document.querySelectorAll('*').length,
                        textLength: (body.innerText || '').length
                      };
                    }
                    """
                )
                browser.close()
                overflow_x = max(
                    0, int(metrics["scrollWidth"] - metrics["clientWidth"])
                )
                overflow_y = max(
                    0, int(metrics["scrollHeight"] - metrics["clientHeight"])
                )
                overflow_score = max(
                    overflow_x / max(1, task.constraints.viewport.width),
                    overflow_y / max(1, task.constraints.viewport.height),
                )
                blank_page = len(visible_text.strip()) < 8
                diagnostics = RenderDiagnostics(
                    blank_page=blank_page,
                    horizontal_overflow_px=overflow_x,
                    vertical_overflow_px=overflow_y,
                    overflow_score=round(float(min(1.0, overflow_score)), 4),
                    clipped_element_count=1 if overflow_score > 0.1 else 0,
                    tiny_text_count=0,
                    contrast_failure_count=0,
                    dom_node_count=int(metrics["domNodeCount"]),
                    html_byte_size=len(candidate.html.encode("utf-8")),
                    visible_text_length=int(metrics["textLength"]),
                    non_white_pixel_ratio=0.5,
                    fallback_used=False,
                )
                if blank_page:
                    status = "failure"
                    error_type = "blank_render"
                elif diagnostics.overflow_score >= 0.2:
                    status = "failure"
                    error_type = "severe_overflow"
                elif console_errors or page_errors:
                    status = "failure"
                    error_type = "js_exception"
        except PlaywrightTimeoutError:
            status = "failure"
            error_type = "timeout"
        except Exception as exc:  # pragma: no cover - runtime dependent
            status = "failure"
            error_type = error_type or "unknown_render_failure"
            page_errors.append(str(exc))

        artifact = write_candidate_artifacts(
            output_dir,
            candidate.candidate_id,
            candidate.html,
            visible_text,
            screenshot_bytes=screenshot_bytes,
        )
        result = RenderResult(
            candidate_id=candidate.candidate_id,
            render_status=status,
            error_type=error_type,
            traceback="\n".join(page_errors) if page_errors else None,
            console_errors=console_errors,
            page_errors=page_errors,
            failed_requests=failed_requests,
            network_violations=network_violations,
            render_time_ms=int((time.time() - started) * 1000),
            screenshot_ready=artifact.screenshot_path is not None,
            diagnostics=diagnostics,
            repair_hint=_repair_hint(error_type),
        )
        return result, artifact


def build_renderer(kind: str):
    return PlaywrightRenderer() if kind == "playwright" else FallbackRenderer()
