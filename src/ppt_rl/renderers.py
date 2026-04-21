from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def _finalize_result(
    output_dir: Path,
    candidate: HtmlCandidate,
    started: float,
    visible_text: str,
    screenshot_bytes: bytes | None,
    fullpage_screenshot_bytes: bytes | None,
    page_screenshots: list[tuple[str, bytes]],
    diagnostics: RenderDiagnostics,
    error_type: str | None,
    page_errors: list[str] | None = None,
    console_errors: list[str] | None = None,
    failed_requests: list[str] | None = None,
    network_violations: list[str] | None = None,
) -> tuple[RenderResult, ArtifactRecord]:
    artifact = write_candidate_artifacts(
        output_dir,
        candidate.candidate_id,
        candidate.html,
        visible_text,
        screenshot_bytes=screenshot_bytes,
        fullpage_screenshot_bytes=fullpage_screenshot_bytes,
        page_screenshots=page_screenshots,
    )
    result = RenderResult(
        candidate_id=candidate.candidate_id,
        render_status="failure" if error_type is not None else "success",
        error_type=error_type,
        traceback="\n".join(page_errors) if page_errors else None,
        console_errors=console_errors or [],
        page_errors=page_errors or [],
        failed_requests=failed_requests or [],
        network_violations=network_violations or [],
        render_time_ms=int((time.time() - started) * 1000),
        screenshot_ready=artifact.screenshot_path is not None,
        diagnostics=diagnostics,
        repair_hint=_repair_hint(error_type),
    )
    return result, artifact


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
        if diagnostics.blank_page:
            error_type = "blank_render"
        elif diagnostics.overflow_score >= 0.22:
            error_type = "severe_overflow"
        return _finalize_result(
            output_dir=output_dir,
            candidate=candidate,
            started=started,
            visible_text=visible_text,
            screenshot_bytes=None,
            fullpage_screenshot_bytes=None,
            page_screenshots=[],
            diagnostics=diagnostics,
            error_type=error_type,
        )


def _capture_deck_page_screenshots(
    page: Any, candidate_id: str, viewport_width: int, viewport_height: int
) -> list[tuple[str, bytes]]:
    selector = None
    for candidate_selector in [
        "section.slide",
        "[data-slide]",
        "[data-page]",
        "section.page",
        "section",
    ]:
        if page.locator(candidate_selector).count() > 1:
            selector = candidate_selector
            break

    if selector is None:
        return []

    total = page.locator(selector).count()
    screenshots: list[tuple[str, bytes]] = []
    for index in range(total):
        page.evaluate(
            """
            ({ selector, index, viewportWidth, viewportHeight }) => {
              const nodes = Array.from(document.querySelectorAll(selector));
              nodes.forEach((node, nodeIndex) => {
                const active = nodeIndex === index;
                node.style.setProperty('position', 'absolute', 'important');
                node.style.setProperty('left', '0px', 'important');
                node.style.setProperty('top', '0px', 'important');
                node.style.setProperty('width', `${viewportWidth}px`, 'important');
                node.style.setProperty('height', `${viewportHeight}px`, 'important');
                node.style.setProperty('visibility', active ? 'visible' : 'hidden', 'important');
                node.style.setProperty('opacity', active ? '1' : '0', 'important');
                node.style.setProperty('pointer-events', active ? 'auto' : 'none', 'important');
                node.style.setProperty('z-index', active ? '2' : '1', 'important');
              });
            }
            """,
            {
                "selector": selector,
                "index": index,
                "viewportWidth": viewport_width,
                "viewportHeight": viewport_height,
            },
        )
        page.wait_for_timeout(50)
        screenshots.append(
            (
                f"{candidate_id}.page_{index + 1:03d}.png",
                page.screenshot(type="png"),
            )
        )
    return screenshots


@dataclass
class PlaywrightRenderer:
    renderer_version: str = "playwright_renderer_v2"
    _playwright: Any = field(default=None, init=False, repr=False)
    _browser: Any = field(default=None, init=False, repr=False)

    def _ensure_browser(self) -> tuple[Any, type[Exception]]:
        if self._browser is not None:
            sync_api = importlib.import_module("playwright.sync_api")
            return self._browser, getattr(sync_api, "TimeoutError")

        try:
            sync_api = importlib.import_module("playwright.sync_api")
            sync_playwright = getattr(sync_api, "sync_playwright")
            timeout_error = getattr(sync_api, "TimeoutError")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Playwright renderer requires the playwright package"
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        return self._browser, timeout_error

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def render(
        self, task: TaskSpec, candidate: HtmlCandidate, output_dir: Path
    ) -> tuple[RenderResult, ArtifactRecord]:
        browser, playwright_timeout_error = self._ensure_browser()

        started = time.time()
        screenshot_bytes: bytes | None = None
        fullpage_screenshot_bytes: bytes | None = None
        visible_text = ""
        console_errors: list[str] = []
        page_errors: list[str] = []
        page_screenshots: list[tuple[str, bytes]] = []
        failed_requests: list[str] = []
        network_violations: list[str] = []
        error_type: str | None = None
        diagnostics = RenderDiagnostics()

        try:
            context = browser.new_context(
                viewport={
                    "width": task.constraints.viewport.width,
                    "height": task.constraints.viewport.height,
                },
                device_scale_factor=2,
                locale="en-US",
                timezone_id="UTC",
                java_script_enabled=True,
            )
            try:
                page = context.new_page()

                def on_console(msg: Any) -> None:
                    if msg.type == "error":
                        console_errors.append(msg.text)

                def on_page_error(err: Exception) -> None:
                    page_errors.append(str(err))

                page.on("console", on_console)
                page.on("pageerror", on_page_error)
                page.set_content(
                    candidate.html,
                    wait_until="networkidle",
                    timeout=5000,
                )
                page.wait_for_timeout(300)
                screenshot_bytes = page.screenshot(type="png")
                fullpage_screenshot_bytes = page.screenshot(type="png", full_page=True)
                visible_text = page.locator("body").inner_text(timeout=1000)
                page_screenshots = _capture_deck_page_screenshots(
                    page,
                    candidate.candidate_id,
                    task.constraints.viewport.width,
                    task.constraints.viewport.height,
                )
                if not page_screenshots and screenshot_bytes is not None:
                    page_screenshots = [
                        (f"{candidate.candidate_id}.page_001.png", screenshot_bytes)
                    ]
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
            finally:
                context.close()

            overflow_x = max(0, int(metrics["scrollWidth"] - metrics["clientWidth"]))
            overflow_y = max(0, int(metrics["scrollHeight"] - metrics["clientHeight"]))
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
                error_type = "blank_render"
            elif diagnostics.overflow_score >= 0.2:
                error_type = "severe_overflow"
            elif console_errors or page_errors:
                error_type = "js_exception"
        except playwright_timeout_error:
            error_type = "timeout"
        except Exception as exc:  # pragma: no cover - runtime dependent
            error_type = error_type or "unknown_render_failure"
            page_errors.append(str(exc))

        return _finalize_result(
            output_dir=output_dir,
            candidate=candidate,
            started=started,
            visible_text=visible_text,
            screenshot_bytes=screenshot_bytes,
            fullpage_screenshot_bytes=fullpage_screenshot_bytes,
            page_screenshots=page_screenshots,
            diagnostics=diagnostics,
            error_type=error_type,
            page_errors=page_errors,
            console_errors=console_errors,
            failed_requests=failed_requests,
            network_violations=network_violations,
        )


def build_renderer(kind: str):
    if kind == "playwright":
        return PlaywrightRenderer()
    return FallbackRenderer()
