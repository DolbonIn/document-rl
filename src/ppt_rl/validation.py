from __future__ import annotations

import re

from .schemas import HtmlCandidate, ValidationIssue, ValidatorResult

ALLOWED_TAGS = {
    "html",
    "head",
    "meta",
    "title",
    "style",
    "body",
    "main",
    "section",
    "article",
    "div",
    "h1",
    "h2",
    "h3",
    "p",
    "span",
    "strong",
    "em",
    "ul",
    "ol",
    "li",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "svg",
    "path",
    "rect",
    "circle",
    "line",
    "polyline",
    "text",
    "g",
    "img",
}
FORBIDDEN_TAGS = {"script", "iframe", "object", "embed", "video", "audio", "form"}
FORBIDDEN_ATTR_PATTERNS = [
    r"\bon[a-z]+\s*=",
    r"javascript:",
    r"@import",
    r"https?://",
    r"//fonts\.",
]
REQUIRED_ROOT = r"<main\b[^>]*class=\"slide\""
MAX_HTML_BYTES = 120_000


def _find_tags(html: str) -> list[str]:
    return [match.lower() for match in re.findall(r"<\s*([a-zA-Z0-9]+)\b", html)]


def validate_candidate(candidate: HtmlCandidate) -> ValidatorResult:
    html = candidate.html
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    for tag in _find_tags(html):
        if tag in FORBIDDEN_TAGS:
            errors.append(
                ValidationIssue(
                    type="forbidden_tag",
                    message=f"Tag {tag} is forbidden.",
                    selector=tag,
                )
            )
        elif tag not in ALLOWED_TAGS:
            warnings.append(
                ValidationIssue(
                    type="unknown_tag",
                    message=f"Tag {tag} is outside the constrained subset.",
                    selector=tag,
                )
            )

    for pattern in FORBIDDEN_ATTR_PATTERNS:
        if re.search(pattern, html, flags=re.IGNORECASE):
            errors.append(
                ValidationIssue(
                    type="forbidden_construct",
                    message=f"Matched forbidden pattern: {pattern}",
                )
            )

    if not re.search(REQUIRED_ROOT, html, flags=re.IGNORECASE):
        errors.append(
            ValidationIssue(
                type="missing_root_container",
                message='Required <main class="slide"> root is missing.',
            )
        )

    if len(html.encode("utf-8")) > MAX_HTML_BYTES:
        errors.append(
            ValidationIssue(
                type="html_too_large", message="HTML exceeds max byte size limit."
            )
        )
    elif len(html.encode("utf-8")) > MAX_HTML_BYTES * 0.8:
        warnings.append(
            ValidationIssue(
                type="large_html", message="HTML is near the byte size limit."
            )
        )

    status = "fail" if errors else "pass"
    error_type = errors[0].type if errors else None
    return ValidatorResult(
        candidate_id=candidate.candidate_id,
        validator_status=status,
        error_type=error_type,
        errors=errors,
        warnings=warnings,
    )
