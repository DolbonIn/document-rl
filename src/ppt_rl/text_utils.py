from __future__ import annotations

import re
from html.parser import HTMLParser


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                self.parts.append(text)


def extract_visible_text(html: str) -> str:
    parser = VisibleTextParser()
    parser.feed(html)
    return " ".join(parser.parts)


def count_tag_occurrences(html: str, tag: str) -> int:
    return len(re.findall(rf"<\s*{re.escape(tag)}\b", html, flags=re.IGNORECASE))
