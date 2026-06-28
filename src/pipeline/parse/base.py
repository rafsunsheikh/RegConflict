"""Common parser primitives. Each parser turns cleaned page text into Blocks.

A Block is the structural unit before chunking. The chunker may split a long
Block (>CHUNK_MAX_TOKENS) but must never merge across Block boundaries unless
the resulting size is still <= target.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Block:
    text: str
    citation: str           # e.g. "Article 4(1)", "RG 165.21", "s.5(1)(a)", "Clause 3"
    kind: str               # article | section | clause | paragraph | recital | annex | preamble | misc
    page_start: int | None = None
    page_end: int | None = None
    parent_citation: str | None = None
    extras: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return self.__dict__


class BaseParser:
    """Subclasses implement parse(text) -> list[Block].

    Input is the *joined* cleaned text produced by clean.joined_text(),
    which embeds ``\f<PAGE n>\f`` sentinels. Parsers may use these to assign
    page_start/page_end to each block.
    """

    def parse(self, text: str) -> list[Block]:  # pragma: no cover - interface
        raise NotImplementedError


# Helpers for page-tracking via sentinels.
_PAGE_SENTINEL_RE = re.compile(r"\f<PAGE\s+(\d+)>\f")


def page_at_offset(text: str, offset: int) -> int | None:
    """Return the page number whose sentinel immediately precedes offset."""
    last = None
    for m in _PAGE_SENTINEL_RE.finditer(text, 0, offset):
        last = int(m.group(1))
    return last


def strip_sentinels(text: str) -> str:
    return _PAGE_SENTINEL_RE.sub("", text).strip()


def split_on_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, int]]:
    """Split text on every match of pattern. Returns list of (segment, start_offset).

    The header that triggered the split is included at the start of each segment
    (except possibly the first preamble segment, which has no preceding header).
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return [(text, 0)]
    segments: list[tuple[str, int]] = []
    first_start = matches[0].start()
    if first_start > 0:
        segments.append((text[:first_start], 0))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append((text[m.start():end], m.start()))
    return segments
