"""Fallback parser - looks for the most common heading shapes and splits on them.

Used for regulators whose documents do not follow a known formal numbering
convention (consultation papers, narrative guidance, FATF recommendations as
prose, etc.). Citation strings here are best-effort.
"""
from __future__ import annotations

import re

from .base import BaseParser, Block, page_at_offset, strip_sentinels, split_on_pattern


# Heading shapes, ordered: try strongest first.
_HEADING_PATTERNS = [
    # "PART I - GENERAL", "Part 4 Application"
    re.compile(r"^\s*PART\s+[IVXLC0-9]+\b[^\n]*", re.IGNORECASE | re.MULTILINE),
    # "Chapter 3", "CHAPTER 12 - Reporting"
    re.compile(r"^\s*Chapter\s+[IVXLC0-9]+\b[^\n]*", re.IGNORECASE | re.MULTILINE),
    # "Section 4", "Section 4.2"
    re.compile(r"^\s*Section\s+\d+(?:\.\d+)?\b[^\n]*", re.IGNORECASE | re.MULTILINE),
    # Numbered headings: "4. Application" / "4.1 Scope" / "4.1.2 ..."
    re.compile(r"^\s*\d+(?:\.\d+){0,3}\.?\s+[A-Z][^\n]{1,160}$", re.MULTILINE),
]


class GenericParser(BaseParser):
    def parse(self, text: str) -> list[Block]:
        # Try each pattern in order; the first that yields more than one segment wins.
        chosen_pattern = None
        for pat in _HEADING_PATTERNS:
            if len(list(pat.finditer(text))) >= 2:
                chosen_pattern = pat
                break

        if chosen_pattern is None:
            # No structural markers detected - one big block, let the chunker carve it.
            cleaned = strip_sentinels(text)
            if not cleaned:
                return []
            return [Block(text=cleaned, citation="(no structure)", kind="misc",
                          page_start=page_at_offset(text, 0),
                          page_end=page_at_offset(text, len(text)))]

        blocks: list[Block] = []
        segments = split_on_pattern(text, chosen_pattern)
        for seg_text, offset in segments:
            stripped = strip_sentinels(seg_text)
            if not stripped:
                continue
            first_line = stripped.split("\n", 1)[0].strip()[:160]
            blocks.append(Block(
                text=stripped,
                citation=first_line,
                kind="section",
                page_start=page_at_offset(text, offset),
                page_end=page_at_offset(text, offset + len(seg_text)),
            ))
        return blocks
