"""MAS Notices: clause numbering is the citation unit.

Typical clause shapes in MAS Notices/Guidelines:
    "1   Introduction"
    "2.1   Scope of this Notice"
    "3.2.1   ..."

The leftmost numeric token (e.g. "3.2.1") is the clause id. Where a clause has
sub-points like (a), (b), they remain inside the parent clause text - chunking
respects the parent clause boundary first.
"""
from __future__ import annotations

import re

from .base import BaseParser, Block, page_at_offset, strip_sentinels, split_on_pattern

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
# Negative lookahead rejects "30 June 2025"-style date lines as clause headers.
_CLAUSE_RE = re.compile(
    rf"^\s*(\d+(?:\.\d+){{0,3}})\s+(?!(?:{_MONTHS})\b)([A-Z][^\n]{{0,200}})$",
    re.MULTILINE,
)


class MASNoticeParser(BaseParser):
    def parse(self, text: str) -> list[Block]:
        matches = list(_CLAUSE_RE.finditer(text))
        if len(matches) < 2:
            # Doesn't look clause-numbered; degrade gracefully.
            from .generic import GenericParser
            return GenericParser().parse(text)

        blocks: list[Block] = []
        # Preamble before first clause.
        first_off = matches[0].start()
        preamble = strip_sentinels(text[:first_off])
        if preamble:
            blocks.append(Block(
                text=preamble, citation="Preamble", kind="preamble",
                page_start=page_at_offset(text, 0),
                page_end=page_at_offset(text, first_off),
            ))

        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            seg = text[m.start():end]
            clause_id = m.group(1)
            title = m.group(2).strip()
            parent = clause_id.rsplit(".", 1)[0] if "." in clause_id else None
            blocks.append(Block(
                text=strip_sentinels(seg),
                citation=f"Clause {clause_id} - {title}",
                kind="clause",
                page_start=page_at_offset(text, m.start()),
                page_end=page_at_offset(text, end),
                parent_citation=f"Clause {parent}" if parent else None,
                extras={"clause_id": clause_id},
            ))
        return blocks
