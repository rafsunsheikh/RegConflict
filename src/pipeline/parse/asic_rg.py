"""ASIC Regulatory Guides - paragraph numbering ``RG NNN.MM`` is the citation unit.

Example: ``RG 165.21 The Corporations Act requires...``
"""
from __future__ import annotations

import re

from .base import BaseParser, Block, page_at_offset, strip_sentinels

_RG_PARA_RE = re.compile(r"\bRG\s?(\d{1,4})\.(\d+)\b")


class ASICRGParser(BaseParser):
    def parse(self, text: str) -> list[Block]:
        matches = list(_RG_PARA_RE.finditer(text))
        if len(matches) < 2:
            from .generic import GenericParser
            return GenericParser().parse(text)

        blocks: list[Block] = []
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
            rg_no = f"RG {m.group(1)}.{m.group(2)}"
            blocks.append(Block(
                text=strip_sentinels(seg),
                citation=rg_no,
                kind="paragraph",
                page_start=page_at_offset(text, m.start()),
                page_end=page_at_offset(text, end),
                parent_citation=f"RG {m.group(1)}",
                extras={"rg_number": m.group(1), "paragraph": m.group(2)},
            ))
        return blocks
