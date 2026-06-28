"""Australian Acts / Singapore SSO Acts - section numbering is the citation unit.

Section headers in OPC-style Acts look like:
    "5  Definitions"
    "5  Object of this Act"
    "12A  Penalties for ..."
And subsections continue inside that section's body:
    "(1)  ..."
    "(2)  ..."
    "(3) (a) ..."

We treat the section as the citation unit (per spec). Subsection numbering is
preserved inside the section text and exposed via extras for downstream use.
"""
from __future__ import annotations

import re

from .base import BaseParser, Block, page_at_offset, strip_sentinels

# Section header: digits (optionally followed by a letter) at start of line, then
# at least two spaces, then a heading word. This is the OPC drafting convention.
_SECTION_RE = re.compile(
    r"^\s*(\d{1,4}[A-Z]?)\s{2,}([A-Z][^\n]{0,200})$",
    re.MULTILINE,
)


class AUActParser(BaseParser):
    def parse(self, text: str) -> list[Block]:
        matches = list(_SECTION_RE.finditer(text))
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
            sec_no = m.group(1)
            title = m.group(2).strip()
            blocks.append(Block(
                text=strip_sentinels(seg),
                citation=f"s.{sec_no} - {title}",
                kind="section",
                page_start=page_at_offset(text, m.start()),
                page_end=page_at_offset(text, end),
                extras={"section": sec_no},
            ))
        return blocks
