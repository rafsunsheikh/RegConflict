"""EU regulations (EUR-Lex, EBA technical standards under MiCA).

Structure:
    Recital (1), Recital (2), ...     [preamble considerations - cite as Recital N]
    TITLE I / CHAPTER 1               [structural containers; not citation units]
    Article 1, Article 2, ...         [citation units]
        1. ...                        [numbered paragraphs inside an article]
        (a), (b), ...                 [points inside a paragraph]
    ANNEX I, ANNEX II                 [annexes - separate citation unit]

Per the spec the Article is the citation unit. Annexes get their own blocks.
"""
from __future__ import annotations

import re

from .base import BaseParser, Block, page_at_offset, strip_sentinels

# Recitals appear as parenthesised numbers at the start of a line: "(1)  Whereas..."
_RECITAL_RE = re.compile(r"^\s*\((\d{1,3})\)\s+(?=[A-Z])", re.MULTILINE)
# Articles
_ARTICLE_RE = re.compile(r"^\s*Article\s+(\d+[a-z]?)\b[^\n]*", re.MULTILINE)
# Annexes
_ANNEX_RE = re.compile(r"^\s*ANNEX\s+([IVXLC0-9]+)\b[^\n]*", re.MULTILINE)


def _slice_matches(text: str, matches: list[re.Match], cite_fn, kind: str) -> list[Block]:
    out: list[Block] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[m.start():end]
        out.append(Block(
            text=strip_sentinels(seg),
            citation=cite_fn(m),
            kind=kind,
            page_start=page_at_offset(text, m.start()),
            page_end=page_at_offset(text, end),
        ))
    return out


class EURegulationParser(BaseParser):
    def parse(self, text: str) -> list[Block]:
        # Find the article + annex regions first; recitals live *before* the first article.
        articles = list(_ARTICLE_RE.finditer(text))
        annexes = list(_ANNEX_RE.finditer(text))

        if not articles:
            from .generic import GenericParser
            return GenericParser().parse(text)

        first_article = articles[0].start()
        first_annex = annexes[0].start() if annexes else len(text)
        articles_end = first_annex

        # Recitals: only within the preamble before the first article.
        recital_matches = [m for m in _RECITAL_RE.finditer(text, 0, first_article)]
        # Articles: from first article up to first annex (or end of text).
        article_matches = [m for m in articles if m.start() < articles_end]
        # Annexes
        annex_matches = annexes

        blocks: list[Block] = []
        # Preamble before first recital (citation header, signatories, etc.).
        first_recital_off = recital_matches[0].start() if recital_matches else first_article
        preamble = strip_sentinels(text[:first_recital_off])
        if preamble:
            blocks.append(Block(
                text=preamble, citation="Preamble", kind="preamble",
                page_start=page_at_offset(text, 0),
                page_end=page_at_offset(text, first_recital_off),
            ))

        # Recitals (truncate region to first_article so we don't bleed into Articles).
        if recital_matches:
            # Replace original matches list with one limited to recitals only.
            rec_text = text  # we still index against full text for offsets
            for i, m in enumerate(recital_matches):
                end = recital_matches[i + 1].start() if i + 1 < len(recital_matches) else first_article
                seg = text[m.start():end]
                blocks.append(Block(
                    text=strip_sentinels(seg),
                    citation=f"Recital {m.group(1)}",
                    kind="recital",
                    page_start=page_at_offset(text, m.start()),
                    page_end=page_at_offset(text, end),
                ))

        # Articles
        for i, m in enumerate(article_matches):
            end = (article_matches[i + 1].start() if i + 1 < len(article_matches) else articles_end)
            seg = text[m.start():end]
            blocks.append(Block(
                text=strip_sentinels(seg),
                citation=f"Article {m.group(1)}",
                kind="article",
                page_start=page_at_offset(text, m.start()),
                page_end=page_at_offset(text, end),
                extras={"article": m.group(1)},
            ))

        # Annexes
        for i, m in enumerate(annex_matches):
            end = annex_matches[i + 1].start() if i + 1 < len(annex_matches) else len(text)
            seg = text[m.start():end]
            blocks.append(Block(
                text=strip_sentinels(seg),
                citation=f"Annex {m.group(1)}",
                kind="annex",
                page_start=page_at_offset(text, m.start()),
                page_end=page_at_offset(text, end),
                extras={"annex": m.group(1)},
            ))
        return blocks
