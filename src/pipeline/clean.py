"""Step 2: cleaning - run BEFORE structural parsing.

Strips boilerplate headers/footers (regulator names, page numbers, document IDs
that repeat across pages), normalises whitespace, and fixes common OCR / PDF
artefacts (ligatures, smart quotes, soft hyphens).

Change logs are deliberately preserved - they help with supersession tracking
(Step 4) and tend to contain dated, citation-worthy content.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Mapping of common PDF/OCR artefacts to ASCII or canonical equivalents.
ARTIFACT_MAP = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "­": "",   # soft hyphen
}
_ARTIFACT_RE = re.compile("|".join(re.escape(k) for k in ARTIFACT_MAP))

_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*/\s*\d{1,4}|\s+of\s+\d{1,4})?\s*$",
                             re.IGNORECASE | re.MULTILINE)
_MULTI_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def _normalise_chars(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return _ARTIFACT_RE.sub(lambda m: ARTIFACT_MAP[m.group(0)], text)


def _normalise_whitespace(text: str) -> str:
    text = _MULTI_WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    # Trim per-line trailing whitespace.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _detect_repeating_lines(pages: list[str], min_frac: float = 0.5) -> set[str]:
    """A header/footer is a short line that repeats across >=min_frac of pages.

    We look at the first 3 and last 3 lines of every page only - body lines
    naturally repeat (definitions, refrains) but not at edges.
    """
    counter: Counter[str] = Counter()
    n_pages = len(pages)
    for txt in pages:
        if not txt:
            continue
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        edges = lines[:3] + lines[-3:]
        for line in edges:
            if 3 <= len(line) <= 120:
                counter[line] += 1
    threshold = max(2, int(n_pages * min_frac))
    return {line for line, c in counter.items() if c >= threshold}


def clean_pages(pages: list[dict]) -> list[dict]:
    """Return cleaned pages [{page, text}, ...]. Removes repeating boilerplate.

    Preserves page boundaries because Step 2's structural parsers and Step 3's
    chunker still need to know which page a chunk came from (for citation).
    """
    raw_texts = [p.get("text") or "" for p in pages]
    normed = [_normalise_chars(t) for t in raw_texts]
    boilerplate = _detect_repeating_lines(normed)

    cleaned: list[dict] = []
    for orig, norm in zip(pages, normed):
        # Drop standalone page-number lines and repeated boilerplate lines.
        text = _PAGE_NUMBER_RE.sub("", norm)
        if boilerplate:
            kept = [l for l in text.split("\n") if l.strip() not in boilerplate]
            text = "\n".join(kept)
        text = _normalise_whitespace(text)
        cleaned.append({"page": orig.get("page"), "text": text})
    return cleaned


def joined_text(pages: list[dict]) -> str:
    """Join cleaned page texts with a page-break sentinel that parsers can split on.

    Sentinel format: ``\f<PAGE n>\f`` (form-feed delimited so it can never collide
    with legitimate body content).
    """
    parts: list[str] = []
    for p in pages:
        parts.append(f"\f<PAGE {p['page']}>\f\n{p['text']}")
    return "\n".join(parts)
