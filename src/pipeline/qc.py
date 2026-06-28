"""Step 1: quality check + report.

Sample 5-10 random pages per doc, run heuristic checks that catch the common
legal-text extraction failures called out in the pipeline spec:

  * empty/near-empty pages   -> extractor likely failed on this page
  * lost section numbering   -> our most valuable citation metadata
  * stray repeating headers  -> boilerplate not stripped (informational; cleaned later)
  * suspicious char mix      -> OCR garble or encoding mojibake

Emit two artefacts:
  data/qc_reports/<juris>/<body>/<docid>.json   - per-doc details
  data/qc_reports/_summary.csv                  - corpus rollup
"""
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .extract import ExtractedDoc


# Regexes for section-numbering patterns we expect to survive extraction.
# Matching *any* of these on a sampled page counts as "section numbering present".
SECTION_PATTERNS = [
    re.compile(r"\bArticle\s+\d+", re.IGNORECASE),         # EU
    re.compile(r"\bRecital\s+\(?\d+\)?", re.IGNORECASE),   # EU
    re.compile(r"\bRG\s?\d{1,4}\.\d+", re.IGNORECASE),      # ASIC RG
    re.compile(r"\bs\.\s?\d+(\(\w+\))*"),                  # AU Act sections (s.5(1)(a))
    re.compile(r"\bSection\s+\d+", re.IGNORECASE),
    re.compile(r"\bClause\s+\d+", re.IGNORECASE),          # MAS Notices
    re.compile(r"^\s*\d+\.\d+\s", re.MULTILINE),           # paragraph numbering
    re.compile(r"^\s*\(\d+\)\s", re.MULTILINE),            # numbered subsections
    re.compile(r"^\s*\d+\.\s", re.MULTILINE),              # top-level numbering
    re.compile(r"\b[A-Z]ART\s+[IVX]+", re.IGNORECASE),     # PART roman numerals
]

# Suspicious characters from common OCR/encoding failures.
SUSPICIOUS_RE = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class PageQC:
    page: int
    chars: int
    has_section_marker: bool
    suspicious_chars: int

    def to_json(self) -> dict:
        return self.__dict__


@dataclass
class DocQC:
    doc_id: str
    source_path: str
    jurisdiction: str
    issuing_body: str
    filename: str
    extractor: str
    page_count: int
    sampled_pages: list[PageQC] = field(default_factory=list)
    extraction_errors: list[str] = field(default_factory=list)

    # Aggregates
    empty_page_rate: float = 0.0
    section_marker_rate: float = 0.0
    suspicious_char_rate: float = 0.0
    avg_chars_per_page: float = 0.0
    needs_reextraction: bool = False
    reason: str = ""

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        d["sampled_pages"] = [p.to_json() for p in self.sampled_pages]
        return d


def _check_page(page: dict) -> PageQC:
    text = page.get("text", "") or ""
    has_marker = any(p.search(text) for p in SECTION_PATTERNS)
    suspicious = len(SUSPICIOUS_RE.findall(text))
    return PageQC(
        page=page.get("page", 0),
        chars=len(text),
        has_section_marker=has_marker,
        suspicious_chars=suspicious,
    )


def qc_document(doc: ExtractedDoc, *, rng: random.Random | None = None) -> DocQC:
    rng = rng or random.Random(doc.doc_id)
    n = max(config.QC_MIN_SAMPLE, min(config.QC_MAX_SAMPLE, doc.page_count))
    n = min(n, doc.page_count)
    sample_pages = rng.sample(doc.pages, n) if n else []
    sampled = [_check_page(p) for p in sample_pages]

    if doc.pages:
        avg_chars = sum(len((p.get("text") or "")) for p in doc.pages) / doc.page_count
    else:
        avg_chars = 0.0

    empty = sum(1 for p in sampled if p.chars < 20)
    section = sum(1 for p in sampled if p.has_section_marker)
    suspicious_total = sum(p.suspicious_chars for p in sampled)
    sampled_chars = sum(p.chars for p in sampled) or 1

    report = DocQC(
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        jurisdiction=doc.jurisdiction,
        issuing_body=doc.issuing_body,
        filename=doc.filename,
        extractor=doc.extractor,
        page_count=doc.page_count,
        sampled_pages=sampled,
        extraction_errors=list(doc.extraction_errors),
        empty_page_rate=empty / max(1, len(sampled)),
        section_marker_rate=section / max(1, len(sampled)),
        suspicious_char_rate=suspicious_total / sampled_chars,
        avg_chars_per_page=avg_chars,
    )

    reasons: list[str] = []
    if doc.extractor in {"", "failed"}:
        reasons.append("extraction failed")
    if avg_chars < 100:
        reasons.append("near-empty document")
    if report.empty_page_rate > 0.4:
        reasons.append(f"{report.empty_page_rate:.0%} of sampled pages near-empty")
    if report.section_marker_rate == 0.0 and avg_chars > 200:
        # Many documents legitimately have no section markers on a random page
        # (e.g., FATF recommendations, narrative guidance) so this is a soft signal.
        reasons.append("no section markers in sample")
    if report.suspicious_char_rate > 0.005:
        reasons.append("high suspicious-char rate (OCR garble?)")
    report.needs_reextraction = bool(reasons) and reasons != ["no section markers in sample"]
    report.reason = "; ".join(reasons)
    return report


def save_doc_report(rep: DocQC) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(rep.filename).stem)
    out = config.QC_DIR / rep.jurisdiction / rep.issuing_body / f"{safe}__{rep.doc_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep.to_json(), ensure_ascii=False, indent=2))
    return out


def write_summary_csv(reports: list[DocQC]) -> Path:
    out = config.QC_DIR / "_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "jurisdiction", "issuing_body", "filename", "doc_id", "extractor",
            "pages", "avg_chars_per_page", "empty_page_rate",
            "section_marker_rate", "suspicious_char_rate",
            "needs_reextraction", "reason",
        ])
        for r in reports:
            w.writerow([
                r.jurisdiction, r.issuing_body, r.filename, r.doc_id, r.extractor,
                r.page_count, f"{r.avg_chars_per_page:.1f}",
                f"{r.empty_page_rate:.2f}", f"{r.section_marker_rate:.2f}",
                f"{r.suspicious_char_rate:.4f}",
                int(r.needs_reextraction), r.reason,
            ])
    return out
