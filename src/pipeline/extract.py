"""Step 1: text extraction with pdfplumber -> pymupdf -> tesseract fallback.

Each document produces one JSON file under data/extracted/<jurisdiction>/<body>/<docid>.json
with structure:
    {
      "doc_id": str,                # sha256[:16] of file bytes
      "source_path": str,
      "jurisdiction": str,
      "issuing_body": str,
      "filename": str,
      "ext": str,
      "extractor": "pdfplumber" | "pymupdf" | "tesseract" | "docx",
      "page_count": int,
      "pages": [{"page": 1, "text": "..."}, ...],
      "extraction_errors": [str, ...],
    }
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import config

log = logging.getLogger(__name__)

# Heuristic threshold: if pdfplumber returns <THRESHOLD chars per page on average,
# the PDF is probably scanned -> fall back to OCR.
MIN_CHARS_PER_PAGE = 50


@dataclass
class ExtractedDoc:
    doc_id: str
    source_path: str
    jurisdiction: str
    issuing_body: str
    filename: str
    ext: str
    extractor: str
    page_count: int
    pages: list[dict] = field(default_factory=list)
    extraction_errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return self.__dict__

    @property
    def total_chars(self) -> int:
        return sum(len(p.get("text", "")) for p in self.pages)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _avg_chars(pages: list[dict]) -> float:
    return (sum(len(p["text"]) for p in pages) / len(pages)) if pages else 0.0


def _extract_pdfplumber(path: Path) -> tuple[list[dict], list[str]]:
    import pdfplumber

    pages: list[dict] = []
    errs: list[str] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:  # pdfplumber occasionally fails on individual pages
                text = ""
                errs.append(f"pdfplumber p{i}: {e}")
            pages.append({"page": i, "text": text})
    return pages, errs


def _extract_pymupdf(path: Path) -> tuple[list[dict], list[str]]:
    import pymupdf  # type: ignore

    pages: list[dict] = []
    errs: list[str] = []
    doc = pymupdf.open(path)
    try:
        for i, page in enumerate(doc, start=1):
            try:
                text = page.get_text("text") or ""
            except Exception as e:
                text = ""
                errs.append(f"pymupdf p{i}: {e}")
            pages.append({"page": i, "text": text})
    finally:
        doc.close()
    return pages, errs


def _extract_tesseract(path: Path) -> tuple[list[dict], list[str]]:
    """Rasterize each page via pymupdf then OCR with tesseract."""
    import pymupdf  # type: ignore
    import pytesseract
    from PIL import Image
    import io

    pages: list[dict] = []
    errs: list[str] = []
    doc = pymupdf.open(path)
    try:
        for i, page in enumerate(doc, start=1):
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img) or ""
            except Exception as e:
                text = ""
                errs.append(f"tesseract p{i}: {e}")
            pages.append({"page": i, "text": text})
    finally:
        doc.close()
    return pages, errs


def _extract_docx(path: Path) -> tuple[list[dict], list[str]]:
    """docx has no native page boundaries; treat the whole document as page 1."""
    import docx  # python-docx

    errs: list[str] = []
    try:
        d = docx.Document(str(path))
        paragraphs = [p.text for p in d.paragraphs]
        text = "\n".join(paragraphs)
        return [{"page": 1, "text": text}], errs
    except Exception as e:
        errs.append(f"docx: {e}")
        return [{"page": 1, "text": ""}], errs


def extract_document(
    path: Path,
    *,
    force_ocr: bool = False,
    jurisdiction: str | None = None,
    issuing_body: str | None = None,
) -> ExtractedDoc:
    """Extract text from a single document, choosing extractor by extension + content.

    If `jurisdiction` and `issuing_body` are provided explicitly, they override
    the path-based inference from `config.source_of`. This is what
    `src/tools/chunk_corpus.py` uses to drive the pipeline from the inventory CSV
    rather than from the corpus directory tree (since RegConflict's layout
    organises documents by license tier, not jurisdiction)."""
    if jurisdiction is None or issuing_body is None:
        jurisdiction, issuing_body = config.source_of(path)
    ext = path.suffix.lower().lstrip(".")
    doc = ExtractedDoc(
        doc_id=_hash_file(path),
        source_path=str(path),
        jurisdiction=jurisdiction,
        issuing_body=issuing_body,
        filename=path.name,
        ext=ext,
        extractor="",
        page_count=0,
    )

    if ext == "docx":
        pages, errs = _extract_docx(path)
        doc.extractor = "docx"
        doc.pages = pages
        doc.extraction_errors.extend(errs)
        doc.page_count = len(pages)
        return doc

    if ext != "pdf":
        doc.extraction_errors.append(f"unsupported extension: {ext}")
        return doc

    # PDF: pdfplumber -> pymupdf -> tesseract
    try:
        if not force_ocr:
            pages, errs = _extract_pdfplumber(path)
            doc.extraction_errors.extend(errs)
            if _avg_chars(pages) >= MIN_CHARS_PER_PAGE:
                doc.extractor = "pdfplumber"
                doc.pages = pages
                doc.page_count = len(pages)
                return doc
    except Exception as e:
        doc.extraction_errors.append(f"pdfplumber failed: {e}")

    try:
        if not force_ocr:
            pages, errs = _extract_pymupdf(path)
            doc.extraction_errors.extend(errs)
            if _avg_chars(pages) >= MIN_CHARS_PER_PAGE:
                doc.extractor = "pymupdf"
                doc.pages = pages
                doc.page_count = len(pages)
                return doc
    except Exception as e:
        doc.extraction_errors.append(f"pymupdf failed: {e}")

    # Falling through: probably scanned -> OCR.
    try:
        pages, errs = _extract_tesseract(path)
        doc.extraction_errors.extend(errs)
        doc.extractor = "tesseract"
        doc.pages = pages
        doc.page_count = len(pages)
    except Exception as e:
        doc.extraction_errors.append(f"tesseract failed: {e}")
        doc.extractor = "failed"
    return doc


def iter_corpus(root: Path | None = None) -> Iterable[Path]:
    root = root or config.CORPUS_ROOT
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.suffix.lower() in {".ds_store", ".rar", ".gif"}:
            continue
        if p.suffix.lower() in {".pdf", ".docx"}:
            yield p


def output_path_for(doc: ExtractedDoc) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(doc.filename).stem)
    return (
        config.EXTRACTED_DIR
        / doc.jurisdiction
        / doc.issuing_body
        / f"{safe_name}__{doc.doc_id}.json"
    )


def save(doc: ExtractedDoc) -> Path:
    out = output_path_for(doc)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc.to_json(), ensure_ascii=False))
    return out
