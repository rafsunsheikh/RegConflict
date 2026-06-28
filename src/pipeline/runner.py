"""End-to-end runner for Steps 1-3.

For each document under CORPUS_ROOT:
  1. Extract (pdfplumber -> pymupdf -> tesseract)
  2. QC report
  3. Clean
  4. Structural parse
  5. Chunk
  6. Tag (stub - noop unless API key is configured)

Writes:
  data/extracted/<juris>/<body>/<docid>.json
  data/qc_reports/<juris>/<body>/<docid>.json
  data/qc_reports/_summary.csv
  data/chunks/<juris>/<body>/<docid>.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from . import chunk as chunk_mod
from . import clean, config, extract, qc, tag
from .parse import get_parser

log = logging.getLogger("pipeline")


def process_document(path: Path) -> tuple[qc.DocQC, int]:
    """Run the full Steps 1-3 pipeline on a single document.

    Returns (qc report, number of chunks written).
    """
    doc = extract.extract_document(path)
    extract.save(doc)

    rep = qc.qc_document(doc)
    qc.save_doc_report(rep)

    if doc.extractor in {"", "failed"} or not doc.pages:
        return rep, 0

    cleaned_pages = clean.clean_pages(doc.pages)
    joined = clean.joined_text(cleaned_pages)

    parser_id, citation_unit = config.parser_id_for(doc.jurisdiction, doc.issuing_body)
    parser = get_parser(parser_id)
    blocks = parser.parse(joined)

    doc_meta = {
        "doc_id": doc.doc_id,
        "jurisdiction": doc.jurisdiction,
        "issuing_body": doc.issuing_body,
        "filename": doc.filename,
        "source_path": doc.source_path,
        "document_type": citation_unit,
        "effective_date": None,
        "source_url": None,
    }
    chunks = chunk_mod.chunk_blocks(blocks, doc_meta=doc_meta)
    chunks = tag.tag_chunks(chunks)
    chunk_mod.write_chunks_jsonl(chunks, doc_meta)
    return rep, len(chunks)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regulatory document ingestion (Steps 1-3)")
    p.add_argument("--root", type=Path, default=config.CORPUS_ROOT,
                   help="Corpus root (defaults to ./Regulatory Documents)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N documents (for smoke tests)")
    p.add_argument("--jurisdiction", default=None,
                   help="Restrict to one jurisdiction (e.g. Singapore)")
    p.add_argument("--body", default=None,
                   help="Restrict to one issuing body (e.g. MAS-Notices)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # pdfminer (used internally by pdfplumber) is extremely chatty at DEBUG.
    for noisy in ("pdfminer", "pdfminer.pdfdocument", "pdfminer.pdfpage",
                  "pdfminer.pdfinterp", "pdfminer.cmapdb", "pdfminer.psparser"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    docs = list(extract.iter_corpus(args.root))
    if args.jurisdiction:
        docs = [d for d in docs if config.source_of(d)[0] == args.jurisdiction]
    if args.body:
        docs = [d for d in docs if config.source_of(d)[1] == args.body]
    if args.limit:
        docs = docs[: args.limit]

    log.info("processing %d documents", len(docs))

    reports: list[qc.DocQC] = []
    chunk_total = 0
    fail = 0
    for path in tqdm(docs):
        try:
            rep, n_chunks = process_document(path)
            reports.append(rep)
            chunk_total += n_chunks
        except Exception as e:
            fail += 1
            log.exception("failed on %s: %s", path, e)

    if reports:
        qc.write_summary_csv(reports)

    # Console rollup.
    needs_re = sum(1 for r in reports if r.needs_reextraction)
    extractors = {}
    for r in reports:
        extractors[r.extractor] = extractors.get(r.extractor, 0) + 1
    log.info("done. docs=%d chunks=%d failures=%d needs_reextraction=%d",
             len(reports), chunk_total, fail, needs_re)
    log.info("extractor breakdown: %s", extractors)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
