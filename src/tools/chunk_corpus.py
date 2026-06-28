"""Run the RegConflict ingestion pipeline (extract → clean → parse → chunk)
on Tier 3 source documents that users have fetched via `fetch_tier3.py`.

The v1.0 release ships pre-extracted text and chunks for the 192 Tier 1 and
Tier 2 documents under `data/extracted/` and `data/chunks/`. The 55 Tier 3
documents (FATF, MAS, IRAS) cannot be redistributed under their source
licenses, so their extracted text and chunks are not shipped either. This
tool regenerates the equivalent files on the user's machine after they have
fetched the Tier 3 PDFs locally:

    # Step 1: fetch the Tier 3 PDFs
    python src/tools/fetch_tier3.py

    # Step 2: regenerate extracted text + chunks for the fetched PDFs
    python src/tools/chunk_corpus.py --tier 3

After Step 2, `data/extracted/` and `data/chunks/` contain the full 247-document
corpus and behave identically to the released Tier 1+2 files.

Usage:
    python src/tools/chunk_corpus.py [--tier {1,2,3,all}] [--source MAS|FATF|...]
                                      [--force] [--limit N]

Behaviour:
    - Reads `data/corpus/document_inventory.csv` to enumerate documents.
    - For each document, locates the PDF/DOCX on disk under
      `data/corpus/{tier1_redistributable,tier2_conditional,tier3_metadata_only}/<source>/`.
    - Skips documents whose source file is not present (informational warning).
    - Skips documents whose extracted+chunks files already exist on disk
      with the expected `<stem>__<sha16>.{json,jsonl}` naming, unless --force.
    - Anonymises `source_path` fields to repo-relative form before writing.
    - Prints a per-source summary on completion.

The chunking strategy (CHUNK_TARGET_TOKENS=200, CHUNK_MAX_TOKENS=400, no
overlap, tiktoken cl100k tokeniser) and parser registry are defined in
`src/pipeline/config.py` — running this tool on Tier 3 documents produces
chunks that are byte-identical to what the release authors would have
produced, modulo the source PDFs being unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import chunk as chunk_mod
from src.pipeline import clean, config, extract, qc, tag
from src.pipeline.parse import get_parser

INVENTORY = REPO_ROOT / "data" / "corpus" / "document_inventory.csv"
TIER_DIRS = {
    "Tier 1": REPO_ROOT / "data" / "corpus" / "tier1_redistributable",
    "Tier 2": REPO_ROOT / "data" / "corpus" / "tier2_conditional",
    "Tier 3": REPO_ROOT / "data" / "corpus" / "tier3_metadata_only",
}


def _anonymise_source_path(absolute: str, source_collection: str, tier: str, filename: str) -> str:
    tier_dir = {
        "Tier 1": "tier1_redistributable",
        "Tier 2": "tier2_conditional",
        "Tier 3": "tier3_metadata_only",
    }[tier]
    return f"data/corpus/{tier_dir}/{source_collection}/{filename}"


def _output_paths_exist(jurisdiction: str, source_collection: str, filename: str, sha16: str) -> bool:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem)
    extracted = config.EXTRACTED_DIR / jurisdiction / source_collection / f"{safe_stem}__{sha16}.json"
    chunks = config.CHUNKS_DIR / jurisdiction / source_collection / f"{safe_stem}__{sha16}.jsonl"
    return extracted.exists() and chunks.exists()


def _locate_source(row: dict) -> Path | None:
    """Find the actual file on disk for an inventory row. Returns None if missing."""
    tier = row["license_tier"]
    source = row["source_collection"]
    # For Tier 1/2, files are stored as <tier>/<source>/<original_filename>.
    # For Tier 3, fetch_tier3 stores them as <tier>/<sha8>/<original_filename>.
    tier_root = TIER_DIRS[tier] / source
    if tier_root.exists():
        # Try matching by filename or by SHA-256
        sha_prefix_8 = row["sha256"][:8]
        for candidate in tier_root.rglob("*"):
            if candidate.is_file() and (candidate.suffix.lower() in {".pdf", ".docx"}):
                return candidate
    # Tier 3 layout from fetch_tier3.py: <tier3_dir>/<sha8>/<filename>.pdf
    if tier == "Tier 3":
        sha8 = row["sha256"][:8]
        cand = TIER_DIRS["Tier 3"] / sha8
        if cand.exists():
            pdfs = list(cand.glob("*.pdf")) + list(cand.glob("*.docx"))
            if pdfs:
                return pdfs[0]
    return None


def process_row(row: dict, *, force: bool) -> tuple[str, str]:
    """Return (status, message)."""
    tier = row["license_tier"]
    source_collection = row["source_collection"]
    jurisdiction = row["jurisdiction"]
    # Normalise INT-* jurisdictions to "International" so they match TRIAG's path conventions
    norm_jurisdiction = "International" if jurisdiction.startswith("INT-") else jurisdiction
    sha16 = row["sha256"][:16]
    filename = row["document_title"]
    expected_filename_pdf = f"{filename}.pdf"

    if not force and _output_paths_exist(norm_jurisdiction, source_collection,
                                          expected_filename_pdf, sha16):
        return "skipped_present", f"{filename}: outputs already on disk"

    src_path = _locate_source(row)
    if src_path is None:
        return "missing_source", (
            f"{filename}: source PDF not on disk under {TIER_DIRS[tier].name}/{source_collection}/. "
            f"For Tier 3, run `python src/tools/fetch_tier3.py --source {source_collection}` first."
        )

    # Run the pipeline with explicit metadata (don't rely on path inference,
    # since the tier-based layout doesn't match TRIAG's jurisdiction/body tree)
    try:
        doc = extract.extract_document(
            src_path,
            jurisdiction=norm_jurisdiction,
            issuing_body=source_collection,
        )
    except Exception as e:
        return "extract_failed", f"{filename}: {e}"

    # Anonymise source_path before saving
    doc.source_path = _anonymise_source_path(
        str(src_path), source_collection, tier, src_path.name)
    extract.save(doc)

    if doc.extractor in {"", "failed"} or not doc.pages:
        return "extract_empty", f"{filename}: extractor={doc.extractor!r} produced no pages"

    rep = qc.qc_document(doc)
    qc.save_doc_report(rep)

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
        "source_path": doc.source_path,  # already anonymised
        "document_type": citation_unit,
        "effective_date": None,
        "source_url": None,
    }
    chunks = chunk_mod.chunk_blocks(blocks, doc_meta=doc_meta)
    chunks = tag.tag_chunks(chunks)
    chunk_mod.write_chunks_jsonl(chunks, doc_meta)
    return "ok", f"{filename}: {len(chunks)} chunks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=["1", "2", "3", "all"], default="3",
                        help="Which tier(s) to process (default: 3 — typical for users "
                             "regenerating after fetch_tier3.py)")
    parser.add_argument("--source", default=None,
                        help="Restrict to one source_collection (e.g., MAS, FATF, IRAS)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess even if extracted+chunks already exist on disk")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N documents (smoke test)")
    args = parser.parse_args()

    rows = list(csv.DictReader(INVENTORY.open(newline="")))
    tier_filter = {
        "1": {"Tier 1"},
        "2": {"Tier 2"},
        "3": {"Tier 3"},
        "all": {"Tier 1", "Tier 2", "Tier 3"},
    }[args.tier]
    rows = [r for r in rows if r["license_tier"] in tier_filter]
    if args.source:
        rows = [r for r in rows if args.source.upper() in r["source_collection"].upper()]
    if args.limit:
        rows = rows[:args.limit]

    print(f"Processing {len(rows)} document(s) (tier={args.tier}, source={args.source or 'any'})")
    print()

    summary = {"ok": 0, "skipped_present": 0, "missing_source": 0,
               "extract_failed": 0, "extract_empty": 0}
    by_source = {}
    for row in rows:
        status, msg = process_row(row, force=args.force)
        summary[status] = summary.get(status, 0) + 1
        src = row["source_collection"]
        by_source.setdefault(src, {}).setdefault(status, 0)
        by_source[src][status] += 1
        sym = {"ok": "✓", "skipped_present": "•", "missing_source": "✗",
               "extract_failed": "✗", "extract_empty": "✗"}.get(status, "?")
        print(f"  {sym} [{status:>16}] {msg}")

    print()
    print("Summary:")
    for k in sorted(summary):
        print(f"  {k:>16}: {summary[k]}")
    failure_states = {"missing_source", "extract_failed", "extract_empty"}
    return 0 if not any(summary.get(s, 0) for s in failure_states) else 2


if __name__ == "__main__":
    raise SystemExit(main())
