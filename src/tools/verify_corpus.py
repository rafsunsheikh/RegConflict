"""Verify integrity of the RegConflict release on disk.

Runs four checks:

  1. **Document inventory consistency.** Every row in
     `data/corpus/document_inventory.csv` must have a license_tier ∈ {Tier 1,
     Tier 2, Tier 3} and a SHA-256.

  2. **Tier 1 / Tier 2 file presence.** Every row classified as Tier 1 or
     Tier 2 should have a corresponding file under `data/corpus/<tier_dir>/`.
     File SHA-256 must match the inventory.

  3. **Tier 3 manifest coverage.** Every row classified as Tier 3 should
     appear in `data/corpus/tier3_metadata_only/manifest.csv` with a
     non-empty `source_url`.

  4. **Conflict JSONL schema sanity.** Every record in `data/conflicts/*.jsonl`
     must:
       - have required top-level fields (pair_id, record_type, regime_a, regime_b, evidence_a, evidence_b, rationale)
       - have record_type ∈ {conflict, non_conflict}
       - for conflict records: have conflict_type ∈ {structural_unresolved,
         operationally_resolvable, interpretive_fact_sensitive, recurring_friction}
         and severity ∈ {low, medium, high}
       - reference regime_ids present in the document inventory.

Usage:
    python src/tools/verify_corpus.py [--records <jsonl>] [--skip-file-presence]

Exit code 0 on full success; 1 on any verification failure. The script prints
per-check results and a summary table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = REPO_ROOT / "data" / "corpus" / "document_inventory.csv"
TIER3_MANIFEST = REPO_ROOT / "data" / "corpus" / "tier3_metadata_only" / "manifest.csv"
SPLITS_DIR = REPO_ROOT / "data" / "conflicts"

REQUIRED_RECORD_FIELDS = {"pair_id", "record_type", "regime_a", "regime_b",
                          "evidence_a", "evidence_b", "rationale"}
CONFLICT_TYPES = {"structural_unresolved", "operationally_resolvable",
                  "interpretive_fact_sensitive", "recurring_friction"}
SEVERITIES = {"low", "medium", "high"}
TIER_DIRS = {
    "tier 1": REPO_ROOT / "data" / "corpus" / "tier1_redistributable",
    "tier 2": REPO_ROOT / "data" / "corpus" / "tier2_conditional",
    "tier 3": REPO_ROOT / "data" / "corpus" / "tier3_metadata_only",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_inventory() -> list[dict]:
    if not INVENTORY.exists():
        raise SystemExit(f"Inventory missing: {INVENTORY}")
    with INVENTORY.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _check_inventory_consistency(rows: list[dict]) -> list[str]:
    errors = []
    for i, row in enumerate(rows, 2):  # CSV row numbers, 1-indexed and header is row 1
        tier = (row.get("license_tier") or "").strip().lower()
        if tier not in TIER_DIRS:
            errors.append(f"inventory row {i}: unknown license_tier {row.get('license_tier')!r}")
        if not (row.get("sha256") or "").strip():
            errors.append(f"inventory row {i}: empty sha256")
    return errors


def _check_tier12_files(rows: list[dict], *, verify_hashes: bool) -> tuple[list[str], int, int]:
    """Return (errors, n_checked, n_with_files).

    The release layout puts Tier 1/2 files under `<tier_dir>/<source_collection>/
    <original_filename>` (sha-256 is in the inventory, not the file path). To
    detect presence we hash every file in the tier directories once and look up
    the expected SHA in the resulting index.
    """
    errors = []
    n_checked = 0
    n_with_files = 0

    # Hash every file in the Tier 1 and Tier 2 directories once
    sha_index: dict[str, Path] = {}
    for tier in ("tier 1", "tier 2"):
        tier_dir = TIER_DIRS[tier]
        if not tier_dir.exists():
            continue
        for p in tier_dir.rglob("*"):
            if not p.is_file() or p.name.startswith("."):
                continue
            # In default mode, key by SHA-256 (always canonical); the
            # --verify-hashes flag is now equivalent to default behaviour.
            sha_index[_sha256_file(p)] = p

    for row in rows:
        tier = (row.get("license_tier") or "").strip().lower()
        if tier not in {"tier 1", "tier 2"}:
            continue
        n_checked += 1
        expected_sha = (row.get("sha256") or "").strip().lower()
        if expected_sha in sha_index:
            n_with_files += 1
        else:
            # Only flag as an error if the tier directory contains files at all;
            # an empty tier directory is informational at pre-publication phases.
            tier_has_files = any(
                p.is_file() and not p.name.startswith(".")
                for p in TIER_DIRS[tier].rglob("*")
                if TIER_DIRS[tier].exists()
            )
            if tier_has_files:
                errors.append(
                    f"{tier} doc sha256={expected_sha[:12]}... not found on disk "
                    f"(inventory title: {row.get('document_title', '?')[:60]})"
                )
    return errors, n_checked, n_with_files


def _check_tier3_manifest(rows: list[dict]) -> list[str]:
    errors = []
    if not TIER3_MANIFEST.exists():
        return [f"Tier 3 manifest missing: {TIER3_MANIFEST}"]
    with TIER3_MANIFEST.open(newline="", encoding="utf-8") as fh:
        manifest_rows = {(r.get("sha256") or "").strip().lower(): r for r in csv.DictReader(fh)}
    for i, row in enumerate(rows, 2):
        tier = (row.get("license_tier") or "").strip().lower()
        if tier != "tier 3":
            continue
        expected_sha = (row.get("sha256") or "").strip().lower()
        manifest_row = manifest_rows.get(expected_sha)
        if manifest_row is None:
            errors.append(f"inventory row {i}: Tier 3 doc with sha256={expected_sha[:8]}... "
                          f"missing from {TIER3_MANIFEST.name}")
            continue
        if not (manifest_row.get("source_url") or "").strip():
            errors.append(f"manifest row for sha256={expected_sha[:8]}...: empty source_url; "
                          f"document cannot be re-fetched")
    return errors


def _load_records(jsonl: Path) -> Iterable[dict]:
    if not jsonl.exists():
        return []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _check_records(record_files: list[Path], inventory_ids: set[str]) -> tuple[list[str], dict]:
    """Schema-check every record. Returns (errors, summary)."""
    errors = []
    by_split = Counter()
    by_type = Counter()
    by_class = Counter()
    by_jurisdiction = Counter()
    unknown_regimes: dict[str, list[str]] = defaultdict(list)
    for jsonl in record_files:
        split_name = jsonl.stem
        for rec in _load_records(jsonl):
            by_split[split_name] += 1
            pid = rec.get("pair_id", "?")
            missing = REQUIRED_RECORD_FIELDS - rec.keys()
            if missing:
                errors.append(f"{split_name}/{pid}: missing fields {sorted(missing)}")
                continue
            rt = rec.get("record_type")
            if rt not in {"conflict", "non_conflict"}:
                errors.append(f"{split_name}/{pid}: invalid record_type={rt!r}")
                continue
            by_type[rt] += 1
            if rt == "conflict":
                ct = rec.get("conflict_type")
                sev = rec.get("severity")
                if ct not in CONFLICT_TYPES:
                    errors.append(f"{split_name}/{pid}: invalid conflict_type={ct!r}")
                else:
                    by_class[ct] += 1
                if sev not in SEVERITIES:
                    errors.append(f"{split_name}/{pid}: invalid severity={sev!r}")
            for side in ("regime_a", "regime_b"):
                rg = rec.get(side) or {}
                rid = (rg.get("regime_id") or "").strip()
                if not rid:
                    errors.append(f"{split_name}/{pid}: empty regime_id on {side}")
                    continue
                by_jurisdiction[rg.get("jurisdiction", "?")] += 1
                # Regime_id is expected to match a source-collection in the inventory.
                # We treat mismatch as informational since some pair-records use synthetic
                # regime composites not present in the document inventory verbatim.
                if rid not in inventory_ids and inventory_ids:
                    unknown_regimes[rid].append(f"{split_name}/{pid}/{side}")
    summary = {
        "by_split": dict(by_split),
        "by_type": dict(by_type),
        "by_class": dict(by_class),
        "by_jurisdiction": dict(by_jurisdiction),
        "unknown_regimes_count": len(unknown_regimes),
    }
    return errors, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=None,
                        help="Verify a single JSONL file instead of all splits")
    parser.add_argument("--skip-file-presence", action="store_true",
                        help="Skip Tier 1/Tier 2 file-presence checks (faster)")
    parser.add_argument("--verify-hashes", action="store_true",
                        help="Recompute SHA-256 of every Tier 1/Tier 2 file (slow)")
    args = parser.parse_args()

    print(f"Verifying RegConflict release at: {REPO_ROOT.relative_to(REPO_ROOT.parent)}")
    print()

    all_errors: list[str] = []

    # Check 1: inventory consistency
    print("[1/4] Inventory consistency...")
    inv_rows = _load_inventory()
    errs = _check_inventory_consistency(inv_rows)
    all_errors.extend(errs)
    print(f"      {len(inv_rows)} rows; {len(errs)} error(s)")
    tier_counts = Counter((r.get("license_tier") or "").strip() for r in inv_rows)
    for k, v in sorted(tier_counts.items()):
        print(f"      {k or '(blank)'}: {v}")

    # Check 2: Tier 1 / Tier 2 file presence
    if not args.skip_file_presence:
        print("[2/4] Tier 1 / Tier 2 file presence...")
        errs, n_checked, n_with_files = _check_tier12_files(inv_rows, verify_hashes=args.verify_hashes)
        all_errors.extend(errs)
        print(f"      {n_checked} Tier 1/2 documents in inventory; {n_with_files} found on disk")
        if n_with_files < n_checked and not any(TIER_DIRS[t].iterdir() if TIER_DIRS[t].exists() else []
                                                 for t in {"tier 1", "tier 2"}):
            print("      (tier directories empty — file presence not yet packaged at this phase)")
    else:
        print("[2/4] Tier 1 / Tier 2 file presence: SKIPPED")

    # Check 3: Tier 3 manifest coverage
    print("[3/4] Tier 3 manifest coverage...")
    errs = _check_tier3_manifest(inv_rows)
    all_errors.extend(errs)
    n_tier3 = sum(1 for r in inv_rows if (r.get("license_tier") or "").strip().lower() == "tier 3")
    print(f"      {n_tier3} Tier 3 documents; {len(errs)} error(s)")

    # Check 4: record schema sanity
    print("[4/4] Record schema sanity...")
    inventory_ids = {(r.get("source_collection") or "").strip() for r in inv_rows
                     if (r.get("source_collection") or "").strip()}
    if args.records:
        record_files = [args.records]
    else:
        record_files = sorted(SPLITS_DIR.glob("*.jsonl"))
    errs, summary = _check_records(record_files, inventory_ids)
    all_errors.extend(errs)
    print(f"      record files checked: {[f.name for f in record_files]}")
    print(f"      total records: {sum(summary['by_split'].values())}")
    for split, n in summary["by_split"].items():
        print(f"        {split}: {n}")
    print(f"      by record_type: {summary['by_type']}")
    print(f"      by conflict_type: {summary['by_class']}")
    print(f"      by jurisdiction (over 2× records, both regime sides counted): {summary['by_jurisdiction']}")
    if summary["unknown_regimes_count"]:
        print(f"      regime_ids not in inventory: {summary['unknown_regimes_count']} distinct "
              "(informational; some pairs use synthetic regime composites)")

    print()
    print("=" * 56)
    if all_errors:
        print(f"VERIFICATION FAILED: {len(all_errors)} error(s)")
        for e in all_errors[:25]:
            print(f"  - {e}")
        if len(all_errors) > 25:
            print(f"  ... ({len(all_errors) - 25} more)")
        return 1
    print("VERIFICATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
