"""Reproduce the train / val / test / gold_iaa splits from raw labelled records.

The v1.0 release pins specific split assignments, generated under seed=42
with stratified sampling on `conflict_type` (5 strata: 4 conflict types +
`non_conflict`) at the regime-pair level. Re-running this script with the
same seed must produce byte-identical outputs to `data/conflicts/*.jsonl`
in the release.

Inputs:
    Raw labelled records, one JSONL per pool:
      - `<source>/conflicts.jsonl`     — labelled conflict pairs
      - `<source>/non_conflicts.jsonl` — labelled non-conflict pairs

Outputs (under --output):
    train.jsonl    — 70% of non-IAA pool, stratified
    val.jsonl      — 15% of non-IAA pool, stratified
    test.jsonl     — 15% of non-IAA pool, stratified
    gold_iaa.jsonl — adjudicated IAA-overlap records (held out before split)
    split_metadata.json — seed, proportions, per-split counts and pair list

Usage:
    python src/tools/build_splits.py \\
        --source data/raw_labels/ \\
        --output data/conflicts/ \\
        --seed 42

The source directory must contain `conflicts.jsonl` and `non_conflicts.jsonl`
files. IAA records (those carrying a `blind_review` block in the raw source)
are held out and written to `gold_iaa.jsonl` separately.

This script is intentionally deterministic. If the input pool changes
(records added, removed, or relabelled), the split assignment will change;
the canonical v1.0 splits are pinned by the input snapshot at the time of
release.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _is_iaa(record: dict) -> bool:
    """A record is in the IAA overlap pool iff it has a `blind_review` block."""
    return "blind_review" in record or "review" in record and \
        record.get("review", {}).get("reviewer_id") is not None


def _stratum(record: dict) -> str:
    """Return the stratification key for a record."""
    if record.get("record_type") == "conflict":
        ct = record.get("conflict_type") or "conflict_unknown"
        return f"conflict::{ct}"
    return "non_conflict"


def _stratified_split(
    records: list[dict],
    proportions: dict[str, float],
    seed: int,
) -> dict[str, list[dict]]:
    """Stratified random split. Returns {split_name: [records]}."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_stratum[_stratum(r)].append(r)
    for stratum_records in by_stratum.values():
        rng.shuffle(stratum_records)

    splits: dict[str, list[dict]] = {name: [] for name in proportions}
    split_names = list(proportions.keys())

    for stratum, stratum_records in by_stratum.items():
        n = len(stratum_records)
        # Compute integer counts per split for this stratum
        counts = {name: int(n * proportions[name]) for name in split_names}
        # Distribute leftover records to splits in descending proportion order
        leftover = n - sum(counts.values())
        for name in sorted(split_names, key=lambda x: -proportions[x]):
            if leftover <= 0:
                break
            counts[name] += 1
            leftover -= 1
        # Assign
        cursor = 0
        for name in split_names:
            chunk = stratum_records[cursor:cursor + counts[name]]
            splits[name].extend(chunk)
            cursor += counts[name]

    # Final shuffle within each split so records aren't grouped by stratum
    for name in split_names:
        rng.shuffle(splits[name])

    return splits


def _write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _summarise(records: list[dict]) -> dict:
    binary = {"conflict": 0, "non_conflict": 0}
    per_type = defaultdict(int)
    for r in records:
        rt = r.get("record_type")
        if rt in binary:
            binary[rt] += 1
        if rt == "conflict":
            per_type[r.get("conflict_type", "unknown")] += 1
    return {"n_records": len(records), "binary_counts": binary,
            "per_conflict_type": dict(per_type)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory containing conflicts.jsonl and non_conflicts.jsonl")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "conflicts",
                        help="Output directory for {train,val,test,gold_iaa}.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    proportions = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}
    if abs(sum(proportions.values()) - 1.0) > 1e-6:
        raise SystemExit(f"Proportions must sum to 1.0; got {sum(proportions.values())}")

    conflicts = _load_jsonl(args.source / "conflicts.jsonl")
    non_conflicts = _load_jsonl(args.source / "non_conflicts.jsonl")
    print(f"Loaded {len(conflicts)} conflict records + {len(non_conflicts)} non-conflict records.")

    all_records = conflicts + non_conflicts

    # Hold out IAA records
    iaa_records = [r for r in all_records if _is_iaa(r)]
    non_iaa_records = [r for r in all_records if not _is_iaa(r)]
    print(f"  IAA overlap pool: {len(iaa_records)} records")
    print(f"  Non-IAA pool:     {len(non_iaa_records)} records")

    # Stratified split on the non-IAA pool
    splits = _stratified_split(non_iaa_records, proportions, args.seed)
    splits["gold_iaa"] = iaa_records

    # Write outputs
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test", "gold_iaa"):
        path = args.output / f"{name}.jsonl"
        n = _write_jsonl(splits[name], path)
        s = _summarise(splits[name])
        print(f"  Wrote {path.relative_to(REPO_ROOT)}: {n} records "
              f"(conflict={s['binary_counts']['conflict']}, "
              f"non_conflict={s['binary_counts']['non_conflict']})")

    # Emit split_metadata.json
    metadata = {
        "seed": args.seed,
        "split_unit": "regime_pair",
        "split_proportions": proportions,
        "stratify_by": "conflict_type (5 strata: 4 conflict types + 'non_conflict')",
        "iaa_holdout_source": "records carrying a `blind_review` or `review` block in the raw source",
        "n_total_records": len(all_records),
        "n_total_pairs": len({r.get("pair_id") for r in all_records if r.get("pair_id")}),
        "splits": [
            {"name": name, **_summarise(splits[name]),
             "pairs": [r["pair_id"] for r in splits[name]]}
            for name in ("train", "val", "test", "gold_iaa")
        ],
    }
    metadata_path = args.output.parent / "splits" / "split_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Wrote {metadata_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
