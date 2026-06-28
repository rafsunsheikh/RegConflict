"""Tests for the labelled splits and the document inventory."""
from __future__ import annotations

import pytest

from src.eval.benchmark.schema import (
    ConflictType,
    LabelledPair,
    Severity,
)

VALID_CONFLICT_TYPES = {ct.value for ct in ConflictType}
VALID_SEVERITIES = {s.value for s in Severity}


class TestSplits:
    """Per-split sanity checks on the JSONL records."""

    def test_all_splits_load(self, all_records):
        for name in ("train", "val", "test", "gold_iaa"):
            assert name in all_records, f"missing split: {name}"

    def test_no_split_is_empty(self, all_records):
        for name, recs in all_records.items():
            assert recs, f"split {name} is empty"

    def test_split_sizes_match_paper(self, all_records):
        # Paper-reported sizes; the split_metadata.json is the canonical source.
        assert len(all_records["train"]) == 497
        assert len(all_records["val"]) == 107
        assert len(all_records["test"]) == 107
        # gold_iaa is currently 13 records (legacy v0.9); v1.0 expects 30.
        # We don't pin a strict count here to allow the v0.9 → v1.0 transition.
        assert 13 <= len(all_records["gold_iaa"]) <= 30

    def test_total_record_count(self, all_records):
        total = sum(len(r) for r in all_records.values())
        # 724 was the v0.9 count (497 + 107 + 107 + 13). v1.0 will be 741 after
        # the 30-pair gold IAA merge. Accept the range.
        assert 724 <= total <= 741

    def test_record_types(self, all_records):
        for name, recs in all_records.items():
            for r in recs:
                assert r["record_type"] in {"conflict", "non_conflict"}, \
                    f"{name}/{r.get('pair_id')}: bad record_type={r['record_type']!r}"

    def test_pair_ids_unique_across_splits(self, all_records):
        """A pair_id may appear multiple times within a split (the v0.9 data has
        ~9 such duplicates, documented in split_metadata.json's n_total_pairs vs
        n_total_records gap), but it must NOT appear in two different splits —
        that would be a leakage failure."""
        from collections import defaultdict
        splits_per_pid: dict[str, set[str]] = defaultdict(set)
        for name, recs in all_records.items():
            for r in recs:
                splits_per_pid[r["pair_id"]].add(name)
        leakage = {pid: splits for pid, splits in splits_per_pid.items() if len(splits) > 1}
        assert not leakage, f"pair_id leakage across splits: {dict(list(leakage.items())[:5])}"


class TestRecordSchema:
    """Validate every record through the Pydantic schema."""

    def test_every_record_parses(self, all_records):
        n = 0
        for name, recs in all_records.items():
            for r in recs:
                LabelledPair.model_validate(r)
                n += 1
        assert n > 0, "no records parsed — fixture is empty"

    def test_conflict_records_have_typology(self, all_records):
        for name, recs in all_records.items():
            for r in recs:
                if r["record_type"] == "conflict":
                    assert r.get("conflict_type") in VALID_CONFLICT_TYPES, \
                        f"{name}/{r['pair_id']}: bad conflict_type={r.get('conflict_type')!r}"
                    assert r.get("severity") in VALID_SEVERITIES, \
                        f"{name}/{r['pair_id']}: bad severity={r.get('severity')!r}"

    def test_non_conflict_records_have_no_typology(self, all_records):
        for name, recs in all_records.items():
            for r in recs:
                if r["record_type"] == "non_conflict":
                    # Non-conflict records may not have conflict_type/severity set,
                    # or they may be explicitly null.
                    assert r.get("conflict_type") in (None, ""), \
                        f"{name}/{r['pair_id']}: non_conflict has conflict_type={r.get('conflict_type')!r}"

    def test_regimes_have_required_fields(self, all_records):
        for name, recs in all_records.items():
            for r in recs:
                for side in ("regime_a", "regime_b"):
                    rg = r.get(side)
                    assert isinstance(rg, dict), f"{name}/{r['pair_id']}: missing {side}"
                    assert rg.get("regime_id"), f"{name}/{r['pair_id']}: empty {side}.regime_id"
                    assert rg.get("jurisdiction") in {"Australia", "EU", "Singapore", "International"}, \
                        f"{name}/{r['pair_id']}: bad {side}.jurisdiction={rg.get('jurisdiction')!r}"

    def test_evidence_passages_present(self, all_records):
        for name, recs in all_records.items():
            for r in recs:
                for side in ("evidence_a", "evidence_b"):
                    ev = r.get(side, [])
                    assert isinstance(ev, list), f"{name}/{r['pair_id']}: {side} not a list"
                    for chunk in ev:
                        assert chunk.get("passage"), f"{name}/{r['pair_id']}: empty passage in {side}"


class TestCleanedCorpus:
    """Ship-state sanity checks on the extracted/ and chunks/ directories."""

    def test_extracted_count_matches_tier12(self, repo_root, inventory):
        """Disk file count should equal the number of distinct
        (sha16, source_collection) pairs across Tier 1+2 inventory rows.

        This isn't a 1-to-1 with inventory rows because:
        - Some SHA-16 prefixes are duplicated within the same source (collapse to 1 file).
        - A few documents are listed under two source_collections (e.g., AML Act 2006
          under both FederalRegister and AUSTRAC), producing 2 disk files per SHA.
        """
        pairs = {(r["sha256"][:16], r["source_collection"]) for r in inventory
                 if r.get("license_tier") in {"Tier 1", "Tier 2"}}
        extracted_dir = repo_root / "data" / "extracted"
        n_extracted = sum(1 for p in extracted_dir.rglob("*.json") if p.is_file())
        assert n_extracted == len(pairs), \
            f"extracted/: {n_extracted} files but {len(pairs)} distinct (sha16, source) pairs"

    def test_chunks_count_matches_tier12(self, repo_root, inventory):
        pairs = {(r["sha256"][:16], r["source_collection"]) for r in inventory
                 if r.get("license_tier") in {"Tier 1", "Tier 2"}}
        chunks_dir = repo_root / "data" / "chunks"
        n_chunks = sum(1 for p in chunks_dir.rglob("*.jsonl") if p.is_file())
        assert n_chunks == len(pairs), \
            f"chunks/: {n_chunks} files but {len(pairs)} distinct (sha16, source) pairs"

    def test_no_absolute_paths_leaked_in_extracted(self, repo_root):
        """source_path fields in extracted JSONs must be repo-relative, never absolute."""
        import json
        extracted_dir = repo_root / "data" / "extracted"
        for p in list(extracted_dir.rglob("*.json"))[:20]:  # spot-check first 20
            data = json.loads(p.read_text())
            sp = data.get("source_path", "")
            assert not sp.startswith("/"), \
                f"{p.relative_to(repo_root)}: absolute source_path leaked: {sp[:80]}"
            assert "mdrafsunsheikh" not in sp, \
                f"{p.relative_to(repo_root)}: author-identifying path leaked: {sp[:80]}"

    def test_no_absolute_paths_leaked_in_chunks(self, repo_root):
        """source_path on every chunk must be repo-relative."""
        import json
        chunks_dir = repo_root / "data" / "chunks"
        for p in list(chunks_dir.rglob("*.jsonl"))[:20]:  # spot-check first 20
            first_line = next((l for l in p.read_text().splitlines() if l.strip()), None)
            if first_line is None:
                continue
            rec = json.loads(first_line)
            sp = rec.get("source_path", "")
            assert not sp.startswith("/"), \
                f"{p.relative_to(repo_root)}: absolute source_path leaked: {sp[:80]}"
            assert "mdrafsunsheikh" not in sp, \
                f"{p.relative_to(repo_root)}: author-identifying path leaked: {sp[:80]}"

    def test_no_tier3_extracted_shipped(self, repo_root):
        """Tier 3 sources (MAS, FATF, IRAS) must NOT have extracted/chunks files shipped."""
        for path in (repo_root / "data" / "extracted").rglob("*"):
            if not path.is_file():
                continue
            parts = path.parts
            # Reject if a Tier 3 source name appears in the path
            for tier3_source in ("FATF", "IRAS"):
                if tier3_source in parts:
                    raise AssertionError(f"Tier 3 source leaked in extracted/: {path}")
            # MAS-* sources start with MAS-
            for part in parts:
                if part.startswith("MAS-"):
                    raise AssertionError(f"Tier 3 source leaked in extracted/: {path}")


class TestInventory:
    """Document inventory consistency."""

    def test_inventory_loads(self, inventory):
        assert inventory, "inventory CSV is empty"

    def test_inventory_size(self, inventory):
        assert len(inventory) == 247, f"expected 247 documents, got {len(inventory)}"

    def test_tier_distribution_matches_audit(self, inventory):
        from collections import Counter
        tiers = Counter((r.get("license_tier") or "").strip() for r in inventory)
        # Per the licensing audit: Tier 1 = 166, Tier 2 = 26, Tier 3 = 55
        assert tiers.get("Tier 1") == 166, f"Tier 1 count: expected 166, got {tiers.get('Tier 1')}"
        assert tiers.get("Tier 2") == 26, f"Tier 2 count: expected 26, got {tiers.get('Tier 2')}"
        assert tiers.get("Tier 3") == 55, f"Tier 3 count: expected 55, got {tiers.get('Tier 3')}"
        assert sum(tiers.values()) == 247

    def test_every_inventory_row_has_sha256(self, inventory):
        for i, row in enumerate(inventory):
            sha = (row.get("sha256") or "").strip()
            assert sha, f"inventory row {i}: empty sha256"
            assert len(sha) == 64, f"inventory row {i}: malformed sha256 length"

    def test_every_inventory_row_has_tier(self, inventory):
        for i, row in enumerate(inventory):
            tier = (row.get("license_tier") or "").strip()
            assert tier in {"Tier 1", "Tier 2", "Tier 3"}, \
                f"inventory row {i}: bad license_tier={tier!r}"


class TestSplitConsistency:
    """Cross-split consistency: split_metadata.json must agree with the JSONL files."""

    def test_metadata_matches_jsonl(self, repo_root, all_records):
        import json
        meta_path = repo_root / "data" / "splits" / "split_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta_splits = {s["name"]: s["n_records"] for s in meta["splits"]}
        for name in ("train", "val", "test", "gold_iaa"):
            assert meta_splits.get(name) == len(all_records[name]), \
                f"split_metadata.json says {name}={meta_splits.get(name)} but JSONL has {len(all_records[name])}"

    def test_metadata_total_matches(self, repo_root, all_records):
        import json
        meta_path = repo_root / "data" / "splits" / "split_metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["n_total_records"] == sum(len(r) for r in all_records.values())
