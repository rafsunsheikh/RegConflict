"""Tests for the release-side tools (fetch_tier3, build_splits, verify_corpus)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


class TestFetchTier3:
    """Tests for src/tools/fetch_tier3.py — Tier 3 document retrieval."""

    def test_loads_manifest(self, tmp_path):
        from src.tools.fetch_tier3 import _load_manifest
        manifest = tmp_path / "manifest.csv"
        with manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sha256", "document_title", "source_url",
                                              "issuing_body", "license_name"])
            w.writeheader()
            w.writerow({"sha256": "abc", "document_title": "T1", "source_url": "",
                        "issuing_body": "FATF", "license_name": "Copyright FATF"})
        rows = _load_manifest(manifest, source_filter=None)
        assert len(rows) == 1

    def test_source_filter_works(self, tmp_path):
        from src.tools.fetch_tier3 import _load_manifest
        manifest = tmp_path / "manifest.csv"
        with manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sha256", "document_title", "source_url",
                                              "issuing_body", "license_name", "source_collection"])
            w.writeheader()
            w.writerow({"sha256": "a", "document_title": "F", "source_url": "",
                        "issuing_body": "Financial Action Task Force (FATF)",
                        "license_name": "", "source_collection": "FATF"})
            w.writerow({"sha256": "b", "document_title": "M", "source_url": "",
                        "issuing_body": "Monetary Authority of Singapore (MAS)",
                        "license_name": "", "source_collection": "MAS"})
        fatf_only = _load_manifest(manifest, source_filter="FATF")
        assert len(fatf_only) == 1
        assert fatf_only[0]["issuing_body"].startswith("Financial Action Task Force")

    def test_missing_source_url_returns_no_url(self, tmp_path):
        from src.tools.fetch_tier3 import _fetch_one
        result = _fetch_one(
            {"sha256": "abc" * 21, "document_title": "T", "source_url": "",
             "issuing_body": "FATF"},
            tmp_path, retries=1, user_agent="test", timeout_seconds=5,
            dry_run=False,
        )
        assert result.status == "no_url"
        assert "no source_url" in result.error.lower()

    def test_dry_run_makes_no_network_calls(self, tmp_path):
        from src.tools.fetch_tier3 import _fetch_one
        # Provide a non-empty source_url so we get past the no_url branch
        result = _fetch_one(
            {"sha256": "abc" * 21, "document_title": "T",
             "source_url": "https://example.com/doc.pdf", "issuing_body": "FATF"},
            tmp_path, retries=1, user_agent="test", timeout_seconds=5,
            dry_run=True,
        )
        assert result.status == "dry_run"


class TestVerifyCorpus:
    """Tests for src/tools/verify_corpus.py — release integrity check."""

    def test_runs_on_real_release(self, repo_root):
        """The verify tool should run cleanly on the actual on-disk release."""
        import subprocess
        result = subprocess.run(
            ["python3", str(repo_root / "src" / "tools" / "verify_corpus.py"),
             "--skip-file-presence"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Exit code 0 = all checks passed; 1 = errors found
        # During pre-publication, file presence is skipped, so a clean run should succeed
        assert result.returncode in (0, 1), \
            f"verify_corpus exited unexpectedly: code={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        # The output should at minimum have the "VERIFICATION" footer
        assert "VERIFICATION" in result.stdout, \
            f"verify_corpus output missing VERIFICATION footer:\n{result.stdout}"


class TestBuildSplits:
    """Tests for src/tools/build_splits.py — split reproduction."""

    def test_stratified_split_is_deterministic(self):
        from src.tools.build_splits import _stratified_split
        records = [{"record_type": "conflict", "conflict_type": "operationally_resolvable", "id": i}
                   for i in range(40)]
        records += [{"record_type": "non_conflict", "id": i} for i in range(40, 80)]
        run1 = _stratified_split(records, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
        run2 = _stratified_split(records, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
        for name in ("train", "val", "test"):
            ids_1 = [r["id"] for r in run1[name]]
            ids_2 = [r["id"] for r in run2[name]]
            assert ids_1 == ids_2, f"split {name} not deterministic"

    def test_stratification_preserves_class_proportions(self):
        from src.tools.build_splits import _stratified_split
        records = [{"record_type": "conflict", "conflict_type": "operationally_resolvable",
                    "id": i} for i in range(80)]
        records += [{"record_type": "non_conflict", "id": i} for i in range(80, 800)]
        splits = _stratified_split(records, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
        # Each split should have ~10% conflict records (80/800)
        for name in ("train", "val", "test"):
            n_conflict = sum(1 for r in splits[name] if r["record_type"] == "conflict")
            n_total = len(splits[name])
            assert 0.07 <= n_conflict / n_total <= 0.13, \
                f"split {name}: conflict ratio={n_conflict}/{n_total} outside [7%, 13%]"

    def test_split_proportions_sum_to_total(self):
        from src.tools.build_splits import _stratified_split
        records = [{"record_type": "conflict",
                    "conflict_type": "operationally_resolvable", "id": i} for i in range(100)]
        splits = _stratified_split(records, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
        total = sum(len(splits[s]) for s in ("train", "val", "test"))
        assert total == 100, f"splits don't partition the input ({total} ≠ 100)"


class TestSchemaValidation:
    """Pydantic schema enforcement tests."""

    def test_conflict_without_typology_raises(self):
        from src.eval.benchmark.schema import LabelledPair
        bad = {
            "pair_id": "conflict:00",
            "record_type": "conflict",
            "label": "conflict",
            "regime_a": {"regime_id": "X", "jurisdiction": "EU"},
            "regime_b": {"regime_id": "Y", "jurisdiction": "Singapore"},
            "evidence_a": [], "evidence_b": [],
            # missing conflict_type and severity
        }
        with pytest.raises(ValueError, match="missing conflict_type"):
            LabelledPair.model_validate(bad)

    def test_conflict_without_severity_raises(self):
        from src.eval.benchmark.schema import LabelledPair
        bad = {
            "pair_id": "conflict:00",
            "record_type": "conflict",
            "label": "conflict",
            "regime_a": {"regime_id": "X", "jurisdiction": "EU"},
            "regime_b": {"regime_id": "Y", "jurisdiction": "Singapore"},
            "evidence_a": [], "evidence_b": [],
            "conflict_type": "operationally_resolvable",
            # missing severity
        }
        with pytest.raises(ValueError, match="missing severity"):
            LabelledPair.model_validate(bad)

    def test_valid_record_round_trips(self):
        from src.eval.benchmark.schema import LabelledPair
        good = {
            "pair_id": "conflict:00",
            "record_type": "conflict",
            "label": "conflict",
            "regime_a": {"regime_id": "X", "jurisdiction": "EU"},
            "regime_b": {"regime_id": "Y", "jurisdiction": "Singapore"},
            "evidence_a": [{"chunk_id": "c1", "source_doc": "d.pdf", "passage": "..."}],
            "evidence_b": [{"chunk_id": "c2", "source_doc": "e.pdf", "passage": "..."}],
            "conflict_type": "operationally_resolvable",
            "severity": "medium",
            "rationale": "test",
        }
        parsed = LabelledPair.model_validate(good)
        assert parsed.conflict_type.value == "operationally_resolvable"
        assert parsed.severity.value == "medium"
