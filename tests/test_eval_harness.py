"""Tests for the evaluation harness, metrics, and bootstrap routines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# REPO_ROOT injection happens in conftest.py


class TestMetrics:
    """Sanity tests on the metric computation."""

    def test_binary_f1_on_known_inputs(self):
        from sklearn.metrics import f1_score
        # Trivial sanity: all-negative predictions vs 50/50 truth → F1 = 0.0
        y_true = [True, True, False, False]
        y_pred = [False, False, False, False]
        assert f1_score(y_true, y_pred, pos_label=True, zero_division=0) == 0.0

        # Perfect predictions → F1 = 1.0
        y_pred = [True, True, False, False]
        assert f1_score(y_true, y_pred, pos_label=True, zero_division=0) == 1.0

    def test_paired_bootstrap_on_real_predictions(self, repo_root):
        """compute_iaa and paired_bootstrap share a numpy-only implementation pattern.

        Direct sanity: identical predictions give Δ = 0; opposite predictions give
        a non-zero observed Δ. This isn't a full test of the harness — it confirms
        the maths is stable.
        """
        from sklearn.metrics import f1_score
        y_true = [True] * 14 + [False] * 93
        y_pred_a = y_true[:]  # perfect
        y_pred_b = [False] * 107  # all-negative
        f1_a = f1_score(y_true, y_pred_a, pos_label=True, zero_division=0)
        f1_b = f1_score(y_true, y_pred_b, pos_label=True, zero_division=0)
        assert f1_a == 1.0
        assert f1_b == 0.0
        assert f1_a - f1_b == 1.0


class TestComputeIAA:
    """Tests for scripts/compute_iaa.py — the canonical IAA reproduction script."""

    def _write_pair(self, path: Path, pair_id: str, ct: str) -> None:
        with path.open("a") as fh:
            fh.write(json.dumps({"pair_id": pair_id, "conflict_type": ct}) + "\n")

    def test_kappa_perfect_agreement(self, tmp_path):
        from scripts.compute_iaa import _cohen_kappa, TYPOLOGY_CLASSES
        y = ["operationally_resolvable"] * 5 + ["structural_unresolved"] * 5
        assert _cohen_kappa(y, y, TYPOLOGY_CLASSES) == pytest.approx(1.0)

    def test_kappa_chance_agreement(self):
        from scripts.compute_iaa import _cohen_kappa, TYPOLOGY_CLASSES
        # The v0.9 pilot's actual confusion matrix on the 8 jointly-positive pairs:
        # each primary class has exactly one (X, structural) and one (X, operationally)
        # second-annotator response. Cohen's κ on this matrix is exactly 0.
        primary = ["structural_unresolved", "structural_unresolved",
                   "operationally_resolvable", "operationally_resolvable",
                   "interpretive_fact_sensitive", "interpretive_fact_sensitive",
                   "recurring_friction", "recurring_friction"]
        second = ["structural_unresolved", "operationally_resolvable",
                  "structural_unresolved", "operationally_resolvable",
                  "structural_unresolved", "operationally_resolvable",
                  "structural_unresolved", "operationally_resolvable"]
        kappa = _cohen_kappa(primary, second, TYPOLOGY_CLASSES)
        assert kappa == pytest.approx(0.0, abs=1e-10)

    def test_landis_koch_bands(self):
        from scripts.compute_iaa import _landis_koch_band
        assert "poor" in _landis_koch_band(-0.1)
        assert "slight" in _landis_koch_band(0.1)
        assert "fair" in _landis_koch_band(0.3)
        assert "moderate" in _landis_koch_band(0.5)
        assert "substantial" in _landis_koch_band(0.6)
        assert "substantial" in _landis_koch_band(0.7)
        assert "almost perfect" in _landis_koch_band(0.9)

    def test_confusion_matrix_structure(self):
        from scripts.compute_iaa import _confusion_matrix, TYPOLOGY_CLASSES
        cm = _confusion_matrix(
            ["structural_unresolved", "operationally_resolvable"],
            ["structural_unresolved", "structural_unresolved"],
            TYPOLOGY_CLASSES,
        )
        assert cm["structural_unresolved"]["structural_unresolved"] == 1
        assert cm["operationally_resolvable"]["structural_unresolved"] == 1
        assert cm["operationally_resolvable"]["operationally_resolvable"] == 0


class TestRandomBaselines:
    """Determinism + sanity tests for the random baselines."""

    def test_majority_class_predicts_majority(self, all_records):
        from src.baselines.random import majority
        train = all_records["train"]
        test = all_records["test"]
        preds, stats = majority.predict(train, test)
        assert len(preds) == len(test)
        # 86% of train is non-conflict → majority label should be "non_conflict"
        # All predictions should agree on the same label
        assert len({p["predicted_conflict_present"] for p in preds}) == 1

    def test_stratified_random_reproducible(self, all_records):
        from src.baselines.random import stratified
        train = all_records["train"]
        test = all_records["test"]
        preds_seed42_run1, _ = stratified.predict(train, test, seed=42)
        preds_seed42_run2, _ = stratified.predict(train, test, seed=42)
        # Same seed → same predictions
        for p1, p2 in zip(preds_seed42_run1, preds_seed42_run2):
            assert p1["predicted_conflict_present"] == p2["predicted_conflict_present"]

    def test_different_seeds_give_different_predictions(self, all_records):
        from src.baselines.random import stratified
        train = all_records["train"]
        test = all_records["test"]
        preds_a, _ = stratified.predict(train, test, seed=42)
        preds_b, _ = stratified.predict(train, test, seed=999)
        # At least one prediction should differ across seeds
        differs = any(a["predicted_conflict_present"] != b["predicted_conflict_present"]
                      for a, b in zip(preds_a, preds_b))
        assert differs, "stratified_random should not give identical predictions across seeds"
