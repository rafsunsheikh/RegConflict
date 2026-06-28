"""Majority-class baseline.

Deterministic predictor that always emits the most-frequent training label.
For the conflict-detection benchmark, the binary majority is almost certainly
'non_conflict' (~87% prior), so this baseline never predicts a conflict —
binary F1 will be zero, but accuracy will track the negative-class rate.

The conflict-type majority (computed over training conflicts only) is included
for completeness; it is only emitted if the binary majority is 'conflict',
which on this dataset never happens.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

CONFLICT_TYPES = (
    "structural_unresolved",
    "operationally_resolvable",
    "interpretive_fact_sensitive",
    "recurring_friction",
)


def compute_majority(train_records: Iterable[dict]) -> dict:
    """Return the majority binary class and majority conflict_type from train."""
    train_records = list(train_records)
    binary_counts = Counter(
        "conflict" if r.get("record_type") == "conflict" else "non_conflict"
        for r in train_records
    )
    majority_binary = binary_counts.most_common(1)[0][0]
    # Conflict-type majority over the positive subset
    conflict_records = [r for r in train_records if r.get("record_type") == "conflict"]
    type_counts = Counter(r.get("conflict_type") for r in conflict_records)
    majority_type = type_counts.most_common(1)[0][0] if type_counts else None
    return {
        "majority_binary": majority_binary,
        "majority_type": majority_type,
        "binary_counts": dict(binary_counts),
        "type_counts": dict(type_counts),
    }


def predict(
    train_records: list[dict],
    eval_records: list[dict],
) -> tuple[list[dict], dict]:
    """Generate majority-class predictions for every record in eval_records.

    Returns (predictions, stats) where predictions is a JSONL-ready list and
    stats captures the training distribution and majority labels used.
    """
    stats = compute_majority(train_records)
    binary_majority_is_conflict = stats["majority_binary"] == "conflict"
    type_pred = stats["majority_type"] if binary_majority_is_conflict else None
    preds = [
        {
            "pair_id": r["pair_id"],
            "predicted_conflict_present": binary_majority_is_conflict,
            "predicted_conflict_type": type_pred,
            "confidence_score": None,  # not a probabilistic model
            "model_metadata": {
                "model_name": "majority_class",
                "model_version": "1.0",
                "config_hash": "majority_class_v1",
            },
        }
        for r in eval_records
    ]
    return preds, stats
