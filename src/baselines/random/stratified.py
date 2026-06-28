"""Stratified-random baseline.

Probabilistic predictor that samples from the training class distribution:
  predicted_conflict_present ~ Bernoulli(P(conflict))
  predicted_conflict_type    ~ Categorical(P(type | conflict))   [if positive]

Deterministic given the seed. confidence_score is set to the Bernoulli
probability so the harness can compute ROC-AUC (which will hover near 0.5
since the predictor uses a single shared probability for all records).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

CONFLICT_TYPES = (
    "structural_unresolved",
    "operationally_resolvable",
    "interpretive_fact_sensitive",
    "recurring_friction",
)


def compute_distribution(train_records: Iterable[dict]) -> dict:
    """Estimate P(conflict_present) and P(conflict_type | conflict_present) from train."""
    train_records = list(train_records)
    n_total = len(train_records)
    n_conflict = sum(1 for r in train_records if r.get("record_type") == "conflict")
    p_conflict = n_conflict / max(n_total, 1)

    type_counts = Counter(
        r.get("conflict_type")
        for r in train_records
        if r.get("record_type") == "conflict"
    )
    type_probs = {
        t: type_counts.get(t, 0) / max(n_conflict, 1) for t in CONFLICT_TYPES
    }
    return {
        "p_conflict": p_conflict,
        "type_probs": type_probs,
        "n_total": n_total,
        "n_conflict": n_conflict,
        "type_counts": dict(type_counts),
    }


def predict(
    train_records: list[dict],
    eval_records: list[dict],
    seed: int,
) -> tuple[list[dict], dict]:
    """Generate stratified-random predictions deterministically given the seed.

    Returns (predictions, stats) where stats includes the estimated training
    distribution and the seed used.
    """
    stats = compute_distribution(train_records)
    rng = np.random.default_rng(seed)
    p = stats["p_conflict"]
    types = list(CONFLICT_TYPES)
    type_p = np.asarray([stats["type_probs"][t] for t in types], dtype=float)
    if type_p.sum() > 0:
        type_p = type_p / type_p.sum()
    else:
        type_p = np.full(len(types), 1.0 / len(types))

    preds = []
    for r in eval_records:
        is_conflict = bool(rng.random() < p)
        conflict_type = None
        if is_conflict:
            conflict_type = str(rng.choice(types, p=type_p))
        preds.append(
            {
                "pair_id": r["pair_id"],
                "predicted_conflict_present": is_conflict,
                "predicted_conflict_type": conflict_type,
                # Confidence == P(conflict) used by the predictor; constant across
                # records so AUC will hover at 0.5.
                "confidence_score": float(p),
                "model_metadata": {
                    "model_name": "stratified_random",
                    "model_version": "1.0",
                    "config_hash": f"stratified_random_v1_seed{seed}",
                },
            }
        )
    stats["seed"] = seed
    return preds, stats
