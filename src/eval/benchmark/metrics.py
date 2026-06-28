"""Metric computations for the conflict-detection benchmark.

All metrics follow the paper's reporting convention: macro-averaging by default.
Computations delegate to scikit-learn where possible.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

CONFLICT_TYPES: tuple[str, ...] = (
    "structural_unresolved",
    "operationally_resolvable",
    "interpretive_fact_sensitive",
    "recurring_friction",
)


# ----------------------------------------------------------------------
# Binary task: conflict present vs not
# ----------------------------------------------------------------------
def compute_binary_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    confidence: list[float | None] | None = None,
) -> dict:
    """Precision, recall, F1 (positive class = conflict), macro-F1, accuracy, AUC.

    The positive class is `True` (conflict present). Macro-F1 averages the
    F1 of present and absent classes.
    """
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)

    # Positive-class (conflict) metrics
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, average="binary", pos_label=True, zero_division=0
    )
    macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    acc = accuracy_score(yt, yp)

    out = {
        "precision_conflict": float(p),
        "recall_conflict": float(r),
        "f1_conflict": float(f1),
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "support_pos": int(yt.sum()),
        "support_neg": int(len(yt) - yt.sum()),
    }

    # ROC-AUC only if every prediction has a confidence score
    if confidence is not None and all(c is not None for c in confidence):
        try:
            out["roc_auc"] = float(roc_auc_score(yt, np.asarray(confidence, dtype=float)))
        except ValueError:
            # Only one class present in y_true — AUC undefined
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None

    return out


# ----------------------------------------------------------------------
# Typology task: 4-class on the conflict subset
# ----------------------------------------------------------------------
def compute_typology_metrics(
    y_true: list[str | None],
    y_pred: list[str | None],
) -> dict:
    """Per-class precision/recall/F1, macro-F1, top-1 accuracy, confusion matrix.

    Inputs are aligned lists of conflict_type strings (or None if not a
    conflict). Pairs where y_true is None are skipped — typology metrics
    are defined ONLY on the gold-positive subset (spec §2 — Part 2).
    """
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None]
    if not pairs:
        return {
            "n_eval": 0,
            "per_class": {},
            "macro_f1": None,
            "accuracy": None,
            "confusion_matrix": None,
        }
    yt = [t for t, _ in pairs]
    yp = [p if p is not None else "__none__" for _, p in pairs]

    labels = list(CONFLICT_TYPES)
    p, r, f, support = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0
    )
    per_class = {
        cls: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(labels)
    }
    macro_f1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(yt, yp)
    cm = confusion_matrix(yt, yp, labels=labels).tolist()

    return {
        "n_eval": len(pairs),
        "per_class": per_class,
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "labels": labels,
        "confusion_matrix": cm,
    }


# ----------------------------------------------------------------------
# Joint task: exact-match on (binary, type)
# ----------------------------------------------------------------------
def compute_joint_metrics(
    binary_true: list[bool],
    binary_pred: list[bool],
    type_true: list[str | None],
    type_pred: list[str | None],
) -> dict:
    """Exact-match accuracy on the (conflict_present, conflict_type) pair.

    A pair is correct iff:
      - binary prediction matches binary truth AND
      - if truth is a conflict, type prediction matches type truth
      - if truth is non-conflict, type prediction is None
    """
    n = len(binary_true)
    correct = 0
    for bt, bp, tt, tp in zip(binary_true, binary_pred, type_true, type_pred):
        if bt != bp:
            continue
        if bt:
            if tt == tp:
                correct += 1
        else:
            if tp is None:
                correct += 1
    return {
        "exact_match_accuracy": correct / n if n else 0.0,
        "n_correct": correct,
        "n_total": n,
    }


# ----------------------------------------------------------------------
# Per-jurisdiction-pair F1
# ----------------------------------------------------------------------
def compute_per_jurisdiction_metrics(
    pair_ids: list[str],
    juris_pairs: list[tuple[str, str]],
    y_true: list[bool],
    y_pred: list[bool],
) -> dict:
    """Binary F1 broken down by canonical jurisdiction-pair (e.g. AU×EU).

    Returns one entry per jurisdiction-pair with support and F1. Jurisdiction
    pairs are canonicalised (sorted) so AU×EU == EU×AU.
    """
    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"y_true": [], "y_pred": [], "pair_ids": []}
    )
    for pid, jp, yt, yp in zip(pair_ids, juris_pairs, y_true, y_pred):
        jp_canon = tuple(sorted(jp))
        grouped[jp_canon]["y_true"].append(yt)
        grouped[jp_canon]["y_pred"].append(yp)
        grouped[jp_canon]["pair_ids"].append(pid)

    out = {}
    for jp, data in grouped.items():
        yt_arr = np.asarray(data["y_true"], dtype=bool)
        yp_arr = np.asarray(data["y_pred"], dtype=bool)
        f1 = f1_score(yt_arr, yp_arr, pos_label=True, zero_division=0)
        macro = f1_score(yt_arr, yp_arr, average="macro", zero_division=0)
        out[f"{jp[0]}__{jp[1]}"] = {
            "support": int(len(yt_arr)),
            "n_conflict": int(yt_arr.sum()),
            "f1_conflict": float(f1),
            "macro_f1": float(macro),
        }
    return out
