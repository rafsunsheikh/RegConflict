"""Main evaluation orchestration: predictions in → all artefacts out."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np

from src.eval.benchmark.bootstrap import bootstrap_ci
from src.eval.benchmark.io import (
    load_predictions,
    load_split,
    make_output_dir,
    write_results,
)
from src.eval.benchmark.metrics import (
    CONFLICT_TYPES,
    compute_binary_metrics,
    compute_joint_metrics,
    compute_per_jurisdiction_metrics,
    compute_typology_metrics,
)
from src.eval.benchmark.validators import (
    check_config_hash_consistency,
    validate_predictions,
)


def _juris_pair(record: dict) -> tuple[str, str]:
    a = (record.get("regime_a") or {}).get("jurisdiction", "?")
    b = (record.get("regime_b") or {}).get("jurisdiction", "?")
    return tuple(sorted([a, b]))


def _align_predictions(
    split_records: list[dict], predictions: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Return (split_aligned, pred_aligned), where indices correspond pair-wise.

    Pairs missing from predictions are filled with a default no-prediction
    record (conflict_present=False, type=None, confidence=None) so that metrics
    treat omissions as negative predictions. The harness alerts about omissions
    via the validators upstream.
    """
    pred_by_id = {p["pair_id"]: p for p in predictions}
    aligned_split = []
    aligned_pred = []
    for r in split_records:
        aligned_split.append(r)
        aligned_pred.append(
            pred_by_id.get(
                r["pair_id"],
                {
                    "pair_id": r["pair_id"],
                    "predicted_conflict_present": False,
                    "predicted_conflict_type": None,
                    "confidence_score": None,
                    "_filled_default": True,
                },
            )
        )
    return aligned_split, aligned_pred


def _build_arrays(split_recs: list[dict], pred_recs: list[dict]):
    binary_true = np.array(
        [r.get("record_type") == "conflict" for r in split_recs], dtype=bool
    )
    binary_pred = np.array(
        [bool(p.get("predicted_conflict_present")) for p in pred_recs], dtype=bool
    )
    type_true = [
        r.get("conflict_type") if r.get("record_type") == "conflict" else None
        for r in split_recs
    ]
    type_pred = [p.get("predicted_conflict_type") for p in pred_recs]
    confidence = [p.get("confidence_score") for p in pred_recs]
    pair_ids = [r["pair_id"] for r in split_recs]
    juris_pairs = [_juris_pair(r) for r in split_recs]
    return binary_true, binary_pred, type_true, type_pred, confidence, pair_ids, juris_pairs


def evaluate(
    predictions_path: Path,
    split: str,
    output_base: Path,
    model_name: str | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Run full evaluation and write all output artefacts. Returns the metrics dict."""
    log_buf = StringIO()
    def log(msg: str) -> None:
        print(msg)
        log_buf.write(msg + "\n")

    log(f"=== TRIAG conflict-detection benchmark eval ===")
    log(f"predictions: {predictions_path}")
    log(f"split:       {split}")

    split_records = load_split(split)
    predictions = load_predictions(predictions_path)
    log(f"split records: {len(split_records)}")
    log(f"predictions:   {len(predictions)}")

    # Validate
    validation = validate_predictions(split_records, predictions)
    log(f"aligned: {validation['aligned']}")
    if validation["missing_pair_ids"]:
        log(
            f"WARNING: {len(validation['missing_pair_ids'])} split pair_ids missing "
            f"from predictions (treated as conflict_present=False)"
        )
    if validation["extra_pair_ids"]:
        log(
            f"WARNING: {len(validation['extra_pair_ids'])} prediction pair_ids "
            f"not in the split (ignored)"
        )
    if validation["type_set_without_conflict"]:
        log(
            f"WARNING: {len(validation['type_set_without_conflict'])} predictions "
            f"have predicted_conflict_type set while predicted_conflict_present=False"
        )

    # Resolve model name from predictions metadata if not given
    if model_name is None:
        for p in predictions:
            md = p.get("model_metadata") or {}
            if md.get("model_name"):
                model_name = md["model_name"]
                break
        if model_name is None:
            model_name = "unknown_model"

    # Align and build arrays
    split_aligned, pred_aligned = _align_predictions(split_records, predictions)
    bt, bp, tt, tp, conf, pids, jps = _build_arrays(split_aligned, pred_aligned)

    # ===== Point-estimate metrics =====
    binary_point = compute_binary_metrics(bt.tolist(), bp.tolist(), conf)
    typology_point = compute_typology_metrics(tt, tp)
    joint_point = compute_joint_metrics(bt.tolist(), bp.tolist(), tt, tp)
    per_juris = compute_per_jurisdiction_metrics(pids, jps, bt.tolist(), bp.tolist())

    # ===== Bootstrap CIs =====
    n = len(bt)

    def _bin_metric(name: str):
        def f(idx):
            yt = bt[idx]
            yp = bp[idx]
            if yt.sum() == 0 and name in {"precision_conflict", "recall_conflict", "f1_conflict"}:
                return 0.0
            m = compute_binary_metrics(yt.tolist(), yp.tolist())
            return m[name]
        return f

    def _macro_f1():
        return _bin_metric("macro_f1")

    def _acc():
        return _bin_metric("accuracy")

    def _typology_macro(idx):
        sub_tt = [tt[i] for i in idx]
        sub_tp = [tp[i] for i in idx]
        m = compute_typology_metrics(sub_tt, sub_tp)
        return m["macro_f1"] if m["macro_f1"] is not None else 0.0

    def _joint_acc(idx):
        m = compute_joint_metrics(
            bt[idx].tolist(),
            bp[idx].tolist(),
            [tt[i] for i in idx],
            [tp[i] for i in idx],
        )
        return m["exact_match_accuracy"]

    bootstrap_results = {
        "binary": {
            "f1_conflict": bootstrap_ci(_bin_metric("f1_conflict"), n, n_bootstrap, seed=seed),
            "precision_conflict": bootstrap_ci(_bin_metric("precision_conflict"), n, n_bootstrap, seed=seed),
            "recall_conflict": bootstrap_ci(_bin_metric("recall_conflict"), n, n_bootstrap, seed=seed),
            "macro_f1": bootstrap_ci(_macro_f1(), n, n_bootstrap, seed=seed),
            "accuracy": bootstrap_ci(_acc(), n, n_bootstrap, seed=seed),
        },
        "typology": {
            "macro_f1": bootstrap_ci(_typology_macro, n, n_bootstrap, seed=seed),
        },
        "joint": {
            "exact_match_accuracy": bootstrap_ci(_joint_acc, n, n_bootstrap, seed=seed),
        },
    }

    # ===== Errors list =====
    errors = []
    for sr, pr in zip(split_aligned, pred_aligned):
        true_bin = sr.get("record_type") == "conflict"
        true_type = sr.get("conflict_type") if true_bin else None
        pred_bin = bool(pr.get("predicted_conflict_present"))
        pred_type = pr.get("predicted_conflict_type")
        is_error = (true_bin != pred_bin) or (true_bin and true_type != pred_type)
        if is_error:
            errors.append(
                {
                    "pair_id": sr["pair_id"],
                    "regime_a_id": (sr.get("regime_a") or {}).get("regime_id"),
                    "regime_b_id": (sr.get("regime_b") or {}).get("regime_id"),
                    "true_conflict_present": true_bin,
                    "true_conflict_type": true_type,
                    "predicted_conflict_present": pred_bin,
                    "predicted_conflict_type": pred_type,
                    "confidence_score": pr.get("confidence_score"),
                    "rationale_snippet": (sr.get("rationale", "") or "")[:240],
                }
            )

    # ===== Config snapshot =====
    sample_pred = predictions[0] if predictions else {}
    model_metadata = sample_pred.get("model_metadata") or {}
    config = {
        "model_name": model_name,
        "split": split,
        "predictions_path": str(predictions_path),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "model_metadata_sample": model_metadata,
        "n_split_records": len(split_records),
        "n_predictions": len(predictions),
        "n_errors": len(errors),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # ===== Config-hash determinism check =====
    cfg_hash = model_metadata.get("config_hash")
    consistency = None
    if cfg_hash:
        summary = {
            "binary_f1": binary_point["f1_conflict"],
            "typology_macro_f1": typology_point.get("macro_f1"),
            "joint_acc": joint_point["exact_match_accuracy"],
        }
        consistency = check_config_hash_consistency(cfg_hash, split, summary)
        if consistency.get("warning"):
            log(f"WARNING: {consistency['warning']}")

    metrics = {
        "binary": {"point": binary_point, "bootstrap": bootstrap_results["binary"]},
        "typology": {"point": typology_point, "bootstrap": bootstrap_results["typology"]},
        "joint": {"point": joint_point, "bootstrap": bootstrap_results["joint"]},
        "validation": validation,
        "consistency_check": consistency,
    }

    confusion_matrix = {
        "labels": list(CONFLICT_TYPES),
        "matrix": typology_point.get("confusion_matrix"),
        "n_eval": typology_point.get("n_eval"),
    }

    # ===== Write artefacts =====
    output_dir = make_output_dir(output_base, model_name, split)

    # ===== Human-readable log =====
    log("")
    log("--- BINARY (conflict present vs not) ---")
    bp_ = binary_point
    log(f"  precision: {bp_['precision_conflict']:.4f}")
    log(f"  recall:    {bp_['recall_conflict']:.4f}")
    log(f"  F1 (conflict): {bp_['f1_conflict']:.4f}   "
        f"CI [{bootstrap_results['binary']['f1_conflict']['ci_lower']:.4f}, "
        f"{bootstrap_results['binary']['f1_conflict']['ci_upper']:.4f}]")
    log(f"  macro-F1:  {bp_['macro_f1']:.4f}")
    log(f"  accuracy:  {bp_['accuracy']:.4f}")
    log(f"  ROC-AUC:   {bp_.get('roc_auc')}")
    log(f"  support:   {bp_['support_pos']} pos / {bp_['support_neg']} neg")
    log("")
    log("--- TYPOLOGY (4-class, on gold-positive subset) ---")
    if typology_point["n_eval"] == 0:
        log("  no gold-positive examples in this split — typology metrics undefined")
    else:
        log(f"  n_eval: {typology_point['n_eval']}")
        for cls, vals in typology_point["per_class"].items():
            log(f"  {cls:<32} P={vals['precision']:.3f}  R={vals['recall']:.3f}  "
                f"F1={vals['f1']:.3f}  (n={vals['support']})")
        log(f"  macro-F1:  {typology_point['macro_f1']:.4f}   "
            f"CI [{bootstrap_results['typology']['macro_f1']['ci_lower']:.4f}, "
            f"{bootstrap_results['typology']['macro_f1']['ci_upper']:.4f}]")
        log(f"  accuracy:  {typology_point['accuracy']:.4f}")
    log("")
    log("--- JOINT (exact-match on binary+type) ---")
    log(f"  accuracy: {joint_point['exact_match_accuracy']:.4f}   "
        f"({joint_point['n_correct']}/{joint_point['n_total']})")
    log("")
    log("--- PER-JURISDICTION-PAIR F1 ---")
    for jp, m in sorted(per_juris.items(), key=lambda kv: -kv[1]["support"]):
        log(f"  {jp:<40} support={m['support']:>4}  pos={m['n_conflict']:>3}  "
            f"F1(conflict)={m['f1_conflict']:.3f}  macro={m['macro_f1']:.3f}")
    log("")
    log(f"errors: {len(errors)}")
    log(f"output: {output_dir}")

    write_results(
        output_dir=output_dir,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        per_jurisdiction=per_juris,
        errors=errors,
        config=config,
        log_text=log_buf.getvalue(),
    )

    return {
        "metrics": metrics,
        "per_jurisdiction": per_juris,
        "errors": errors,
        "output_dir": str(output_dir),
    }
