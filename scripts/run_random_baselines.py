"""Run both random baselines end-to-end: predictions → harness → comparison table.

Outputs:
  eval/results/majority_class/test_predictions.jsonl
  eval/results/majority_class/gold_iaa_predictions.jsonl
  eval/results/majority_class/test_*/metrics.json    (harness)
  eval/results/majority_class/gold_iaa_*/metrics.json

  eval/results/stratified_random/seed_<seed>/test_predictions.jsonl     (100×)
  eval/results/stratified_random/seed_<seed>/gold_iaa_predictions.jsonl (100×)
  eval/results/stratified_random/runs/...                               (harness output, per seed)
  eval/results/stratified_random/aggregate_metrics.json                 (cross-seed summary)

A side-by-side comparison table is printed to stdout. The representative
single-seed bootstrap CI is reported for seed=42.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.random import majority, stratified  # noqa: E402
from src.eval.benchmark.io import load_split  # noqa: E402
from src.eval.benchmark.runner import evaluate  # noqa: E402

RESULTS_BASE = REPO_ROOT / "results"
N_SEEDS = 100
REPRESENTATIVE_SEED = 42


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _extract_key_metrics(metrics: dict) -> dict:
    """Pull the headline numbers out of a harness metrics dict."""
    bp = metrics["binary"]["point"]
    tp = metrics["typology"]["point"]
    jp = metrics["joint"]["point"]
    return {
        "binary_f1": bp["f1_conflict"],
        "binary_precision": bp["precision_conflict"],
        "binary_recall": bp["recall_conflict"],
        "macro_f1": bp["macro_f1"],
        "accuracy": bp["accuracy"],
        "roc_auc": bp.get("roc_auc"),
        "typology_macro_f1": tp.get("macro_f1"),
        "typology_accuracy": tp.get("accuracy"),
        "joint_em": jp["exact_match_accuracy"],
    }


# --------------------------------------------------------------------------
# Majority class
# --------------------------------------------------------------------------
def run_majority(train: list[dict], splits: dict[str, list[dict]], n_bootstrap: int) -> dict:
    out_dir = RESULTS_BASE / "majority_class"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_test, stats = majority.predict(train, splits["test"])
    preds_gold, _ = majority.predict(train, splits["gold_iaa"])
    pred_path_test = out_dir / "test_predictions.jsonl"
    pred_path_gold = out_dir / "gold_iaa_predictions.jsonl"
    _write_jsonl(preds_test, pred_path_test)
    _write_jsonl(preds_gold, pred_path_gold)

    print(f"\n=== Majority-class baseline ===")
    print(f"  train binary distribution:   {stats['binary_counts']}")
    print(f"  majority binary class:       {stats['majority_binary']}")
    print(f"  train conflict-type counts:  {stats['type_counts']}")
    print(f"  majority conflict type:      {stats['majority_type']}")
    print(f"  predictions written:         {pred_path_test}")

    # Evaluate
    result_test = evaluate(
        predictions_path=pred_path_test,
        split="test",
        output_base=RESULTS_BASE,
        model_name="majority_class",
        n_bootstrap=n_bootstrap,
    )
    result_gold = evaluate(
        predictions_path=pred_path_gold,
        split="gold_iaa",
        output_base=RESULTS_BASE,
        model_name="majority_class",
        n_bootstrap=n_bootstrap,
    )
    return {
        "stats": stats,
        "test_metrics": result_test["metrics"],
        "gold_iaa_metrics": result_gold["metrics"],
    }


# --------------------------------------------------------------------------
# Stratified random
# --------------------------------------------------------------------------
def run_stratified(
    train: list[dict],
    splits: dict[str, list[dict]],
    n_seeds: int,
    n_bootstrap_per_seed: int,
    n_bootstrap_representative: int,
) -> dict:
    base = RESULTS_BASE / "stratified_random"
    base.mkdir(parents=True, exist_ok=True)
    runs_dir = base / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stratified-random baseline ({n_seeds} seeds) ===")
    # Compute training distribution once for reporting
    dist = stratified.compute_distribution(train)
    print(f"  P(conflict) = {dist['p_conflict']:.4f}")
    print(f"  P(type | conflict):")
    for t, p in dist["type_probs"].items():
        print(f"    {t:<35} {p:.4f}")
    print(f"  running {n_seeds} seeds × 2 splits = {2*n_seeds} harness evaluations...")

    per_seed_test = []
    per_seed_gold = []
    representative_test = None
    representative_gold = None

    for seed in range(n_seeds):
        preds_test, _ = stratified.predict(train, splits["test"], seed=seed)
        preds_gold, _ = stratified.predict(train, splits["gold_iaa"], seed=seed)
        pred_path_test = base / f"seed_{seed:03d}" / "test_predictions.jsonl"
        pred_path_gold = base / f"seed_{seed:03d}" / "gold_iaa_predictions.jsonl"
        _write_jsonl(preds_test, pred_path_test)
        _write_jsonl(preds_gold, pred_path_gold)

        # Bootstrap budget: full for representative seed, lighter for the rest
        nbt = (
            n_bootstrap_representative
            if seed == REPRESENTATIVE_SEED
            else n_bootstrap_per_seed
        )
        rt = evaluate(
            predictions_path=pred_path_test,
            split="test",
            output_base=runs_dir,
            model_name=f"seed_{seed:03d}",
            n_bootstrap=nbt,
        )
        rg = evaluate(
            predictions_path=pred_path_gold,
            split="gold_iaa",
            output_base=runs_dir,
            model_name=f"seed_{seed:03d}",
            n_bootstrap=nbt,
        )
        mt = _extract_key_metrics(rt["metrics"])
        mg = _extract_key_metrics(rg["metrics"])
        mt["seed"] = seed
        mg["seed"] = seed
        per_seed_test.append(mt)
        per_seed_gold.append(mg)
        if seed == REPRESENTATIVE_SEED:
            representative_test = rt["metrics"]
            representative_gold = rg["metrics"]

    # Empirical class proportions across all seeds (sanity vs training prior)
    n_positives = 0
    n_total = 0
    type_emit_counts = {t: 0 for t in stratified.CONFLICT_TYPES}
    for seed in range(n_seeds):
        pred_path = base / f"seed_{seed:03d}" / "test_predictions.jsonl"
        rows = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]
        for r in rows:
            n_total += 1
            if r.get("predicted_conflict_present"):
                n_positives += 1
                t = r.get("predicted_conflict_type")
                if t in type_emit_counts:
                    type_emit_counts[t] += 1
    emp_p_conflict = n_positives / max(n_total, 1)
    emp_type_probs = {
        t: type_emit_counts[t] / max(n_positives, 1) for t in stratified.CONFLICT_TYPES
    }

    # Aggregate cross-seed stats
    def _aggregate(per_seed_list: list[dict]) -> dict:
        keys = [k for k in per_seed_list[0].keys() if k != "seed"]
        agg = {}
        for k in keys:
            vals = [
                d[k] for d in per_seed_list
                if d.get(k) is not None and isinstance(d[k], (int, float))
            ]
            if not vals:
                agg[k] = {"mean": None, "std": None, "min": None, "max": None, "n": 0}
                continue
            agg[k] = {
                "mean": float(statistics.fmean(vals)),
                "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                "min": float(min(vals)),
                "max": float(max(vals)),
                "n": len(vals),
            }
        return agg

    test_agg = _aggregate(per_seed_test)
    gold_agg = _aggregate(per_seed_gold)

    aggregate = {
        "n_seeds": n_seeds,
        "representative_seed": REPRESENTATIVE_SEED,
        "training_distribution": dist,
        "empirical_test_distribution_over_all_seeds": {
            "p_conflict": emp_p_conflict,
            "type_probs": emp_type_probs,
        },
        "test": {
            "cross_seed_aggregate": test_agg,
            "representative_seed_metrics": _extract_key_metrics(representative_test) if representative_test else None,
            "representative_seed_bootstrap_ci": (
                representative_test.get("binary", {}).get("bootstrap") if representative_test else None
            ),
        },
        "gold_iaa": {
            "cross_seed_aggregate": gold_agg,
            "representative_seed_metrics": _extract_key_metrics(representative_gold) if representative_gold else None,
            "representative_seed_bootstrap_ci": (
                representative_gold.get("binary", {}).get("bootstrap") if representative_gold else None
            ),
        },
    }
    (base / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))
    print(f"  aggregate written: {base / 'aggregate_metrics.json'}")
    return aggregate


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------
def print_comparison(maj: dict, strat: dict) -> None:
    rows = [
        ("Binary F1 (conflict)", "binary_f1"),
        ("Binary precision", "binary_precision"),
        ("Binary recall", "binary_recall"),
        ("Macro-F1", "macro_f1"),
        ("Accuracy", "accuracy"),
        ("ROC-AUC", "roc_auc"),
        ("Typology macro-F1", "typology_macro_f1"),
        ("Joint exact-match", "joint_em"),
    ]

    def fmt_val(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    def fmt_ms(agg, key):
        a = agg.get(key, {})
        m = a.get("mean")
        s = a.get("std")
        if m is None:
            return "—"
        return f"{m:.4f} ± {s:.4f}" if s is not None else f"{m:.4f}"

    print()
    print("=" * 100)
    print(f"{'BASELINE COMPARISON — TEST SPLIT':^100}")
    print("=" * 100)
    print(f"{'Metric':<28} {'Majority class':<22} {'Stratified random (mean ± std, 100 seeds)':<46}")
    print("-" * 100)
    maj_test = _extract_key_metrics(maj["test_metrics"])
    strat_agg = strat["test"]["cross_seed_aggregate"]
    for label, key in rows:
        print(f"{label:<28} {fmt_val(maj_test[key]):<22} {fmt_ms(strat_agg, key):<46}")
    print()
    print(f"Representative-seed bootstrap CIs (stratified_random, seed={REPRESENTATIVE_SEED}):")
    rep = strat["test"]["representative_seed_bootstrap_ci"] or {}
    if rep.get("f1_conflict"):
        ci = rep["f1_conflict"]
        print(f"  binary F1: {ci['mean']:.4f}  bootstrap CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]   "
              f"(cross-seed std: {strat_agg['binary_f1']['std']:.4f})")
    if rep.get("macro_f1"):
        ci = rep["macro_f1"]
        print(f"  macro-F1:  {ci['mean']:.4f}  bootstrap CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]   "
              f"(cross-seed std: {strat_agg['macro_f1']['std']:.4f})")

    print()
    print("=" * 100)
    print(f"{'BASELINE COMPARISON — GOLD_IAA SPLIT':^100}")
    print("=" * 100)
    print(f"{'Metric':<28} {'Majority class':<22} {'Stratified random (mean ± std, 100 seeds)':<46}")
    print("-" * 100)
    maj_gold = _extract_key_metrics(maj["gold_iaa_metrics"])
    strat_gold_agg = strat["gold_iaa"]["cross_seed_aggregate"]
    for label, key in rows:
        print(f"{label:<28} {fmt_val(maj_gold[key]):<22} {fmt_ms(strat_gold_agg, key):<46}")

    print()
    print("=" * 100)
    print("Empirical proportions across all 100 seeds (test split):")
    emp = strat["empirical_test_distribution_over_all_seeds"]
    train_dist = strat["training_distribution"]
    print(f"  P(conflict):  empirical={emp['p_conflict']:.4f}   training={train_dist['p_conflict']:.4f}   "
          f"abs diff={abs(emp['p_conflict']-train_dist['p_conflict']):.4f}")
    print(f"  P(type | conflict):")
    for t in stratified.CONFLICT_TYPES:
        e = emp["type_probs"][t]
        tr = train_dist["type_probs"][t]
        print(f"    {t:<35} empirical={e:.4f}   training={tr:.4f}   abs diff={abs(e-tr):.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--n-bootstrap-per-seed", type=int, default=100,
                   help="Bootstrap iterations per stratified-random seed (default 100 — cheap)")
    p.add_argument("--n-bootstrap-representative", type=int, default=1000,
                   help=f"Bootstrap iterations for the representative seed={REPRESENTATIVE_SEED} run")
    p.add_argument("--n-bootstrap-majority", type=int, default=1000)
    args = p.parse_args()

    train = load_split("train")
    splits = {
        "test": load_split("test"),
        "gold_iaa": load_split("gold_iaa"),
    }
    print(f"Loaded splits: train={len(train)}, test={len(splits['test'])}, gold_iaa={len(splits['gold_iaa'])}")

    maj_results = run_majority(train, splits, args.n_bootstrap_majority)
    strat_results = run_stratified(
        train,
        splits,
        n_seeds=args.n_seeds,
        n_bootstrap_per_seed=args.n_bootstrap_per_seed,
        n_bootstrap_representative=args.n_bootstrap_representative,
    )
    print_comparison(maj_results, strat_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
