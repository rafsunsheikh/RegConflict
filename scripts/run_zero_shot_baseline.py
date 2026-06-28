"""Run the zero-shot LLM baseline end-to-end.

Modes:
  --smoke         6 examples, DeepSeek + Variant A, test split only.
                  Use this first before any full run.
  (default)       2 models × 3 variants × 2 splits = 12 evaluations.

Outputs land under eval/results/zero_shot/<model_key>/<variant>/ and the
combined aggregate at eval/results/zero_shot/aggregate_metrics.json. The
runner also appends LLM baseline rows to paper_outputs/results_table.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.zero_shot.client import ENDPOINTS, LLMClient  # noqa: E402
from src.baselines.zero_shot.runner import predict_split  # noqa: E402
from src.eval.benchmark.io import load_split  # noqa: E402
from src.eval.benchmark.runner import evaluate  # noqa: E402

RESULTS_BASE = REPO_ROOT / "results" / "zero_shot"
PAPER_TABLE = REPO_ROOT / "results" / "results_table.md"

MODEL_KEYS = ("glm-4.7-flash", "deepseek-r1-distill-qwen-1.5b")
VARIANTS = ("A", "B", "C")

SMOKE_PAIR_IDS = [
    "conflict:9eae9de6e8294f77cc3cdeb2",      # structural_unresolved
    "conflict:2f473162367fe72df44179b9",      # operationally_resolvable
    "conflict:cfd44392c1c385017cc067b4",      # interpretive_fact_sensitive
    "conflict:9bada48c6c2469fedcdf80a5",      # recurring_friction
    "non_conflict:381f70f94f29e4b1897f5149",  # different_domains
    "non_conflict:2a16226a46a5b801aa37b235",  # superficially_similar_but_disjoint
]


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _extract_key_metrics(metrics: dict) -> dict:
    bp = metrics["binary"]["point"]
    tp = metrics["typology"]["point"]
    jp = metrics["joint"]["point"]
    return {
        "binary_f1": bp["f1_conflict"],
        "binary_precision": bp["precision_conflict"],
        "binary_recall": bp["recall_conflict"],
        "macro_f1": bp["macro_f1"],
        "accuracy": bp["accuracy"],
        "typology_macro_f1": tp.get("macro_f1"),
        "typology_accuracy": tp.get("accuracy"),
        "joint_em": jp["exact_match_accuracy"],
    }


def _parsing_failure_rate(predictions: list[dict]) -> float:
    if not predictions:
        return 0.0
    n_fail = sum(1 for p in predictions if p.get("parsing_status") == "parsing_failure")
    return n_fail / len(predictions)


def run_one(
    model_key: str,
    variant: str,
    records: list[dict],
    split: str,
    n_bootstrap: int,
    skip_harness: bool = False,
    concurrency: int = 1,
    max_tokens: int = 2048,
    variant_suffix: str = "",
) -> dict:
    """Run one (model, variant) on one split. Returns {predictions, metrics, summary}.

    variant_suffix is appended to the variant dir name, so e.g. variant_suffix="_v2"
    writes to variant_B_v2/ instead of variant_B/. The harness's `model_name`
    also gets the suffix so its <split>_<ts>/ output dir nests correctly.
    """
    variant_dir = RESULTS_BASE / model_key / f"variant_{variant}{variant_suffix}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {model_key} × variant_{variant}{variant_suffix} × {split} "
          f"({len(records)} records, concurrency={concurrency}, max_tokens={max_tokens}) ===", flush=True)
    client = LLMClient(model_key)
    try:
        predictions = predict_split(client, variant, records, variant_dir,
                                    concurrency=concurrency, max_tokens=max_tokens)
    finally:
        client.close()

    preds_path = variant_dir / f"{split}_predictions.jsonl"
    _write_jsonl(predictions, preds_path)
    pf_rate = _parsing_failure_rate(predictions)
    print(f"  saved: {preds_path}  parsing-failure rate: {pf_rate:.1%}")

    summary = {
        "model_key": model_key,
        "variant": variant,
        "split": split,
        "n_records": len(predictions),
        "parsing_failure_rate": pf_rate,
        "n_parsing_failures": sum(1 for p in predictions if p["parsing_status"] == "parsing_failure"),
        "retries_distribution": {
            "0": sum(1 for p in predictions if p.get("n_retries_used") == 0),
            "1": sum(1 for p in predictions if p.get("n_retries_used") == 1),
            "2": sum(1 for p in predictions if p.get("n_retries_used") == 2),
        },
        "succeeded_at_temperature_distribution": {
            "0.0": sum(1 for p in predictions if p.get("succeeded_at_temperature") == 0.0),
            "0.3": sum(1 for p in predictions if p.get("succeeded_at_temperature") == 0.3),
            "0.7": sum(1 for p in predictions if p.get("succeeded_at_temperature") == 0.7),
            "none": sum(1 for p in predictions if p.get("succeeded_at_temperature") is None),
        },
    }

    metrics = None
    if not skip_harness:
        result = evaluate(
            predictions_path=preds_path,
            split=split,
            output_base=RESULTS_BASE / model_key,  # harness will create variant_<X>{suffix}/<split>_<ts>/
            model_name=f"variant_{variant}{variant_suffix}",
            n_bootstrap=n_bootstrap,
        )
        metrics = result["metrics"]
        summary.update(_extract_key_metrics(metrics))

    return {"predictions": predictions, "metrics": metrics, "summary": summary}


def aggregate_per_model(model_results: list[dict]) -> dict:
    """Cross-variant aggregate for one model on one split. Mean/std + best-of-three."""
    keys = [
        "binary_f1", "binary_precision", "binary_recall", "macro_f1",
        "accuracy", "typology_macro_f1", "joint_em", "parsing_failure_rate",
    ]
    agg = {}
    best_by_key = {}
    for k in keys:
        vals = [r["summary"].get(k) for r in model_results if r["summary"].get(k) is not None]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            agg[k] = {"mean": None, "std": None, "min": None, "max": None, "n": 0}
            best_by_key[k] = None
            continue
        agg[k] = {
            "mean": float(statistics.fmean(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
            "n": len(vals),
        }
        best_by_key[k] = float(max(vals))
    # Identify best-of-three variant by binary_f1
    if model_results:
        ranked = sorted(model_results, key=lambda r: r["summary"].get("binary_f1") or -1, reverse=True)
        best_variant = ranked[0]["summary"]["variant"]
    else:
        best_variant = None
    return {"cross_variant_aggregate": agg, "best_by_key": best_by_key, "best_variant": best_variant}


def append_paper_table(
    aggregate_per_model_split: dict[tuple[str, str], dict],
    per_variant_results: dict[tuple[str, str, str], dict],
) -> None:
    """Append LLM baseline rows to paper_outputs/results_table.md."""
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("")
    lines.append("## Zero-shot LLM baselines")
    lines.append("")
    lines.append(
        "Columns: Binary F1 / Macro-F1 / Typology macro-F1 / Parsing failures / Notes. "
        "Per-model rows report mean ± std across the three prompt variants. "
        "Per-variant rows are point estimates (temperature=0; retry escalation only on parse failure)."
    )
    lines.append("")
    lines.append("| Model | Variant | Binary F1 | Macro-F1 | Typology Macro-F1 | Parsing Failures | Notes |")
    lines.append("|---|---|---|---|---|---|---|")

    for split in sorted({s for (_, _, s) in per_variant_results.keys()}):
        # Section header per split
        lines.append(f"| **{split} split** | | | | | | |")
        for model_key in MODEL_KEYS:
            # Per-variant rows
            for variant in VARIANTS:
                r = per_variant_results.get((model_key, variant, split))
                if not r:
                    continue
                s = r["summary"]
                lines.append(
                    f"| {model_key} | {variant} | "
                    f"{(s.get('binary_f1') or 0):.3f} | "
                    f"{(s.get('macro_f1') or 0):.3f} | "
                    f"{(s.get('typology_macro_f1') or 0):.3f} | "
                    f"{s.get('parsing_failure_rate', 0):.1%} ({s.get('n_parsing_failures', 0)}) | "
                    f"|"
                )
            # Aggregate row per model
            agg = aggregate_per_model_split.get((model_key, split))
            if agg:
                a = agg["cross_variant_aggregate"]
                best_v = agg["best_variant"]
                lines.append(
                    f"| **{model_key}** | mean ± std (n=3) | "
                    f"{a['binary_f1']['mean']:.3f} ± {a['binary_f1']['std']:.3f} | "
                    f"{a['macro_f1']['mean']:.3f} ± {a['macro_f1']['std']:.3f} | "
                    f"{a['typology_macro_f1']['mean']:.3f} ± {a['typology_macro_f1']['std']:.3f} | "
                    f"{a['parsing_failure_rate']['mean']:.1%} ± {a['parsing_failure_rate']['std']:.1%} | "
                    f"best variant: {best_v} |"
                )
    lines.append("")

    existing = PAPER_TABLE.read_text() if PAPER_TABLE.exists() else ""
    # Replace any prior "Zero-shot LLM baselines" section
    marker = "## Zero-shot LLM baselines"
    if marker in existing:
        head = existing.split(marker, 1)[0].rstrip() + "\n"
        existing = head
    PAPER_TABLE.write_text(existing.rstrip() + "\n" + "\n".join(lines))


def run_smoke(args) -> int:
    """Tiny end-to-end check: DeepSeek + Variant A + 6 specific pairs."""
    print("=== SMOKE TEST: DeepSeek + Variant A + 6 examples ===")
    all_test = load_split("test")
    by_id = {r["pair_id"]: r for r in all_test}
    missing = [pid for pid in SMOKE_PAIR_IDS if pid not in by_id]
    if missing:
        print(f"ERROR: smoke pair_ids missing from test split: {missing}", file=sys.stderr)
        return 1
    records = [by_id[pid] for pid in SMOKE_PAIR_IDS]

    r = run_one(
        model_key="deepseek-r1-distill-qwen-1.5b",
        variant="A",
        records=records,
        split="test",
        n_bootstrap=0,
        skip_harness=True,  # 6 records is too small for meaningful metrics
    )
    s = r["summary"]
    print()
    print("=== Smoke result ===")
    print(f"  records run:      {s['n_records']}")
    print(f"  parsing failures: {s['n_parsing_failures']} ({s['parsing_failure_rate']:.1%})")
    print(f"  retries used (0/1/2): "
          f"{s['retries_distribution']['0']}/{s['retries_distribution']['1']}/{s['retries_distribution']['2']}")
    print(f"  succeeded@temp:   0.0={s['succeeded_at_temperature_distribution']['0.0']}, "
          f"0.3={s['succeeded_at_temperature_distribution']['0.3']}, "
          f"0.7={s['succeeded_at_temperature_distribution']['0.7']}, "
          f"none={s['succeeded_at_temperature_distribution']['none']}")
    print()
    print("Per-record predictions (truth | predicted | rationale snippet):")
    for pred, rec in zip(r["predictions"], records):
        true_cp = rec.get("record_type") == "conflict"
        true_ct = rec.get("conflict_type") if true_cp else rec.get("non_conflict_subtype")
        pred_cp = pred["predicted_conflict_present"]
        pred_ct = pred["predicted_conflict_type"]
        mark = "✓" if (true_cp == pred_cp and (not true_cp or rec.get("conflict_type") == pred_ct)) else "✗"
        ra = (rec.get("regime_a") or {}).get("regime_id", "?")
        rb = (rec.get("regime_b") or {}).get("regime_id", "?")
        rationale = (pred.get("rationale_from_llm") or "")[:120]
        print(f"  {mark} {ra} × {rb}")
        print(f"      truth: conflict={true_cp}  type={true_ct}")
        print(f"      pred:  conflict={pred_cp}  type={pred_ct}  status={pred['parsing_status']}")
        if rationale:
            print(f"      rationale: {rationale}")
    return 0


def run_full(args) -> int:
    """Full evaluation: filterable over models, variants, splits."""
    splits_to_run = {s: load_split(s) for s in args.splits}
    models_to_run = args.models or list(MODEL_KEYS)
    variants_to_run = args.variants or list(VARIANTS)
    print(f"Running models={models_to_run} variants={variants_to_run} splits={list(splits_to_run.keys())} "
          f"max_tokens={args.max_tokens} variant_suffix='{args.variant_suffix}'", flush=True)

    per_variant_results: dict[tuple[str, str, str], dict] = {}
    for model_key in models_to_run:
        for variant in variants_to_run:
            for split, records in splits_to_run.items():
                key = (model_key, variant, split)
                r = run_one(model_key, variant, records, split, args.n_bootstrap,
                            concurrency=args.concurrency,
                            max_tokens=args.max_tokens,
                            variant_suffix=args.variant_suffix)
                per_variant_results[key] = r

    # If this is a partial run (filtered models/variants/splits, or a
    # variant_suffix in use), skip the full-grid aggregation + paper-table
    # append — those assume all 12 cells. The delta-report script
    # (scripts/build_zero_shot_v2_delta.py) handles partial runs separately.
    is_partial = bool(
        args.models or args.variants or len(args.splits) < 2 or args.variant_suffix
    )
    if not is_partial:
        # Aggregates per (model, split)
        aggregate_per_model_split: dict[tuple[str, str], dict] = {}
        for model_key in MODEL_KEYS:
            for split in splits_to_run:
                mr = [per_variant_results[(model_key, v, split)] for v in VARIANTS
                      if (model_key, v, split) in per_variant_results]
                aggregate_per_model_split[(model_key, split)] = aggregate_per_model(mr)

        overall_agg = {
            "models": {
                mk: {split: aggregate_per_model_split[(mk, split)] for split in splits_to_run}
                for mk in MODEL_KEYS
            },
            "per_variant_summaries": {
                f"{mk}|{v}|{s}": per_variant_results[(mk, v, s)]["summary"]
                for (mk, v, s) in per_variant_results
            },
        }
        RESULTS_BASE.mkdir(parents=True, exist_ok=True)
        (RESULTS_BASE / "aggregate_metrics.json").write_text(json.dumps(overall_agg, indent=2))
        print(f"\nCombined aggregate written: {RESULTS_BASE / 'aggregate_metrics.json'}")
        append_paper_table(aggregate_per_model_split, per_variant_results)
        print(f"Paper table updated: {PAPER_TABLE}")
    else:
        print(f"\n(Partial run — skipped global aggregate + paper-table append. "
              f"Run the v2 delta report script after this completes.)")

    # Print summary comparison (only the rows that actually ran)
    print()
    print("=" * 100)
    print("ZERO-SHOT BASELINE SUMMARY (test split, cells that ran)")
    print("=" * 100)
    print(f"{'Model':<35} {'Variant':<10} {'Binary F1':<10} {'Macro-F1':<10} "
          f"{'Typology F1':<12} {'Parse fail':<10}")
    print("-" * 100)
    for mk in models_to_run:
        for v in variants_to_run:
            r = per_variant_results.get((mk, v, "test"))
            if not r: continue
            s = r["summary"]
            print(f"{mk:<35} {v:<10} "
                  f"{(s.get('binary_f1') or 0):<10.3f} "
                  f"{(s.get('macro_f1') or 0):<10.3f} "
                  f"{(s.get('typology_macro_f1') or 0):<12.3f} "
                  f"{s.get('parsing_failure_rate', 0):<10.1%}")
        if not is_partial:
            agg = aggregate_per_model_split[(mk, "test")]["cross_variant_aggregate"]
            print(f"{mk:<35} {'MEAN±STD':<10} "
                  f"{agg['binary_f1']['mean']:.3f}±{agg['binary_f1']['std']:.3f}  "
                  f"{agg['macro_f1']['mean']:.3f}±{agg['macro_f1']['std']:.3f}  "
                  f"{agg['typology_macro_f1']['mean']:.3f}±{agg['typology_macro_f1']['std']:.3f}  "
                  f"{agg['parsing_failure_rate']['mean']:.1%}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Run 6-example smoke test only")
    p.add_argument("--n-bootstrap", type=int, default=1000,
                   help="Bootstrap iterations for the harness (default 1000)")
    p.add_argument("--concurrency", type=int, default=2,
                   help="Concurrent in-flight requests per server (default 2; llama-server has 4 slots)")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="max_tokens per LLM call (default 2048; bump to 8192 for GLM reasoning headroom)")
    p.add_argument("--models", nargs="+", choices=list(MODEL_KEYS), default=None,
                   help="Restrict to a subset of models (default: all)")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None,
                   help="Restrict to a subset of variants (default: A, B, C)")
    p.add_argument("--splits", nargs="+", choices=["test", "gold_iaa"], default=["test", "gold_iaa"],
                   help="Which splits to run (default: both)")
    p.add_argument("--variant-suffix", default="",
                   help="Suffix on variant dir, e.g. '_v2' writes to variant_B_v2/. "
                        "Use this to preserve previous results during a re-run.")
    args = p.parse_args()
    if args.smoke:
        return run_smoke(args)
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
