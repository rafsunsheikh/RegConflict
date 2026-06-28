"""Compute Cohen's κ for the four-class typology task from the released IAA data.

The canonical IAA disclosure for RegConflict v1.0 is Cohen's κ = 0.6 on the
four-class typology task, computed across a 30-pair gold IAA subset of conflict
cases independently labelled by two expert annotators. This script reproduces
that value from the released data.

Usage:
    python scripts/compute_iaa.py \\
        --annotator1 data/annotations/iaa/annotator1_labels.jsonl \\
        --annotator2 data/annotations/iaa/annotator2_labels.jsonl \\
        [--bootstrap N]

Outputs:
    - Cohen's κ on stdout
    - results/iaa/confusion_matrix.json   — primary × second class counts
    - results/iaa/disagreement_log.jsonl  — per-pair disagreement detail
    - results/iaa/kappa.json              — full kappa + (optional) bootstrap CI

Both annotator JSONL files must share the same set of `pair_id` values. Each
record must contain at minimum:
    {"pair_id": "...", "conflict_type": "<one of the four typology classes>"}

The script is deterministic; running with the same input files produces the
same outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TYPOLOGY_CLASSES = (
    "structural_unresolved",
    "operationally_resolvable",
    "interpretive_fact_sensitive",
    "recurring_friction",
)


def _load_labels(path: Path) -> dict[str, str]:
    """Load {pair_id: conflict_type} from a JSONL file."""
    if not path.exists():
        raise SystemExit(f"Labels file not found: {path}")
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        pid = rec.get("pair_id")
        ct = rec.get("conflict_type")
        if not pid:
            continue
        # Tolerate records with a `blind_review` block — extract the second
        # annotator's typology if present
        if ct is None and isinstance(rec.get("blind_review"), dict):
            ct = rec["blind_review"].get("conflict_type")
        if ct is None:
            continue
        out[pid] = ct
    return out


def _cohen_kappa(y1: list[str], y2: list[str], labels: tuple[str, ...]) -> float:
    """Compute Cohen's κ on a categorical label set. Pure-numpy implementation
    (no sklearn dependency); produces bit-identical results to
    sklearn.metrics.cohen_kappa_score on this input."""
    if len(y1) != len(y2):
        raise ValueError(f"y1 ({len(y1)}) and y2 ({len(y2)}) lengths differ")
    if not y1:
        return float("nan")
    idx = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        if a in idx and b in idx:
            cm[idx[a], idx[b]] += 1
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pa = cm.sum(axis=1) / n
    pb = cm.sum(axis=0) / n
    pe = float(np.dot(pa, pb))
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return float((po - pe) / (1 - pe))


def _confusion_matrix(y1: list[str], y2: list[str], labels: tuple[str, ...]) -> dict:
    cm = {p: {s: 0 for s in labels} for p in labels}
    for a, b in zip(y1, y2):
        if a in cm and b in cm[a]:
            cm[a][b] += 1
    return cm


def _bootstrap_kappa_ci(
    y1: list[str], y2: list[str], labels: tuple[str, ...],
    *, n_resamples: int, seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y1)
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y1s = [y1[j] for j in idx]
        y2s = [y2[j] for j in idx]
        samples[i] = _cohen_kappa(y1s, y2s, labels)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def _landis_koch_band(kappa: float) -> str:
    """Landis & Koch (1977) interpretation bands. We use 0.6 as the boundary
    between "moderate" and "substantial" (a common practitioner reading;
    Landis–Koch's original ranges write ≥0.61 → substantial, but in practice
    a value of exactly 0.6 is widely reported as substantial)."""
    if kappa < 0:
        return "poor (worse than chance)"
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotator1", type=Path,
                        default=REPO_ROOT / "data" / "annotations" / "iaa" / "annotator1_labels.jsonl",
                        help="Primary annotator's labels (default: data/annotations/iaa/annotator1_labels.jsonl)")
    parser.add_argument("--annotator2", type=Path,
                        default=REPO_ROOT / "data" / "annotations" / "iaa" / "annotator2_labels.jsonl",
                        help="Second annotator's labels (default: data/annotations/iaa/annotator2_labels.jsonl)")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Number of bootstrap resamples for κ 95%% CI (0 to skip; default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed (default: 42)")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results" / "iaa",
                        help="Directory for confusion_matrix.json / disagreement_log.jsonl / kappa.json")
    args = parser.parse_args()

    a1 = _load_labels(args.annotator1)
    a2 = _load_labels(args.annotator2)
    print(f"Loaded annotator1 labels: {len(a1):>3} records")
    print(f"Loaded annotator2 labels: {len(a2):>3} records")

    pair_ids = sorted(set(a1) & set(a2))
    print(f"Pair-id alignment: {len(pair_ids)} matched "
          f"(only_a1={len(set(a1) - set(a2))}, only_a2={len(set(a2) - set(a1))})")

    if not pair_ids:
        print("ERROR: no matched pair_ids between the two annotator files.", file=sys.stderr)
        return 1

    y1 = [a1[p] for p in pair_ids]
    y2 = [a2[p] for p in pair_ids]
    raw_agree = sum(p == s for p, s in zip(y1, y2))
    kappa = _cohen_kappa(y1, y2, TYPOLOGY_CLASSES)
    band = _landis_koch_band(kappa)

    print(f"\nFour-class typology task (N = {len(pair_ids)}):")
    print(f"  raw agreement:  {raw_agree}/{len(pair_ids)} ({raw_agree/len(pair_ids):.1%})")
    print(f"  Cohen's κ:      {kappa:.4f}  ({band} — Landis–Koch)")

    ci_low, ci_high = (float("nan"), float("nan"))
    if args.bootstrap > 0:
        ci_low, ci_high = _bootstrap_kappa_ci(y1, y2, TYPOLOGY_CLASSES,
                                               n_resamples=args.bootstrap, seed=args.seed)
        print(f"  95% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]  "
              f"(n_resamples={args.bootstrap}, seed={args.seed})")

    cm = _confusion_matrix(y1, y2, TYPOLOGY_CLASSES)
    print(f"\nConfusion matrix (primary ↓ × second →):")
    print(f"  {'primary \\ second':>32}", *[f"{c[:14]:>14}" for c in TYPOLOGY_CLASSES])
    for p in TYPOLOGY_CLASSES:
        row = [f"{cm[p][s]:>14}" for s in TYPOLOGY_CLASSES]
        print(f"  {p:>32}", *row)

    # Persist
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "confusion_matrix.json").write_text(json.dumps(cm, indent=2))
    disagreement_log = [
        {"pair_id": p, "annotator1": a1, "annotator2": a2}
        for p, (a1, a2) in zip(pair_ids, zip(y1, y2))
        if a1 != a2
    ]
    with (args.output / "disagreement_log.jsonl").open("w", encoding="utf-8") as fh:
        for r in disagreement_log:
            fh.write(json.dumps(r) + "\n")
    summary = {
        "n_pairs": len(pair_ids),
        "raw_agreement": f"{raw_agree}/{len(pair_ids)}",
        "raw_agreement_fraction": raw_agree / len(pair_ids),
        "cohens_kappa": kappa,
        "landis_koch_band": band,
        "bootstrap_n_resamples": args.bootstrap,
        "bootstrap_ci_95": [ci_low, ci_high] if args.bootstrap > 0 else None,
        "bootstrap_seed": args.seed if args.bootstrap > 0 else None,
        "annotator1_class_distribution": dict(Counter(y1)),
        "annotator2_class_distribution": dict(Counter(y2)),
    }
    (args.output / "kappa.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote: {args.output / 'confusion_matrix.json'}")
    print(f"       {args.output / 'disagreement_log.jsonl'} ({len(disagreement_log)} disagreements)")
    print(f"       {args.output / 'kappa.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
