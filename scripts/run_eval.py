"""Run the TRIAG conflict-detection benchmark evaluation on a predictions file.

Usage:
    python scripts/run_eval.py \\
        --predictions path/to/preds.jsonl \\
        --split test \\
        --output-dir eval/results

Each prediction is JSONL with:
    {
      "pair_id": "...",
      "predicted_conflict_present": true/false,
      "predicted_conflict_type": "structural_unresolved" | ... | null,
      "confidence_score": 0.0-1.0 | null,
      "model_metadata": {"model_name": "...", "model_version": "...", "config_hash": "..."}
    }

Outputs land in eval/results/<model_name>/<split>_<timestamp>/.
See src/eval/benchmark/runner.py for the orchestration details.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark.runner import evaluate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSONL")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "gold_iaa"])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Base directory for per-run output folders (default: eval/results)",
    )
    p.add_argument("--model-name", default=None, help="Override model_name (otherwise inferred from predictions)")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    evaluate(
        predictions_path=args.predictions,
        split=args.split,
        output_base=args.output_dir,
        model_name=args.model_name,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
