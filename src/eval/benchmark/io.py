"""Loading and writing for the evaluation harness."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = ROOT / "data" / "splits"


def load_split(split: str) -> list[dict]:
    """Load one of the named splits: train / val / test / gold_iaa."""
    valid = {"train", "val", "test", "gold_iaa"}
    if split not in valid:
        raise ValueError(f"Unknown split {split!r}; expected one of {valid}")
    path = SPLITS_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file {path} not found. Run scripts/build_splits.py first."
        )
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_predictions(path: Path) -> list[dict]:
    """Load a predictions JSONL file.

    Required fields per record:
      - pair_id (str)
      - predicted_conflict_present (bool)
      - predicted_conflict_type (str or null)
      - confidence_score (float or null)
      - model_metadata: {model_name, model_version, config_hash}
    """
    if not path.exists():
        raise FileNotFoundError(f"Predictions file {path} not found")
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    required = {"pair_id", "predicted_conflict_present"}
    for r in records:
        missing = required - set(r.keys())
        if missing:
            raise ValueError(f"Prediction record {r} missing required fields: {missing}")
    return records


def write_results(
    output_dir: Path,
    metrics: dict,
    confusion_matrix: dict,
    per_jurisdiction: dict,
    errors: list[dict],
    config: dict,
    log_text: str,
) -> None:
    """Write all evaluation artefacts to the model's timestamped output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "confusion_matrix.json").write_text(json.dumps(confusion_matrix, indent=2))
    (output_dir / "per_jurisdiction.json").write_text(json.dumps(per_jurisdiction, indent=2))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "eval_log.txt").write_text(log_text)
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as fh:
        for err in errors:
            fh.write(json.dumps(err, ensure_ascii=False) + "\n")


def make_output_dir(base: Path, model_name: str, split: str) -> Path:
    """Create eval/results/<model_name>/<split>_<ts>/ and return the path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = model_name.replace("/", "_").replace(" ", "_")
    out = base / safe / f"{split}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out
