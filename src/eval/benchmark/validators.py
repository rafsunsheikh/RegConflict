"""Sanity checks on predictions before metrics are computed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONFIG_HASH_INDEX = Path(__file__).resolve().parents[3] / "eval" / "config_hash_index.json"


def validate_predictions(
    split_records: list[dict], predictions: list[dict]
) -> dict:
    """Validate the alignment between split ground truth and a predictions list.

    Returns a dict with:
      - missing_pair_ids: split pair_ids not present in predictions
      - extra_pair_ids:   prediction pair_ids not present in the split
      - type_set_without_conflict: predictions where type is set but conflict_present=False
      - aligned: number of pairs successfully aligned

    The function logs warnings but does not raise unless alignment is impossible.
    """
    split_pids = {r["pair_id"] for r in split_records}
    pred_pids = {p["pair_id"] for p in predictions}
    missing = sorted(split_pids - pred_pids)
    extra = sorted(pred_pids - split_pids)

    type_set_without_conflict = []
    for p in predictions:
        if (
            not p.get("predicted_conflict_present")
            and p.get("predicted_conflict_type") is not None
        ):
            type_set_without_conflict.append(p["pair_id"])

    return {
        "n_split": len(split_pids),
        "n_predictions": len(pred_pids),
        "aligned": len(split_pids & pred_pids),
        "missing_pair_ids": missing,
        "extra_pair_ids": extra,
        "type_set_without_conflict": type_set_without_conflict,
    }


def check_config_hash_consistency(
    config_hash: str, split: str, metrics_summary: dict
) -> dict:
    """Detect non-determinism: same (config_hash, split) producing different metrics.

    Stores a small index mapping (config_hash, split) → metric fingerprint.
    If the same hash + split is seen with a different fingerprint, the prior
    run is surfaced. The split is part of the key because the same model
    evaluated on a different split legitimately produces different metrics —
    that is not non-determinism.
    """
    fingerprint = hashlib.sha256(
        json.dumps(metrics_summary, sort_keys=True).encode()
    ).hexdigest()[:16]

    if CONFIG_HASH_INDEX.exists():
        index = json.loads(CONFIG_HASH_INDEX.read_text())
    else:
        CONFIG_HASH_INDEX.parent.mkdir(parents=True, exist_ok=True)
        index = {}

    key = f"{config_hash}|{split}"
    prior = index.get(key)
    warning = None
    if prior and prior["fingerprint"] != fingerprint:
        warning = (
            f"config_hash {config_hash} on split {split} has been seen before "
            f"with a different metric fingerprint ({prior['fingerprint']} vs "
            f"current {fingerprint}). This suggests non-determinism."
        )

    index[key] = {"fingerprint": fingerprint, "metrics_summary": metrics_summary}
    CONFIG_HASH_INDEX.write_text(json.dumps(index, indent=2))
    return {"warning": warning, "fingerprint": fingerprint}
