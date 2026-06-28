"""Shared pytest fixtures for RegConflict tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def splits_dir(repo_root) -> Path:
    return repo_root / "data" / "conflicts"


@pytest.fixture(scope="session")
def all_records(splits_dir) -> dict[str, list[dict]]:
    """Load every labelled JSONL split into a dict {split_name: records}."""
    out: dict[str, list[dict]] = {}
    for name in ("train", "val", "test", "gold_iaa"):
        path = splits_dir / f"{name}.jsonl"
        if not path.exists():
            out[name] = []
            continue
        out[name] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return out


@pytest.fixture(scope="session")
def inventory(repo_root) -> list[dict]:
    """Load the document inventory CSV as a list of dicts."""
    import csv
    path = repo_root / "data" / "corpus" / "document_inventory.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
