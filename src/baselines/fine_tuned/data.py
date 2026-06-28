"""Data preprocessing for the DeBERTa-v3-base cross-encoder baseline.

Cross-encoder input format:

    [CLS] regime_A_context [SEP] regime_B_context [SEP]

`regime_A_context` and `regime_B_context` are the concatenations of each
regime's evidence passages, in their existing list order (which reflects
the hybrid retriever's ranking from annotation time).

**Truncation policy (JOINT BUDGET).** Both sides together must fit in the
model's 512-token positional window, minus 3 special tokens — so the
joint content budget is 509 tokens. We add chunks alternately from each
side, taking the next chunk from whichever side has consumed less of the
joint budget, until either no more chunks fit or both sides are
exhausted. This is a deliberate deviation from a per-side cap, justified
empirically: per-side capping at 240 truncated 54% of training records,
while joint 509 truncates 0% (and only 1/497 needs mid-chunk truncation).

The four conflict types are mapped to integer ids 0-3 for the typology
head; non-conflict records carry typology_id = -100 (PyTorch ignore_index
sentinel) so the masked cross-entropy loss receives no gradient from them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = REPO_ROOT / "data" / "conflicts"

CONFLICT_TYPES = (
    "structural_unresolved",
    "operationally_resolvable",
    "interpretive_fact_sensitive",
    "recurring_friction",
)
CONFLICT_TYPE_TO_ID = {t: i for i, t in enumerate(CONFLICT_TYPES)}
IGNORE_INDEX = -100

# Reasonable defaults; can be overridden per-config in the CLI.
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_TOTAL_MAX = 512       # DeBERTa-v3-base positional limit
DEFAULT_JOINT_BUDGET = DEFAULT_TOTAL_MAX - 3  # 509 — minus [CLS] + 2× [SEP]


def load_records(split: str) -> list[dict]:
    """Load one of the named splits: train/val/test/gold_iaa."""
    path = SPLITS_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _regime_context(evidence: Iterable[dict]) -> list[str]:
    """Return chunk texts in the existing order — list of raw passage strings.

    The annotation pipeline ordered evidence by hybrid-retriever rank; we keep
    that order so 'truncate from the tail' equals 'drop lowest-ranked chunks'.
    """
    parts = []
    for chunk in evidence or []:
        text = (chunk.get("passage") or "").strip()
        if text:
            parts.append(text)
    return parts


def _joint_truncate(
    a_chunks: list[str],
    b_chunks: list[str],
    tokenizer,
    joint_budget: int,
) -> tuple[list[int], list[int], int, int, bool, bool, bool, bool]:
    """Allocate the joint budget across both regimes' evidence chunks.

    Algorithm:
      * Tokenise each chunk separately.
      * If both first-chunks together exceed the budget, mid-truncate each
        to half the budget (preserving rough symmetry).
      * Otherwise, greedily add chunks from whichever side currently uses
        fewer tokens (round-robin tie-broken toward A) until the next
        candidate chunk doesn't fit.

    Returns:
      (a_ids, b_ids, n_a_used, n_b_used,
       a_truncated, b_truncated, a_mid_truncated, b_mid_truncated)
    """
    a_chunk_ids = [tokenizer.encode(c, add_special_tokens=False) for c in a_chunks]
    b_chunk_ids = [tokenizer.encode(c, add_special_tokens=False) for c in b_chunks]
    a_ids: list[int] = []
    b_ids: list[int] = []
    a_used = b_used = 0
    a_truncated = b_truncated = False
    a_mid = b_mid = False

    # Edge case: even the first chunk on either side alone might exceed the budget.
    # If so, mid-truncate to budget // 2 on the offending side.
    if a_chunk_ids and not a_ids and len(a_chunk_ids[0]) > joint_budget:
        cap = joint_budget // 2
        a_ids = a_chunk_ids[0][:cap]
        a_used = 1
        a_mid = True
        a_truncated = len(a_chunk_ids) > 1
    if b_chunk_ids and not b_ids and len(b_chunk_ids[0]) > joint_budget:
        cap = joint_budget // 2
        b_ids = b_chunk_ids[0][:cap]
        b_used = 1
        b_mid = True
        b_truncated = len(b_chunk_ids) > 1

    # Round-robin greedy adds
    while True:
        # Pick the side with fewer tokens (ties → A)
        next_a = a_chunk_ids[a_used] if a_used < len(a_chunk_ids) and not a_mid else None
        next_b = b_chunk_ids[b_used] if b_used < len(b_chunk_ids) and not b_mid else None
        if next_a is None and next_b is None:
            break
        used_so_far = len(a_ids) + len(b_ids)
        remaining = joint_budget - used_so_far
        if remaining <= 0:
            if next_a is not None or next_b is not None:
                if next_a is not None and a_used < len(a_chunk_ids):
                    a_truncated = True
                if next_b is not None and b_used < len(b_chunk_ids):
                    b_truncated = True
            break

        # Decide which side to add to first
        if next_a is not None and (next_b is None or len(a_ids) <= len(b_ids)):
            if len(next_a) <= remaining:
                a_ids.extend(next_a)
                a_used += 1
            else:
                a_truncated = True
                # Try B if there's still room
                if next_b is not None and len(next_b) <= remaining:
                    b_ids.extend(next_b)
                    b_used += 1
                else:
                    if next_b is not None:
                        b_truncated = True
                    break
        elif next_b is not None:
            if len(next_b) <= remaining:
                b_ids.extend(next_b)
                b_used += 1
            else:
                b_truncated = True
                if next_a is not None and len(next_a) <= remaining:
                    a_ids.extend(next_a)
                    a_used += 1
                else:
                    if next_a is not None:
                        a_truncated = True
                    break
        else:
            break

    return a_ids, b_ids, a_used, b_used, a_truncated, b_truncated, a_mid, b_mid


@dataclass
class Example:
    pair_id: str
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int] | None
    binary_label: int          # 0 = non-conflict, 1 = conflict
    typology_label: int        # 0-3 if conflict, -100 (ignore_index) otherwise
    record_type: str           # "conflict" | "non_conflict"
    conflict_type: str | None  # gold conflict_type or None
    n_chunks_a_used: int
    n_chunks_a_total: int
    n_chunks_b_used: int
    n_chunks_b_total: int
    a_truncated: bool
    b_truncated: bool
    a_mid_chunk_truncated: bool
    b_mid_chunk_truncated: bool


def build_example(record: dict, tokenizer, *,
                  joint_budget: int = DEFAULT_JOINT_BUDGET,
                  total_max: int = DEFAULT_TOTAL_MAX) -> Example:
    """Build one Example from a split record using the joint-budget policy."""
    a_chunks = _regime_context(record.get("evidence_a") or [])
    b_chunks = _regime_context(record.get("evidence_b") or [])
    a_ids, b_ids, a_used, b_used, a_trunc, b_trunc, a_mid, b_mid = _joint_truncate(
        a_chunks, b_chunks, tokenizer, joint_budget
    )

    # Build [CLS] A [SEP] B [SEP] manually so we control budget exactly
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    input_ids = [cls] + a_ids + [sep] + b_ids + [sep]
    # Defensive clamp (should never trigger if _joint_truncate respected the budget)
    if len(input_ids) > total_max:
        input_ids = input_ids[:total_max - 1] + [sep]
    attention_mask = [1] * len(input_ids)
    # Side A = 0, side B = 1
    sep_pos = 1 + len(a_ids)
    token_type_ids = [0] * (sep_pos + 1) + [1] * (len(input_ids) - sep_pos - 1)

    is_conflict = record.get("record_type") == "conflict"
    binary_label = 1 if is_conflict else 0
    if is_conflict:
        ct = record.get("conflict_type")
        if ct not in CONFLICT_TYPE_TO_ID:
            raise ValueError(f"Unknown conflict_type {ct!r} on pair {record.get('pair_id')}")
        typology_label = CONFLICT_TYPE_TO_ID[ct]
    else:
        typology_label = IGNORE_INDEX  # masked from loss

    return Example(
        pair_id=record["pair_id"],
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        binary_label=binary_label,
        typology_label=typology_label,
        record_type=record.get("record_type", "?"),
        conflict_type=record.get("conflict_type") if is_conflict else None,
        n_chunks_a_used=a_used, n_chunks_a_total=len(a_chunks),
        n_chunks_b_used=b_used, n_chunks_b_total=len(b_chunks),
        a_truncated=a_trunc, b_truncated=b_trunc,
        a_mid_chunk_truncated=a_mid, b_mid_chunk_truncated=b_mid,
    )


class CrossEncoderDataset(Dataset):
    """PyTorch Dataset that yields per-example dicts ready for DataLoader collation."""

    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "pair_id": ex.pair_id,
            "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(ex.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(ex.token_type_ids, dtype=torch.long),
            "binary_label": torch.tensor(ex.binary_label, dtype=torch.long),
            "typology_label": torch.tensor(ex.typology_label, dtype=torch.long),
        }


def collate(batch: list[dict], pad_token_id: int) -> dict:
    """Right-pad variable-length sequences to the longest in the batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    out = {
        "pair_id": [b["pair_id"] for b in batch],
        "input_ids": torch.full((len(batch), max_len), pad_token_id, dtype=torch.long),
        "attention_mask": torch.zeros((len(batch), max_len), dtype=torch.long),
        "token_type_ids": torch.zeros((len(batch), max_len), dtype=torch.long),
        "binary_label": torch.stack([b["binary_label"] for b in batch]),
        "typology_label": torch.stack([b["typology_label"] for b in batch]),
    }
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        out["input_ids"][i, :n] = b["input_ids"]
        out["attention_mask"][i, :n] = b["attention_mask"]
        out["token_type_ids"][i, :n] = b["token_type_ids"]
    return out


def truncation_stats(examples: list[Example]) -> dict:
    """Aggregate stats for a list of Examples — used for the pre-training report."""
    n = len(examples)
    if n == 0:
        return {}
    a_trunc = sum(1 for e in examples if e.a_truncated)
    b_trunc = sum(1 for e in examples if e.b_truncated)
    any_trunc = sum(1 for e in examples if e.a_truncated or e.b_truncated)
    a_mid = sum(1 for e in examples if e.a_mid_chunk_truncated)
    b_mid = sum(1 for e in examples if e.b_mid_chunk_truncated)
    any_mid = sum(1 for e in examples if e.a_mid_chunk_truncated or e.b_mid_chunk_truncated)
    lengths = [len(e.input_ids) for e in examples]
    return {
        "n_examples": n,
        "n_truncated_a_only": a_trunc,
        "n_truncated_b_only": b_trunc,
        "n_truncated_any_side": any_trunc,
        "pct_truncated_any_side": any_trunc / n,
        "n_mid_chunk_truncated_any_side": any_mid,
        "pct_mid_chunk_truncated_any_side": any_mid / n,
        "n_mid_chunk_truncated_a": a_mid,
        "n_mid_chunk_truncated_b": b_mid,
        "input_ids_length": {
            "min": min(lengths),
            "mean": sum(lengths) / n,
            "median": sorted(lengths)[n // 2],
            "max": max(lengths),
            "p90": sorted(lengths)[int(0.90 * n)],
        },
        "binary_class_distribution": {
            "non_conflict": sum(1 for e in examples if e.binary_label == 0),
            "conflict": sum(1 for e in examples if e.binary_label == 1),
        },
        "typology_class_distribution": {
            t: sum(1 for e in examples if e.conflict_type == t)
            for t in CONFLICT_TYPES
        },
    }


def prepare(split: str, tokenizer, **kwargs) -> tuple[CrossEncoderDataset, dict]:
    """Convenience: load a split, build examples, return dataset + stats."""
    records = load_records(split)
    examples = [build_example(r, tokenizer, **kwargs) for r in records]
    return CrossEncoderDataset(examples), truncation_stats(examples)
