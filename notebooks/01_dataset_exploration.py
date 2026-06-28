# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 01 — Dataset exploration
#
# An interactive tour of the RegConflict v1.0 benchmark. After running this notebook end-to-end you'll have:
#
# - the per-split record counts and class distributions
# - the per-jurisdiction pair breakdown
# - a few sample records of each kind (conflict / non-conflict, each typology class)
# - a feel for what the model sees at evaluation time
#
# Run as a plain script (`python notebooks/01_dataset_exploration.py`) or convert to a Jupyter notebook with `jupytext --to ipynb notebooks/01_dataset_exploration.py`.

# %%
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# %% [markdown]
# ## Load the splits

# %%
def load_split(name):
    path = REPO_ROOT / "data" / "conflicts" / f"{name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


splits = {name: load_split(name) for name in ("train", "val", "test", "gold_iaa")}
for name, recs in splits.items():
    n_conflict = sum(1 for r in recs if r["record_type"] == "conflict")
    print(f"{name:>10}: {len(recs):>4} records ({n_conflict} conflicts, {len(recs)-n_conflict} non-conflicts)")

# %% [markdown]
# ## Per-conflict-type counts

# %%
print(f"{'class':>32}  {'train':>6} {'val':>6} {'test':>6} {'iaa':>6}")
for cls in ("structural_unresolved", "operationally_resolvable",
            "interpretive_fact_sensitive", "recurring_friction"):
    counts = []
    for name in ("train", "val", "test", "gold_iaa"):
        counts.append(sum(1 for r in splits[name]
                          if r.get("conflict_type") == cls))
    print(f"{cls:>32}  {counts[0]:>6} {counts[1]:>6} {counts[2]:>6} {counts[3]:>6}")

# %% [markdown]
# ## Jurisdictional distribution (test split)

# %%
juris_pairs = Counter()
for r in splits["test"]:
    pair = tuple(sorted([r["regime_a"]["jurisdiction"], r["regime_b"]["jurisdiction"]]))
    juris_pairs[pair] += 1
print("Jurisdiction pairs in test split:")
for pair, n in juris_pairs.most_common():
    print(f"  {pair[0]:>15} ↔ {pair[1]:<15}  {n}")

# %% [markdown]
# ## Sample records — one of each typology class

# %%
for cls in ("structural_unresolved", "operationally_resolvable",
            "interpretive_fact_sensitive", "recurring_friction"):
    samples = [r for r in splits["train"] if r.get("conflict_type") == cls]
    if not samples:
        continue
    r = samples[0]
    print(f"\n=== {cls} (severity={r.get('severity')}) ===")
    print(f"  regime_a: {r['regime_a']['regime_id']} ({r['regime_a']['jurisdiction']})")
    print(f"  regime_b: {r['regime_b']['regime_id']} ({r['regime_b']['jurisdiction']})")
    rationale = r.get("rationale", "")
    print(f"  rationale: {rationale[:250]}{'...' if len(rationale) > 250 else ''}")

# %% [markdown]
# ## What the model sees at evaluation time

# %%
sample = next(r for r in splits["test"] if r["record_type"] == "conflict")
print(f"pair_id: {sample['pair_id']}")
print(f"label: {sample['label']} / {sample.get('conflict_type')} / severity={sample.get('severity')}")
print()
print(f"Regime A — {sample['regime_a']['regime_id']} ({sample['regime_a']['jurisdiction']}):")
for chunk in sample["evidence_a"]:
    page = f", p.{chunk['page']}" if chunk.get("page") else ""
    print(f"  [Chunk{page}] {chunk['passage'][:180]}{'...' if len(chunk['passage']) > 180 else ''}")
print()
print(f"Regime B — {sample['regime_b']['regime_id']} ({sample['regime_b']['jurisdiction']}):")
for chunk in sample["evidence_b"]:
    page = f", p.{chunk['page']}" if chunk.get("page") else ""
    print(f"  [Chunk{page}] {chunk['passage'][:180]}{'...' if len(chunk['passage']) > 180 else ''}")

# %% [markdown]
# ## Schema-validated access (alternative to dict access)

# %%
from src.eval.benchmark.schema import LabelledPair

parsed = LabelledPair.model_validate(sample)
print(f"Parsed via Pydantic: pair_id={parsed.pair_id}")
print(f"  conflict_type = {parsed.conflict_type.value if parsed.conflict_type else None}")
print(f"  severity      = {parsed.severity.value if parsed.severity else None}")
print(f"  evidence_a    = {len(parsed.evidence_a)} chunks")
print(f"  evidence_b    = {len(parsed.evidence_b)} chunks")
