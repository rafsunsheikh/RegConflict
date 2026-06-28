# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03 — Error analysis
#
# Qualitative exploration of where baseline predictions go wrong. After running this notebook you'll have:
#
# - per-baseline per-class confusion matrices (binary task)
# - the universally-hard test cases (failed by every baseline)
# - a side-by-side view of one case's predictions across all baselines
#
# **Prerequisites:** the baselines must have been run end-to-end (see `02_baseline_reproduction.py`). The cells below load `results/<baseline>/.../predictions.jsonl` files.

# %%
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
TEST_RECORDS_PATH = REPO_ROOT / "data" / "conflicts" / "test.jsonl"

# %% [markdown]
# ## Load test gold + all available baseline predictions

# %%
def load_predictions(predictions_path):
    return {json.loads(l)["pair_id"]: json.loads(l)
            for l in predictions_path.read_text().splitlines() if l.strip()}


gold = {json.loads(l)["pair_id"]: json.loads(l)
        for l in TEST_RECORDS_PATH.read_text().splitlines() if l.strip()}

baselines = {}
for predictions_file in sorted(RESULTS.rglob("test_predictions.jsonl")):
    # Identify the baseline name from the path
    rel = predictions_file.relative_to(RESULTS)
    name = "/".join(rel.parts[:-1]) or rel.parts[0]
    baselines[name] = load_predictions(predictions_file)

print(f"Loaded gold: {len(gold)} test records")
print(f"Loaded baseline predictions:")
for name in sorted(baselines):
    print(f"  {name}: {len(baselines[name])} predictions")

# %% [markdown]
# ## Per-baseline confusion matrix on the test split

# %%
def confusion(predictions, gold_by_pid):
    tp = fp = tn = fn = 0
    for pid, gold_rec in gold_by_pid.items():
        true_positive = gold_rec["record_type"] == "conflict"
        pred = predictions.get(pid)
        if pred is None:
            continue
        pred_positive = pred.get("predicted_conflict_present", False)
        if true_positive and pred_positive: tp += 1
        elif true_positive and not pred_positive: fn += 1
        elif not true_positive and pred_positive: fp += 1
        else: tn += 1
    return tp, fp, fn, tn


print(f"{'baseline':>45}  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  precision  recall   F1")
for name in sorted(baselines):
    tp, fp, fn, tn = confusion(baselines[name], gold)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"{name:>45}  {tp:>3} {fp:>3} {fn:>3} {tn:>3}    {p:.3f}    {r:.3f}  {f1:.3f}")

# %% [markdown]
# ## Universally-hard cases: conflicts every baseline misses

# %%
universally_hard = []
for pid, gold_rec in gold.items():
    if gold_rec["record_type"] != "conflict":
        continue
    missed_by = []
    for name in baselines:
        pred = baselines[name].get(pid)
        if pred is None:
            continue
        if not pred.get("predicted_conflict_present", False):
            missed_by.append(name)
    if len(missed_by) >= max(1, len(baselines) - 1):  # missed by all (or all-but-one) baselines
        universally_hard.append((pid, gold_rec, missed_by))


print(f"\n{len(universally_hard)} universally-hard conflicts (missed by ≥{max(1, len(baselines)-1)}/{len(baselines)} baselines):")
for pid, gold_rec, missed_by in universally_hard[:10]:
    print(f"\n  {pid}")
    print(f"    regime_a: {gold_rec['regime_a']['regime_id']} ({gold_rec['regime_a']['jurisdiction']})")
    print(f"    regime_b: {gold_rec['regime_b']['regime_id']} ({gold_rec['regime_b']['jurisdiction']})")
    print(f"    conflict_type: {gold_rec.get('conflict_type')} (severity={gold_rec.get('severity')})")
    print(f"    missed by:     {len(missed_by)}/{len(baselines)}")

# %% [markdown]
# ## Side-by-side: one case across all baselines

# %%
if universally_hard:
    pid, gold_rec, _ = universally_hard[0]
elif gold:
    pid = next(iter(gold))
    gold_rec = gold[pid]
else:
    pid, gold_rec = None, None

if pid:
    print(f"=== Pair: {pid} ===")
    print(f"Gold: conflict={gold_rec['record_type']=='conflict'}, type={gold_rec.get('conflict_type')}, severity={gold_rec.get('severity')}")
    print(f"\n  {'baseline':>45}  pred_conflict  pred_type")
    for name in sorted(baselines):
        pred = baselines[name].get(pid, {})
        pc = pred.get("predicted_conflict_present", "?")
        pt = pred.get("predicted_conflict_type", "—")
        print(f"  {name:>45}  {str(pc):>13}  {pt}")

# %% [markdown]
# ## Per-conflict-type recall (where on the typology do baselines fail?)

# %%
recall_by_class = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # baseline → class → [n_caught, n_total]
for pid, gold_rec in gold.items():
    if gold_rec["record_type"] != "conflict":
        continue
    cls = gold_rec.get("conflict_type", "?")
    for name in baselines:
        pred = baselines[name].get(pid, {})
        recall_by_class[name][cls][1] += 1  # total
        if pred.get("predicted_conflict_present"):
            recall_by_class[name][cls][0] += 1  # caught

for name in sorted(baselines):
    by_class = recall_by_class[name]
    parts = []
    for cls in ("structural_unresolved", "operationally_resolvable",
                "interpretive_fact_sensitive", "recurring_friction"):
        if cls not in by_class:
            continue
        caught, total = by_class[cls]
        parts.append(f"{cls[:6]}={caught}/{total}")
    print(f"  {name:>45}  recall: {' | '.join(parts)}")
