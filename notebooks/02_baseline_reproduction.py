# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 02 — Baseline reproduction
#
# Reproduce the paper's reported baseline numbers from the released code and data.
#
# **Expected outputs** (v1.0 release):
#
# | Baseline | Test binary F1 |
# |---|---|
# | Majority class | 0.000 |
# | Stratified random (mean ± std over 100 seeds) | 0.127 ± 0.077 |
# | Zero-shot DeepSeek-1.5B (mean ± std over 3 variants) | 0.135 ± 0.098 |
# | Zero-shot GLM-4.7-Flash (mean ± std over 3 variants) | 0.100 ± 0.082 |
# | Fine-tuned DeBERTa-v3-base (mean ± std over 5 seeds) | **0.517 ± 0.022** |
#
# The DeBERTa baseline's advantage over both zero-shot LLMs is statistically significant under paired bootstrap (p = 0.002 vs DeepSeek, p < 0.001 vs GLM).
#
# This notebook runs the random baselines end-to-end (no GPU or network required). Zero-shot LLM and fine-tuned baselines require external setup (llama.cpp servers / GPU); the cells for those reference the relevant scripts and expected outputs.

# %%
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# %% [markdown]
# ## 1. Random baselines (no GPU / network needed)
#
# Runs majority class + 100-seed stratified random against test + gold_iaa.

# %%
result = subprocess.run(
    ["python3", str(REPO_ROOT / "scripts" / "run_random_baselines.py")],
    cwd=str(REPO_ROOT),
    capture_output=True,
    text=True,
    timeout=300,
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])

# %% [markdown]
# ## 2. Inspect the headline numbers

# %%
results_dir = REPO_ROOT / "results"
for baseline in ("majority_class", "stratified_random"):
    metrics_files = sorted((results_dir / baseline).rglob("metrics.json"))
    if not metrics_files:
        print(f"{baseline}: no metrics.json found (did the runner write outputs?)")
        continue
    # Most-recent metrics file for each baseline
    latest = metrics_files[-1]
    metrics = json.loads(latest.read_text())
    print(f"\n=== {baseline} ===")
    print(f"  test binary F1: {metrics.get('binary_f1', '?'):.4f}")
    print(f"  test accuracy:  {metrics.get('accuracy', '?'):.4f}")

# %% [markdown]
# ## 3. Zero-shot LLM baseline (requires llama.cpp servers running)
#
# The zero-shot baseline needs two llama.cpp servers running on ports 8080 (GLM) and 8081 (DeepSeek). Launch commands and SHA-256 checksums for the GGUF models are in Appendix D of the paper.
#
# Once both servers are up:
#
# ```bash
# python scripts/run_zero_shot_baseline.py
# ```
#
# This runs 2 models × 3 variants × 2 splits = 12 evaluation cells, each emitting per-cell `predictions.jsonl` / `metrics.json` under `results/zero_shot/`. Expect ~30 minutes to several hours depending on the model server's throughput and the variant (Variant B/C reasoning chains at max_tokens=8192 can take several minutes per call).
#
# Expected v1.0 test split numbers:
# - GLM-4.7-Flash (mean over 3 variants): binary F1 = 0.100 ± 0.082
# - DeepSeek-R1-Distill-Qwen-1.5B (mean over 3 variants): binary F1 = 0.135 ± 0.098

# %% [markdown]
# ## 4. Fine-tuned DeBERTa baseline (requires GPU or Apple Silicon)
#
# The DeBERTa-v3-base cross-encoder baseline trains a 184M-parameter model with gradient checkpointing. On Apple Silicon (M1/M2/M3) expect ~15 minutes per seed × 5 seeds = ~75 minutes for the full multi-seed run; on NVIDIA hardware it's faster.
#
# ```bash
# python scripts/train_deberta_baseline.py --mode multi_seed
# ```
#
# Expected v1.0 test split numbers (mean ± std over seeds 42–46):
# - binary F1 = **0.517 ± 0.022**
# - macro-F1 = 0.717 ± 0.015
# - typology macro-F1 = 0.219 ± 0.000

# %% [markdown]
# ## 5. Paired bootstrap significance test
#
# Once all three baseline families have been evaluated:
#
# ```bash
# python scripts/paired_bootstrap_baselines.py
# ```
#
# This runs 1,000 paired bootstrap resamples on the 107-pair test set, computing the binary-F1 difference distribution for DeBERTa vs each zero-shot baseline. Expected output:
#
# - DeBERTa vs DeepSeek: Δ = +0.382 (p = 0.002, two-sided)
# - DeBERTa vs GLM: Δ = +0.417 (p < 0.001, two-sided)
#
# Both 95% CIs on Δ exclude zero, so the null hypothesis of equal mean F1 is rejected at α = 0.05 for both comparisons.

# %% [markdown]
# ## 6. Inter-annotator agreement
#
# Once the 30-pair gold IAA second-annotator data is merged into `data/annotations/iaa/`:
#
# ```bash
# python scripts/compute_iaa.py
# ```
#
# Expected: Cohen's κ = 0.6 on the four-class typology task — substantial agreement on the Landis–Koch scale.
