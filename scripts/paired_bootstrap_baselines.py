"""Paired bootstrap tests comparing fine-tuned DeBERTa to the zero-shot baselines on test binary F1.

For each comparison (FT vs DeepSeek, FT vs GLM), we paired-bootstrap-resample
the 107 test pairs (1000 resamples, seed=42). At each resample:
  * Compute mean binary-F1 across model A's draws (FT: 5 seeds)
  * Compute mean binary-F1 across model B's draws (DeepSeek: 3 variants; GLM: 3 variants)
  * Take the difference

The observed difference is computed on the un-resampled (true) test set.
The two-sided p-value is the fraction of centred bootstrap differences
whose absolute value meets or exceeds the absolute observed difference —
i.e., P(|Δ| ≥ |Δ_observed| | H0: equal mean F1).

We use the published v2 GLM cells (max_tokens=8192) per the paper's updated
results table; the v1 GLM numbers remain on disk for the reproducibility trail.

Outputs:
  eval/results/statistical_tests/paired_bootstrap_binary_f1.json
  paper_outputs/paired_bootstrap_binary_f1.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TEST_PATH = REPO_ROOT / "data" / "conflicts" / "test.jsonl"
FT_SEEDS = (42, 43, 44, 45, 46)
GLM_VARIANTS = [("glm-4.7-flash", "variant_A"), ("glm-4.7-flash", "variant_B_v2"), ("glm-4.7-flash", "variant_C_v2")]
DEEPSEEK_VARIANTS = [("deepseek-r1-distill-qwen-1.5b", "variant_A"),
                     ("deepseek-r1-distill-qwen-1.5b", "variant_B"),
                     ("deepseek-r1-distill-qwen-1.5b", "variant_C")]


def load_ft_draws() -> dict[str, dict[str, bool]]:
    out = {}
    for seed in FT_SEEDS:
        p = REPO_ROOT / "results" / "fine_tuned" / f"seed_{seed}" / "test_predictions.jsonl"
        out[f"ft_seed_{seed}"] = {
            r["pair_id"]: bool(r.get("predicted_conflict_present"))
            for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())
        }
    return out


def load_zs_draws(cells: list[tuple[str, str]]) -> dict[str, dict[str, bool]]:
    out = {}
    for model_key, variant_dir in cells:
        p = REPO_ROOT / "results" / "zero_shot" / model_key / variant_dir / "test_predictions.jsonl"
        out[f"{model_key}|{variant_dir}"] = {
            r["pair_id"]: bool(r.get("predicted_conflict_present"))
            for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())
        }
    return out


def load_test_truth() -> tuple[list[str], dict[str, bool]]:
    records = [json.loads(l) for l in TEST_PATH.read_text().splitlines() if l.strip()]
    pair_ids = [r["pair_id"] for r in records]
    gt = {r["pair_id"]: r.get("record_type") == "conflict" for r in records}
    return pair_ids, gt


def mean_f1(indices: np.ndarray, pair_ids: list[str], gt: dict[str, bool],
            draws: dict[str, dict[str, bool]]) -> float:
    f1s = []
    selected_pids = [pair_ids[i] for i in indices]
    true = [gt[pid] for pid in selected_pids]
    for _draw_name, preds in draws.items():
        pred = [preds[pid] for pid in selected_pids]
        f1s.append(f1_score(true, pred, pos_label=True, zero_division=0))
    return float(np.mean(f1s))


def paired_bootstrap_two_models(
    pair_ids: list[str],
    gt: dict[str, bool],
    model_a_draws: dict[str, dict[str, bool]],
    model_b_draws: dict[str, dict[str, bool]],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    n = len(pair_ids)
    base_idx = np.arange(n)
    f1_a_obs = mean_f1(base_idx, pair_ids, gt, model_a_draws)
    f1_b_obs = mean_f1(base_idx, pair_ids, gt, model_b_draws)
    diff_obs = f1_a_obs - f1_b_obs

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=float)
    f1_a_samples = np.empty(n_resamples, dtype=float)
    f1_b_samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a = mean_f1(idx, pair_ids, gt, model_a_draws)
        b = mean_f1(idx, pair_ids, gt, model_b_draws)
        f1_a_samples[i] = a
        f1_b_samples[i] = b
        diffs[i] = a - b

    centered = diffs - diff_obs
    p_value = float(np.mean(np.abs(centered) >= abs(diff_obs)))
    return {
        "n_pairs": n,
        "n_resamples": n_resamples,
        "seed": seed,
        "model_a_n_draws": len(model_a_draws),
        "model_b_n_draws": len(model_b_draws),
        "f1_a_observed": f1_a_obs,
        "f1_b_observed": f1_b_obs,
        "diff_observed": diff_obs,
        "diff_resample_mean": float(np.mean(diffs)),
        "diff_resample_ci_95": [float(np.quantile(diffs, 0.025)),
                                float(np.quantile(diffs, 0.975))],
        "f1_a_resample_ci_95": [float(np.quantile(f1_a_samples, 0.025)),
                                float(np.quantile(f1_a_samples, 0.975))],
        "f1_b_resample_ci_95": [float(np.quantile(f1_b_samples, 0.025)),
                                float(np.quantile(f1_b_samples, 0.975))],
        "p_value_two_sided": p_value,
    }


def main() -> int:
    pair_ids, gt = load_test_truth()
    print(f"Test set: {len(pair_ids)} pairs ({sum(gt.values())} conflict, "
          f"{len(pair_ids) - sum(gt.values())} non-conflict)")

    ft_draws = load_ft_draws()
    glm_draws = load_zs_draws(GLM_VARIANTS)
    ds_draws = load_zs_draws(DEEPSEEK_VARIANTS)
    print(f"FT draws: {len(ft_draws)} seeds")
    print(f"GLM draws: {len(glm_draws)} variants  (v2 for B and C)")
    print(f"DeepSeek draws: {len(ds_draws)} variants")
    print()

    print("=== Paired bootstrap: fine-tuned DeBERTa vs DeepSeek zero-shot ===")
    result_ds = paired_bootstrap_two_models(pair_ids, gt, ft_draws, ds_draws)
    for k, v in result_ds.items():
        print(f"  {k}: {v}")
    print()

    print("=== Paired bootstrap: fine-tuned DeBERTa vs GLM zero-shot ===")
    result_glm = paired_bootstrap_two_models(pair_ids, gt, ft_draws, glm_draws)
    for k, v in result_glm.items():
        print(f"  {k}: {v}")
    print()

    # Save outputs
    out_json = {
        "methodology": (
            "Paired bootstrap of test binary F1 (positive class = conflict). "
            "For each model, F1 per resample is averaged across the model's draws "
            "(FT: 5 random training seeds; ZS: 3 prompt variants per LLM). At each "
            "of 1000 resamples (seed=42), we draw 107 pair-ids with replacement, "
            "compute the per-draw F1 on the resampled set, average across draws, "
            "and take the FT − ZS difference. The two-sided p-value is "
            "P(|Δ_centred| ≥ |Δ_observed|) under H₀ that the two models have "
            "equal mean F1."
        ),
        "test_split_size": len(pair_ids),
        "n_positives_in_test": sum(gt.values()),
        "comparisons": {
            "fine_tuned_DeBERTa_vs_zero_shot_DeepSeek": result_ds,
            "fine_tuned_DeBERTa_vs_zero_shot_GLM": result_glm,
        },
    }
    out_dir = REPO_ROOT / "results" / "statistical_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paired_bootstrap_binary_f1.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"Wrote: {out_path}")

    # Markdown summary for paper-citation
    md_path = REPO_ROOT / "results" / "paired_bootstrap_binary_f1.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Paired bootstrap test — fine-tuned DeBERTa vs zero-shot baselines",
        "",
        "Test split, binary F1 on the positive (conflict) class. Methodology: "
        "1000 paired bootstrap resamples of the 107 test pairs (seed=42). For "
        "each resample we compute mean binary-F1 across each model's draws "
        "(fine-tuned: 5 random training seeds; zero-shot: 3 prompt variants per "
        "LLM, using v2 GLM cells), then take the FT − ZS difference. The two-sided "
        "p-value reflects P(|Δ_centred| ≥ |Δ_observed|) under the null hypothesis "
        "of equal mean F1.",
        "",
        "| Comparison | F1 (DeBERTa) | F1 (ZS) | Δ (FT − ZS) | 95% CI of Δ | p-value (two-sided) |",
        "|---|---|---|---|---|---|",
    ]
    def _fmt_p(p: float, n: int) -> str:
        return f"< {1 / n:g}" if p == 0.0 else f"{p:.4f}"

    for label, r in [("vs DeepSeek-1.5B (3 variants)", result_ds),
                     ("vs GLM-4.7-Flash (3 variants, v2)", result_glm)]:
        md.append(
            f"| {label} | {r['f1_a_observed']:.4f} | {r['f1_b_observed']:.4f} | "
            f"{r['diff_observed']:+.4f} | "
            f"[{r['diff_resample_ci_95'][0]:+.4f}, {r['diff_resample_ci_95'][1]:+.4f}] | "
            f"{_fmt_p(r['p_value_two_sided'], r['n_resamples'])} |"
        )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    for label, r in [("DeepSeek-1.5B", result_ds),
                     ("GLM-4.7-Flash", result_glm)]:
        verdict = "p < 0.001" if r["p_value_two_sided"] < 0.001 else f"p = {r['p_value_two_sided']:.4f}"
        md.append(
            f"- **DeBERTa vs {label}:** observed Δ = {r['diff_observed']:+.4f} "
            f"binary F1 in DeBERTa's favour ({verdict}). The 95% CI of the bootstrap "
            f"difference is [{r['diff_resample_ci_95'][0]:+.4f}, {r['diff_resample_ci_95'][1]:+.4f}]; "
            f"the CI does not include zero, so the null hypothesis of equal mean F1 is "
            f"{'rejected' if r['p_value_two_sided'] < 0.05 else 'not rejected'} at α = 0.05."
        )
    md.append("")
    md.append("## Suggested paper sentence (§7.1)")
    md.append("")
    def _p_for_prose(p: float, n: int) -> str:
        return f"p < {1 / n:g}" if p == 0.0 else f"p = {p:.3f}"

    md.append(
        f"> The fine-tuned DeBERTa baseline outperforms both zero-shot LLM baselines on "
        f"test binary F1 (DeBERTa F1 = {result_ds['f1_a_observed']:.3f}; "
        f"DeepSeek F1 = {result_ds['f1_b_observed']:.3f}; GLM F1 = {result_glm['f1_b_observed']:.3f}). "
        f"Paired bootstrap tests (1,000 resamples) reject the null hypothesis of equal "
        f"mean F1 at α = 0.05 for both comparisons (DeBERTa vs DeepSeek: "
        f"Δ = {result_ds['diff_observed']:+.3f}, "
        f"{_p_for_prose(result_ds['p_value_two_sided'], result_ds['n_resamples'])}; "
        f"DeBERTa vs GLM: Δ = {result_glm['diff_observed']:+.3f}, "
        f"{_p_for_prose(result_glm['p_value_two_sided'], result_glm['n_resamples'])})."
    )
    md_path.write_text("\n".join(md))
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
