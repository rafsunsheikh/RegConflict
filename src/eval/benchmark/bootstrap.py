"""Bootstrap utilities for confidence intervals and paired significance tests.

The spec calls for 1000 resamples with 95% CIs and a paired bootstrap test for
two-system comparison. Seed is fixed (default 42) for reproducibility.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    n: int,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap a scalar metric.

    `metric_fn` receives an array of integer indices into the dataset (length n)
    and returns a scalar. The caller is responsible for binding the underlying
    arrays into the closure.
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        val = metric_fn(idx)
        samples[i] = val if val is not None else np.nan
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n_resamples": 0}
    alpha = (1.0 - ci_level) / 2.0
    return {
        "mean": float(samples.mean()),
        "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
        "n_resamples": int(len(samples)),
    }


def paired_bootstrap_test(
    metric_a_fn: Callable[[np.ndarray], float],
    metric_b_fn: Callable[[np.ndarray], float],
    n: int,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap: for each resample, evaluate both systems on the same indices.

    Returns the mean difference (A − B), CI of the difference, and a two-sided
    p-value approximating the probability of observing |Δ| ≥ |observed Δ| under
    the null hypothesis that the systems are equivalent.
    """
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a = metric_a_fn(idx)
        b = metric_b_fn(idx)
        diffs[i] = (a if a is not None else np.nan) - (b if b is not None else np.nan)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return {
            "mean_diff": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "n_resamples": 0,
        }
    observed = float(diffs.mean())
    # Centre the distribution at zero (null), then count how often |centered| ≥ |observed|
    centered = diffs - observed
    p_value = float(np.mean(np.abs(centered) >= abs(observed)))
    return {
        "mean_diff": observed,
        "ci_lower": float(np.quantile(diffs, 0.025)),
        "ci_upper": float(np.quantile(diffs, 0.975)),
        "p_value": p_value,
        "n_resamples": int(len(diffs)),
    }
