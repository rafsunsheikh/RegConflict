"""TRIAG conflict-detection benchmark evaluation harness.

Public API for evaluating a predictions file against a ground-truth split.

This is the benchmark harness used for the EMNLP submission. The earlier WS6
evaluation code in src/eval/ is preserved alongside; this subpackage is the
canonical entry point for any baseline that follows the benchmark spec.
"""
from src.eval.benchmark.metrics import (
    CONFLICT_TYPES,
    compute_binary_metrics,
    compute_joint_metrics,
    compute_per_jurisdiction_metrics,
    compute_typology_metrics,
)
from src.eval.benchmark.bootstrap import bootstrap_ci, paired_bootstrap_test
from src.eval.benchmark.io import load_predictions, load_split, write_results
from src.eval.benchmark.validators import validate_predictions

__all__ = [
    "CONFLICT_TYPES",
    "compute_binary_metrics",
    "compute_typology_metrics",
    "compute_joint_metrics",
    "compute_per_jurisdiction_metrics",
    "bootstrap_ci",
    "paired_bootstrap_test",
    "load_predictions",
    "load_split",
    "write_results",
    "validate_predictions",
]
