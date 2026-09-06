"""Object-level K3 metrics, seed ensembles, and locked promotion gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..physics.axial import axial_angular_error_deg
from .grid import AxialMode, extract_axial_modes


class K3EvaluationError(ValueError):
    """Raised when evaluation arrays or gate inputs are inconsistent."""


def oracle_at_k_error_deg(predictions: np.ndarray, targets: np.ndarray, *, k: int = 3) -> float:
    """Return the best antipode-aware prediction/target separation."""
    predicted = np.asarray(predictions, dtype=np.float64)
    source = np.asarray(targets, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or source.ndim != 2 or source.shape[1] != 3:
        raise K3EvaluationError("predictions and targets must have shape [N,3]")
    if predicted.shape[0] != k or source.shape[0] == 0:
        raise K3EvaluationError("oracle_at_k requires exactly k predictions and at least one target")
    errors = np.asarray(axial_angular_error_deg(predicted[:, None, :], source[None, :, :]))
    return float(np.min(errors))


def summarize_errors(errors: Sequence[float]) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 90):
        raise K3EvaluationError("errors must be finite values in [0,90]")
    return {
        "n_objects": float(values.size),
        "mean_error_deg": float(np.mean(values)),
        "median_error_deg": float(np.median(values)),
        "fraction_within_20_deg": float(np.mean(values <= 20.0)),
        "fraction_within_30_deg": float(np.mean(values <= 30.0)),
    }


def ensemble_score_grids(score_grids: Sequence[np.ndarray]) -> np.ndarray:
    """Average complete score maps before deterministic mode extraction."""
    values = np.asarray(score_grids, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] != 6144 or not np.all(np.isfinite(values)):
        raise K3EvaluationError("ensemble score grids must have shape [seeds,objects,6,144]")
    result = np.mean(values, axis=0)
    if not np.all(np.isfinite(result)):
        raise K3EvaluationError("ensemble score grid is non-finite")
    return result


def modes_from_score_grid(score_grid: np.ndarray) -> tuple[AxialMode, ...]:
    values = np.asarray(score_grid, dtype=np.float64)
    if values.shape != (6144,):
        raise K3EvaluationError("one score grid must contain 6,144 axial scores")
    return extract_axial_modes(values)


def paired_bootstrap_ci(
    improvements: Sequence[float], *, resamples: int = 10000, seed: int = 20260901
) -> tuple[float, float]:
    values = np.asarray(improvements, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise K3EvaluationError("paired improvements must be a finite vector of length at least two")
    if resamples <= 0:
        raise K3EvaluationError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_permutation_pvalue(
    improvements: Sequence[float], *, sign_flips: int = 100000, seed: int = 20260901
) -> float:
    values = np.asarray(improvements, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise K3EvaluationError("paired improvements must be a finite vector of length at least two")
    if sign_flips <= 0:
        raise K3EvaluationError("permutation sign_flips must be positive")
    rng = np.random.default_rng(seed)
    observed = float(np.mean(values))
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(sign_flips, values.size))
    permuted = np.mean(signs * values[None, :], axis=1)
    # One-sided paired sign-flip test: with positive improvement, count null
    # statistics at least as large as the observed improvement (and vice versa).
    exceedance = permuted >= observed if observed > 0 else permuted <= observed
    return float((1 + np.sum(exceedance)) / (sign_flips + 1))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down familywise adjustment with stable key order."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise K3EvaluationError("p-values must lie in [0,1]")
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


@dataclass(frozen=True)
class K3PromotionVerdict:
    passed: bool
    metrics: dict[str, float]
    p_values_holm: dict[str, float]
    failures: tuple[str, ...]

    def as_mapping(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "p_values_holm": dict(self.p_values_holm),
            "failures": list(self.failures),
        }


def evaluate_oof_promotion_gate(
    k3_errors: Sequence[float],
    *,
    v1_errors: Sequence[float],
    atlas_errors: Sequence[float],
    deranged_errors: Sequence[float],
    seed_k3_errors: Mapping[int, Sequence[float]] | None = None,
    seed_input_swap_errors: Mapping[int, Sequence[float]] | None = None,
    bootstrap_resamples: int = 10000,
    permutation_sign_flips: int = 100000,
    seed: int = 20260901,
) -> K3PromotionVerdict:
    """Apply every locked retrospective OOF gate without threshold tuning."""
    methods = {"k3": np.asarray(k3_errors, dtype=np.float64), "v1": np.asarray(v1_errors, dtype=np.float64), "atlas": np.asarray(atlas_errors, dtype=np.float64), "deranged": np.asarray(deranged_errors, dtype=np.float64)}
    lengths = {values.size for values in methods.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise K3EvaluationError("all comparator vectors must have the same length and at least two objects")
    for name, values in methods.items():
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 90):
            raise K3EvaluationError(f"{name} errors are invalid")
    metrics = {}
    metrics.update({f"k3_{name}": value for name, value in summarize_errors(methods["k3"]).items() if name != "n_objects"})
    p_values: dict[str, float] = {}
    failures: list[str] = []
    if metrics["k3_mean_error_deg"] > 22.0:
        failures.append("K3 mean oracle-at-3 error exceeds 22 degrees")
    if metrics["k3_median_error_deg"] > 20.0:
        failures.append("K3 median oracle-at-3 error exceeds 20 degrees")
    if metrics["k3_fraction_within_20_deg"] < 0.50:
        failures.append("fewer than half of objects are within 20 degrees")
    for comparator in ("v1", "atlas", "deranged"):
        improvement = methods[comparator] - methods["k3"]
        mean_improvement = float(np.mean(improvement))
        ci_low, ci_high = paired_bootstrap_ci(improvement, resamples=bootstrap_resamples, seed=seed)
        p_value = paired_permutation_pvalue(improvement, sign_flips=permutation_sign_flips, seed=seed)
        metrics[f"improvement_vs_{comparator}_deg"] = mean_improvement
        metrics[f"improvement_vs_{comparator}_ci95_low_deg"] = ci_low
        metrics[f"improvement_vs_{comparator}_ci95_high_deg"] = ci_high
        p_values[f"vs_{comparator}"] = p_value
        if mean_improvement < 5.0:
            failures.append(f"point improvement versus {comparator} is below 5 degrees")
        if ci_low < 2.0:
            failures.append(f"bootstrap lower improvement versus {comparator} is below 2 degrees")
    adjusted = holm_adjust(p_values)
    if any(value > 0.01 for value in adjusted.values()):
        failures.append("Holm-adjusted paired permutation p-value exceeds 0.01")
    if (seed_k3_errors is None) != (seed_input_swap_errors is None):
        raise K3EvaluationError(
            "per-seed raw and input-swap errors must be supplied together"
        )
    if seed_k3_errors is not None and seed_input_swap_errors is not None:
        if set(seed_k3_errors) != set(seed_input_swap_errors):
            raise K3EvaluationError("per-seed raw/input-swap seed sets differ")
        for seed_value, seed_errors in seed_k3_errors.items():
            seed_values = np.asarray(seed_errors, dtype=np.float64)
            swap_values = np.asarray(
                seed_input_swap_errors[seed_value], dtype=np.float64
            )
            if (
                seed_values.shape != methods["k3"].shape
                or swap_values.shape != methods["k3"].shape
                or not np.all(np.isfinite(seed_values))
                or not np.all(np.isfinite(swap_values))
                or np.any(seed_values < 0)
                or np.any(seed_values > 90)
                or np.any(swap_values < 0)
                or np.any(swap_values > 90)
            ):
                raise K3EvaluationError(
                    f"seed {seed_value} K3 raw/input-swap errors are invalid"
                )
            gap = float(np.mean(swap_values - seed_values))
            metrics[f"seed_{seed_value}_input_swap_gap_deg"] = gap
            if gap <= 0:
                failures.append(
                    f"seed {seed_value} has no positive input-swap gap"
                )
    return K3PromotionVerdict(not failures, metrics, adjusted, tuple(failures))
