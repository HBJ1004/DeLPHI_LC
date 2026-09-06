"""Object-pooled, preregistered statistics for DeLPHI V2 promotion gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

BOOTSTRAP_SEED = 20260901
BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_SIGN_FLIPS = 100_000


class StatisticsContractError(ValueError):
    """Raised when a result table cannot support a scientific claim."""


@dataclass(frozen=True)
class PairedBootstrapResult:
    n_objects: int
    point_delta: float
    ci95_lower: float
    ci95_upper: float
    one_sided_p_value: float
    seed: int
    resamples: int


@dataclass(frozen=True)
class PairedPermutationResult:
    n_objects: int
    observed_mean_delta: float
    one_sided_p_value: float
    seed: int
    sign_flips: int


def require_one_object_one_vote(object_ids: Sequence[str]) -> None:
    if not object_ids or any(not isinstance(item, str) or not item for item in object_ids):
        raise StatisticsContractError("at least one nonempty object ID is required")
    duplicate_ids = sorted({item for item in object_ids if object_ids.count(item) > 1})
    if duplicate_ids:
        raise StatisticsContractError(
            f"each object must contribute exactly one row: {duplicate_ids[:10]}"
        )


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise StatisticsContractError(f"{name} must be a nonempty finite vector")
    return vector


def object_pooled_summary(
    object_ids: Sequence[str], errors_deg: Sequence[float]
) -> dict[str, float | int]:
    """Summarize errors with exactly one vote per object, never per epoch/fold."""
    require_one_object_one_vote(object_ids)
    errors = _finite_vector(errors_deg, "errors_deg")
    if errors.size != len(object_ids) or np.any(errors < 0):
        raise StatisticsContractError("errors_deg must align with IDs and be nonnegative")
    return {
        "n_objects": int(errors.size),
        "mean_error_deg": float(np.mean(errors)),
        "median_error_deg": float(np.median(errors)),
        "fraction_within_10_deg": float(np.mean(errors <= 10.0)),
        "fraction_within_20_deg": float(np.mean(errors <= 20.0)),
        "fraction_within_30_deg": float(np.mean(errors <= 30.0)),
    }


def aggregate_seed_rows(
    rows: Sequence[Mapping[str, object]],
    metric_key: str,
    *,
    required_seeds: Sequence[int] = (17, 42, 137, 777, 2027),
) -> tuple[list[str], np.ndarray]:
    """Reduce complete object-by-seed OOF rows to one mean metric per object."""
    if not required_seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in required_seeds
    ):
        raise StatisticsContractError("required_seeds must be nonempty integers")
    required = set(required_seeds)
    if len(required) != len(required_seeds):
        raise StatisticsContractError("required_seeds must be unique")
    grouped: dict[str, dict[int, float]] = {}
    for row in rows:
        object_id = row.get("object_id")
        seed = row.get("seed")
        metric = row.get(metric_key)
        if (
            not isinstance(object_id, str)
            or not object_id
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise StatisticsContractError("OOF rows require string object_id and integer seed")
        if (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(metric)
        ):
            raise StatisticsContractError(f"OOF rows require finite {metric_key}")
        if seed not in required:
            raise StatisticsContractError(f"unexpected OOF seed: {seed}")
        by_seed = grouped.setdefault(object_id, {})
        if seed in by_seed:
            raise StatisticsContractError("duplicate object/seed OOF row")
        by_seed[seed] = float(metric)
    if not grouped:
        raise StatisticsContractError("at least one OOF row is required")
    if any(set(by_seed) != required for by_seed in grouped.values()):
        raise StatisticsContractError(
            "every object requires exactly one row for every preregistered seed"
        )
    object_ids = sorted(grouped)
    ordered_seeds = tuple(sorted(required))
    return object_ids, np.asarray(
        [np.mean([grouped[item][seed] for seed in ordered_seeds]) for item in object_ids],
        dtype=np.float64,
    )


def paired_object_bootstrap(
    object_ids: Sequence[str],
    model_errors_deg: Sequence[float],
    comparator_errors_deg: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> PairedBootstrapResult:
    """Bootstrap comparator-minus-model error; positive values favor the model."""
    require_one_object_one_vote(object_ids)
    model = _finite_vector(model_errors_deg, "model_errors_deg")
    comparator = _finite_vector(comparator_errors_deg, "comparator_errors_deg")
    if model.shape != comparator.shape or model.size != len(object_ids):
        raise StatisticsContractError("paired error vectors must align with object IDs")
    if np.any(model < 0) or np.any(comparator < 0) or resamples < 1000:
        raise StatisticsContractError("errors must be nonnegative and resamples >= 1000")
    delta = comparator - model
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, delta.size, size=(resamples, delta.size))
    samples = delta[indexes].mean(axis=1)
    # Add one for a conservative finite-resample one-sided p-value.
    p_value = (1 + int(np.count_nonzero(samples <= 0.0))) / (resamples + 1)
    return PairedBootstrapResult(
        n_objects=int(delta.size),
        point_delta=float(delta.mean()),
        ci95_lower=float(np.quantile(samples, 0.025)),
        ci95_upper=float(np.quantile(samples, 0.975)),
        one_sided_p_value=float(p_value),
        seed=seed,
        resamples=resamples,
    )


def paired_sign_flip_permutation(
    object_ids: Sequence[str],
    model_errors_deg: Sequence[float],
    comparator_errors_deg: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    sign_flips: int = PERMUTATION_SIGN_FLIPS,
) -> PairedPermutationResult:
    """One-sided paired randomization test; positive deltas favor the model."""
    require_one_object_one_vote(object_ids)
    model = _finite_vector(model_errors_deg, "model_errors_deg")
    comparator = _finite_vector(comparator_errors_deg, "comparator_errors_deg")
    if model.shape != comparator.shape or model.size != len(object_ids):
        raise StatisticsContractError("paired error vectors must align with object IDs")
    if np.any(model < 0) or np.any(comparator < 0):
        raise StatisticsContractError("errors must be nonnegative")
    if isinstance(sign_flips, bool) or not isinstance(sign_flips, int) or sign_flips < 1_000:
        raise StatisticsContractError("sign_flips must be an integer >= 1000")
    delta = comparator - model
    observed = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    # Chunking bounds memory for the preregistered 100,000 by 170 calculation.
    while completed < sign_flips:
        count = min(10_000, sign_flips - completed)
        signs = rng.integers(0, 2, size=(count, delta.size), dtype=np.int8) * 2 - 1
        permuted = np.mean(signs * delta, axis=1)
        extreme += int(np.count_nonzero(permuted >= observed))
        completed += count
    return PairedPermutationResult(
        n_objects=int(delta.size),
        observed_mean_delta=observed,
        one_sided_p_value=float((extreme + 1) / (sign_flips + 1)),
        seed=seed,
        sign_flips=sign_flips,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values keyed by preregistered test ID."""
    if not p_values:
        raise StatisticsContractError("at least one p-value is required")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    n = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise StatisticsContractError("p-values must lie in [0, 1]")
        running = max(running, min(1.0, (n - index) * value))
        adjusted[name] = running
    return adjusted


def risk_coverage_curve(
    object_ids: Sequence[str],
    risk_deg: Sequence[float],
    errors_deg: Sequence[float],
    coverages: Iterable[float] = (0.5, 0.8, 1.0),
) -> list[dict[str, float | int]]:
    """Compute selective risk after retaining the lowest declared-risk objects."""
    require_one_object_one_vote(object_ids)
    risk = _finite_vector(risk_deg, "risk_deg")
    errors = _finite_vector(errors_deg, "errors_deg")
    if (
        risk.shape != errors.shape
        or risk.size != len(object_ids)
        or np.any(risk < 0)
        or np.any(errors < 0)
    ):
        raise StatisticsContractError("risk/errors must align and be nonnegative")
    order = np.argsort(risk, kind="stable")
    curve: list[dict[str, float | int]] = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise StatisticsContractError("coverage must lie in (0, 1]")
        count = max(1, math.ceil(coverage * errors.size))
        selected = errors[order[:count]]
        curve.append(
            {
                "coverage": float(count / errors.size),
                "n_objects": count,
                "risk_deg": float(np.mean(selected)),
            }
        )
    return curve


def validate_checkpoint_selection(selection_split: str, outer_test_ids_used: bool) -> None:
    """Reject training metadata that used outer-test outcomes for selection."""
    if selection_split not in {"training", "validation", "calibration"} or outer_test_ids_used:
        raise StatisticsContractError(
            "checkpoint selection must use training/validation only, never outer test"
        )
