"""Pure scoring for the retrospective known-period K3 grid benchmark.

This module intentionally has no file I/O and no execution-orchestration code.
The caller must freeze and verify the label-blind execution artifact before it
passes rows and reference axes to :func:`score_rows`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.stats import beta

from ..physics.axial import axial_angular_error_deg

SCHEMA = "delphi.k3-retrospective-known-period-grid-score.v1"
ARMS = ("classical20", "guided20", "classical15", "guided15", "standard6")
TIMING_MODES = ("cold", "warm")
REPEAT_INDICES = (0, 1, 2)
RECOVERY_MAX_DEGREES = 20.0
NONINFERIORITY_MARGIN = -0.05
RMS_RATIO_MAXIMUM = 1.01
_ROW_FIELDS = {
    "object_id",
    "fold",
    "repeat_index",
    "arm",
    "timing_mode",
    "wall_seconds",
    "completed",
    "selected_fit",
}
_FIT_FIELDS = {"axis", "final_relative_rms", "start_index"}
_ANALYSES = (
    ("cold_step20_primary", "classical20", "guided20", "cold", "primary"),
    (
        "warm_step20_separately_scoped",
        "classical20",
        "guided20",
        "warm",
        "separately_scoped",
    ),
    (
        "cold_step15_sensitivity",
        "classical15",
        "guided15",
        "cold",
        "sensitivity",
    ),
    (
        "warm_step15_sensitivity",
        "classical15",
        "guided15",
        "warm",
        "sensitivity",
    ),
)


class GridBenchmarkScoringError(ValueError):
    """Raised when a scoring input violates the frozen factorial contract."""


def _plain_number(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GridBenchmarkScoringError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise GridBenchmarkScoringError(f"{name} must be {qualifier}")
    return result


def _vector(value: object, *, name: str, require_unit: bool) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise GridBenchmarkScoringError(f"{name} must contain exactly three numbers")
    components = np.asarray(
        [_plain_number(component, name=f"{name} component") for component in value],
        dtype=np.float64,
    )
    # Scaling before normalization avoids overflow and underflow for otherwise
    # valid finite reference vectors.
    scale = float(np.max(np.abs(components)))
    if scale == 0.0:
        raise GridBenchmarkScoringError(f"{name} must be nonzero")
    scaled = components / scale
    scaled_norm = float(np.linalg.norm(scaled))
    if not math.isfinite(scaled_norm) or scaled_norm == 0.0:
        raise GridBenchmarkScoringError(f"{name} cannot be normalized")
    unit = scaled / scaled_norm
    if require_unit:
        original_norm = math.hypot(*(float(component) for component in components))
        if not math.isclose(original_norm, 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise GridBenchmarkScoringError(f"{name} must be a unit vector")
    return unit


def _validate_inputs(
    rows: list[Mapping[str, object]], references: Mapping[str, Sequence[Sequence[float]]]
) -> tuple[
    tuple[str, ...],
    dict[str, int],
    dict[tuple[str, str, str, int], dict[str, object]],
    dict[str, np.ndarray],
]:
    if not isinstance(rows, list):
        raise GridBenchmarkScoringError("rows must be a list")
    if not isinstance(references, Mapping):
        raise GridBenchmarkScoringError("references must be a mapping")

    records: dict[tuple[str, str, str, int], dict[str, object]] = {}
    folds: dict[str, int] = {}
    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != _ROW_FIELDS:
            raise GridBenchmarkScoringError(
                f"row {position} must have exactly the frozen grid-benchmark fields"
            )
        object_id = raw["object_id"]
        if not isinstance(object_id, str) or not object_id:
            raise GridBenchmarkScoringError(f"row {position} has an invalid object_id")
        fold = raw["fold"]
        repeat = raw["repeat_index"]
        if isinstance(fold, bool) or not isinstance(fold, Integral) or int(fold) not in range(5):
            raise GridBenchmarkScoringError(f"row {position} has an invalid fold")
        if (
            isinstance(repeat, bool)
            or not isinstance(repeat, Integral)
            or int(repeat) not in REPEAT_INDICES
        ):
            raise GridBenchmarkScoringError(f"row {position} has an invalid repeat_index")
        arm = raw["arm"]
        mode = raw["timing_mode"]
        if arm not in ARMS or mode not in TIMING_MODES:
            raise GridBenchmarkScoringError(f"row {position} has an unknown arm or timing_mode")
        seconds = _plain_number(raw["wall_seconds"], name="wall_seconds", positive=True)
        completed = raw["completed"]
        if not isinstance(completed, bool):
            raise GridBenchmarkScoringError(f"row {position} completed must be boolean")
        selected = raw["selected_fit"]
        if completed != (selected is not None):
            raise GridBenchmarkScoringError(
                "completed and selected_fit must agree exactly (success has a fit; failure has null)"
            )

        fit: dict[str, object] | None = None
        if selected is not None:
            if not isinstance(selected, Mapping) or set(selected) != _FIT_FIELDS:
                raise GridBenchmarkScoringError(
                    f"row {position} selected_fit must have exactly axis, final_relative_rms, start_index"
                )
            start_index = selected["start_index"]
            if (
                isinstance(start_index, bool)
                or not isinstance(start_index, Integral)
                or int(start_index) < 0
            ):
                raise GridBenchmarkScoringError(
                    "selected_fit.start_index must be a nonnegative integer"
                )
            fit = {
                "axis": _vector(selected["axis"], name="selected_fit.axis", require_unit=True),
                "final_relative_rms": _plain_number(
                    selected["final_relative_rms"],
                    name="selected_fit.final_relative_rms",
                    positive=True,
                ),
                "start_index": int(start_index),
            }

        key = (object_id, str(arm), str(mode), int(repeat))
        if key in records:
            raise GridBenchmarkScoringError(f"duplicate execution cell: {key!r}")
        if object_id in folds and folds[object_id] != int(fold):
            raise GridBenchmarkScoringError(f"object {object_id!r} has inconsistent folds")
        folds[object_id] = int(fold)
        records[key] = {
            "wall_seconds": seconds,
            "completed": completed,
            "selected_fit": fit,
        }

    object_ids = tuple(sorted(folds))
    if len(object_ids) not in (5, 170):
        raise GridBenchmarkScoringError(
            "the cohort must contain exactly 5 pilot or 170 full objects"
        )
    expected = {
        (object_id, arm, mode, repeat)
        for object_id in object_ids
        for arm in ARMS
        for mode in TIMING_MODES
        for repeat in REPEAT_INDICES
    }
    missing = expected - set(records)
    extra = set(records) - expected
    if missing or extra or len(rows) != len(expected):
        raise GridBenchmarkScoringError(
            f"rows must form the complete Cartesian grid; missing={len(missing)}, extra={len(extra)}"
        )
    if any(not isinstance(key, str) for key in references) or set(references) != set(object_ids):
        raise GridBenchmarkScoringError("references must have exactly the same object IDs as rows")

    normalized_references: dict[str, np.ndarray] = {}
    for object_id in object_ids:
        values = references[object_id]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) < 1:
            raise GridBenchmarkScoringError(
                f"references for {object_id!r} must be a nonempty list of axes"
            )
        normalized_references[object_id] = np.vstack(
            [
                _vector(axis, name=f"reference axis for {object_id!r}", require_unit=False)
                for axis in values
            ]
        )

    return object_ids, folds, records, normalized_references


def _axial_error_degrees(axis: np.ndarray, references: np.ndarray) -> float:
    return float(np.min(axial_angular_error_deg(axis[None, :], references)))


def _optional_summary(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _paired_binary_noninferiority(
    baseline: np.ndarray, candidate: np.ndarray, *, alpha_component: float = 0.025
) -> dict[str, object]:
    """Conservative simultaneous exact lower bound for a paired difference."""
    if (
        baseline.ndim != 1
        or baseline.shape != candidate.shape
        or baseline.size < 1
        or not np.all(np.isin(baseline, (0, 1)))
        or not np.all(np.isin(candidate, (0, 1)))
    ):
        raise GridBenchmarkScoringError("paired binary endpoints must be aligned zero/one vectors")
    n = int(baseline.size)
    favorable = int(np.sum((baseline == 0) & (candidate == 1)))
    adverse = int(np.sum((baseline == 1) & (candidate == 0)))
    favorable_lower = (
        0.0 if favorable == 0 else float(beta.ppf(alpha_component, favorable, n - favorable + 1))
    )
    adverse_upper = (
        1.0 if adverse == n else float(beta.ppf(1.0 - alpha_component, adverse + 1, n - adverse))
    )
    return {
        "object_count": n,
        "baseline_success_count": int(np.sum(baseline)),
        "candidate_success_count": int(np.sum(candidate)),
        "concordant_failure_count": int(np.sum((baseline == 0) & (candidate == 0))),
        "favorable_discordance_count": favorable,
        "adverse_discordance_count": adverse,
        "concordant_success_count": int(np.sum((baseline == 1) & (candidate == 1))),
        "baseline_success_rate": float(np.mean(baseline)),
        "candidate_success_rate": float(np.mean(candidate)),
        "point_difference": float((favorable - adverse) / n),
        "favorable_rate_cp_lower": favorable_lower,
        "adverse_rate_cp_upper": adverse_upper,
        "simultaneous_exact_lower": favorable_lower - adverse_upper,
        "component_one_sided_alpha": alpha_component,
        "coverage_at_least": 1.0 - 2.0 * alpha_component,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "gate_passed": favorable_lower - adverse_upper >= NONINFERIORITY_MARGIN,
    }


def _stratified_indices(
    folds: np.ndarray, *, resamples: int, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    blocks = []
    sizes: dict[str, int] = {}
    for fold in sorted(np.unique(folds).tolist()):
        members = np.flatnonzero(folds == fold)
        sizes[str(int(fold))] = int(members.size)
        blocks.append(rng.choice(members, size=(resamples, members.size), replace=True))
    return np.concatenate(blocks, axis=1), sizes


def _runtime_metric(
    baseline: np.ndarray,
    candidate: np.ndarray,
    folds: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    sampled, sizes = _stratified_indices(folds, resamples=resamples, seed=seed)
    ratios = np.sum(baseline[sampled], axis=(1, 2)) / np.sum(candidate[sampled], axis=(1, 2))
    lower, upper = np.quantile(ratios, (0.025, 0.975))
    point = float(np.sum(baseline) / np.sum(candidate))
    return {
        "estimand": "sum_classical_wall_seconds_divided_by_sum_guided_wall_seconds",
        "point_ratio": point,
        "classical_seconds": {
            "count": int(baseline.size),
            "sum": float(np.sum(baseline)),
            "mean": float(np.mean(baseline)),
            "median": float(np.median(baseline)),
        },
        "guided_seconds": {
            "count": int(candidate.size),
            "sum": float(np.sum(candidate)),
            "mean": float(np.mean(candidate)),
            "median": float(np.median(candidate)),
        },
        "percentile_interval_95": {"lower": float(lower), "upper": float(upper)},
        "acceptance_lower": float(lower),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "stratum_sizes": sizes,
        "resampling": "object_clustered_stratified_by_fold_preserving_all_three_repeats",
        "interval_label": "percentile_95_confidence_interval_2.5_97.5",
        "acceptance_rule": "lower_strictly_greater_than_1",
        "gate_passed": bool(lower > 1.0),
    }


def _rms_metric(
    baseline_rms: np.ndarray,
    candidate_rms: np.ndarray,
    joint: np.ndarray,
    object_ids: tuple[str, ...],
    folds: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    joint_counts = np.sum(joint, axis=1)
    supported = joint_counts >= 2
    support = int(np.sum(supported))
    required = len(object_ids)
    support_rows = [
        {
            "object_id": object_id,
            "fold": int(folds[index]),
            "joint_completed_repeat_count": int(joint_counts[index]),
            "supported": bool(supported[index]),
        }
        for index, object_id in enumerate(object_ids)
    ]
    common: dict[str, object] = {
        "estimand": "geometric_mean_object_level_selected_fit_rms_ratio_guided_over_classical",
        "support_conditioning": "joint_completion_only_without_reference_conditioning",
        "minimum_joint_repeats_per_object": 2,
        "joint_completed_repeat_count": int(np.sum(joint)),
        "object_support": support,
        "required_object_support": required,
        "full_predeclared_object_support": support == required,
        "object_joint_support": support_rows,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "resampling": "object_clustered_stratified_by_fold_preserving_repeat_aggregation",
        "interval_label": "percentile_95_confidence_interval_2.5_97.5",
        "acceptance_rule": "upper_strictly_less_than_1.01_with_full_object_support",
    }
    if support != required:
        return {
            **common,
            "estimable": False,
            "point_ratio": None,
            "median_object_ratio": None,
            "percentile_interval_95": None,
            "acceptance_upper": None,
            "gate_passed": False,
            "gate_failure": "insufficient_predeclared_object_support",
        }

    object_logs = np.asarray(
        [
            np.mean(np.log(candidate_rms[index, joint[index]] / baseline_rms[index, joint[index]]))
            for index in range(required)
        ],
        dtype=np.float64,
    )
    sampled, sizes = _stratified_indices(folds, resamples=resamples, seed=seed)
    boot = np.exp(np.mean(object_logs[sampled], axis=1))
    lower, upper = np.quantile(boot, (0.025, 0.975))
    point = float(np.exp(np.mean(object_logs)))
    return {
        **common,
        "estimable": True,
        "point_ratio": point,
        "mean_object_log_ratio": float(np.mean(object_logs)),
        "median_object_ratio": float(np.median(np.exp(object_logs))),
        "object_ratios": [
            {
                "object_id": object_id,
                "fold": int(folds[index]),
                "joint_completed_repeat_count": int(joint_counts[index]),
                "guided_over_classical_ratio": float(math.exp(object_logs[index])),
            }
            for index, object_id in enumerate(object_ids)
        ],
        "percentile_interval_95": {"lower": float(lower), "upper": float(upper)},
        "acceptance_upper": float(upper),
        "stratum_sizes": sizes,
        "gate_passed": bool(upper < RMS_RATIO_MAXIMUM),
    }


def score_rows(
    rows: list[Mapping[str, object]],
    references: Mapping[str, Sequence[Sequence[float]]],
    *,
    resamples: int = 10000,
    seed: int = 20260911,
) -> dict[str, object]:
    """Validate and score a complete pilot or full grid, without any I/O."""
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or not 1 <= resamples <= 1_000_000
    ):
        raise GridBenchmarkScoringError("resamples must be an integer from 1 through 1,000,000")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise GridBenchmarkScoringError("seed must be an integer in the NumPy uint64 range")

    object_ids, fold_lookup, records, normalized_references = _validate_inputs(rows, references)
    n = len(object_ids)
    folds = np.asarray([fold_lookup[object_id] for object_id in object_ids], dtype=np.int8)

    arm_data: dict[tuple[str, str], dict[str, Any]] = {}
    arm_summaries: dict[str, dict[str, object]] = {arm: {} for arm in ARMS}
    selected_errors: dict[str, dict[str, object]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for mode in TIMING_MODES:
            times = np.empty((n, 3), dtype=np.float64)
            completed = np.zeros((n, 3), dtype=bool)
            recovered = np.zeros((n, 3), dtype=bool)
            rms = np.full((n, 3), np.nan, dtype=np.float64)
            per_object_errors = []
            all_errors: list[float] = []
            all_rms: list[float] = []
            for object_index, object_id in enumerate(object_ids):
                errors: list[float | None] = []
                for repeat in REPEAT_INDICES:
                    record = records[(object_id, arm, mode, repeat)]
                    times[object_index, repeat] = float(record["wall_seconds"])
                    completed[object_index, repeat] = bool(record["completed"])
                    fit = record["selected_fit"]
                    if fit is None:
                        errors.append(None)
                        continue
                    value = _axial_error_degrees(
                        fit["axis"],
                        normalized_references[object_id],  # type: ignore[arg-type]
                    )
                    errors.append(value)
                    all_errors.append(value)
                    recovered[object_index, repeat] = value <= RECOVERY_MAX_DEGREES
                    rms_value = float(fit["final_relative_rms"])  # type: ignore[index]
                    rms[object_index, repeat] = rms_value
                    all_rms.append(rms_value)
                selected = [value for value in errors if value is not None]
                per_object_errors.append(
                    {
                        "object_id": object_id,
                        "fold": int(folds[object_index]),
                        "repeat_error_degrees": errors,
                        "selected_fit_count": len(selected),
                        "mean_selected_error_degrees": (
                            float(np.mean(selected)) if selected else None
                        ),
                        "median_selected_error_degrees": (
                            float(np.median(selected)) if selected else None
                        ),
                        "repeat_recovery_count": int(np.sum(recovered[object_index])),
                        "object_recovered_by_two_of_three": bool(
                            np.sum(recovered[object_index]) >= 2
                        ),
                    }
                )
            object_completion = np.sum(completed, axis=1) >= 2
            object_recovery = np.sum(recovered, axis=1) >= 2
            arm_data[(arm, mode)] = {
                "times": times,
                "completed": completed,
                "recovered": recovered,
                "object_completion": object_completion,
                "object_recovery": object_recovery,
                "rms": rms,
            }
            arm_summaries[arm][mode] = {
                "object_count": n,
                "repeat_count": n * 3,
                "runtime_seconds": {
                    "count": int(times.size),
                    "sum": float(np.sum(times)),
                    "mean": float(np.mean(times)),
                    "median": float(np.median(times)),
                },
                "completion": {
                    "repeat_success_count": int(np.sum(completed)),
                    "repeat_denominator": int(completed.size),
                    "repeat_rate": float(np.mean(completed)),
                    "object_success_count": int(np.sum(object_completion)),
                    "object_denominator": n,
                    "object_rate": float(np.mean(object_completion)),
                },
                "recovery_within_20_degrees": {
                    "repeat_success_count": int(np.sum(recovered)),
                    "repeat_denominator_including_failures": int(recovered.size),
                    "repeat_rate": float(np.mean(recovered)),
                    "object_success_count": int(np.sum(object_recovery)),
                    "object_denominator": n,
                    "object_rate": float(np.mean(object_recovery)),
                },
                "selected_axial_error_degrees": _optional_summary(all_errors),
                "selected_fit_final_relative_rms": _optional_summary(all_rms),
            }
            selected_errors[arm][mode] = per_object_errors

    analyses: dict[str, object] = {}
    for name, baseline_arm, candidate_arm, mode, scope in _ANALYSES:
        baseline = arm_data[(baseline_arm, mode)]
        candidate = arm_data[(candidate_arm, mode)]
        runtime = _runtime_metric(
            baseline["times"], candidate["times"], folds, resamples=resamples, seed=seed
        )
        completion = _paired_binary_noninferiority(
            baseline["object_completion"].astype(np.int8),
            candidate["object_completion"].astype(np.int8),
        )
        recovery = _paired_binary_noninferiority(
            baseline["object_recovery"].astype(np.int8),
            candidate["object_recovery"].astype(np.int8),
        )
        joint = baseline["completed"] & candidate["completed"]
        rms = _rms_metric(
            baseline["rms"],
            candidate["rms"],
            joint,
            object_ids,
            folds,
            resamples=resamples,
            seed=seed,
        )
        failed = []
        if not runtime["gate_passed"]:
            failed.append("runtime_ratio_percentile95_lower_strictly_above_1")
        if not recovery["gate_passed"]:
            failed.append("recovery_simultaneous_exact_lower_at_least_minus_0.05")
        if not completion["gate_passed"]:
            failed.append("completion_simultaneous_exact_lower_at_least_minus_0.05")
        if not rms["full_predeclared_object_support"]:
            failed.append("rms_full_predeclared_object_support")
        elif not rms["gate_passed"]:
            failed.append("rms_ratio_percentile95_upper_strictly_below_1.01")
        analyses[name] = {
            "scope": scope,
            "timing_mode": mode,
            "baseline_arm": baseline_arm,
            "candidate_arm": candidate_arm,
            "runtime": runtime,
            "recovery": recovery,
            "completion": completion,
            "rms": rms,
            "decision": {
                "passed_all_four_conditions": not failed,
                "failed_gates": failed,
            },
        }

    primary = analyses["cold_step20_primary"]
    warm = analyses["warm_step20_separately_scoped"]
    return {
        "schema": SCHEMA,
        "artifact_family": "retrospective_known_period_k3_grid_benchmark",
        "protocol_separation": (
            "independent_of_delphi.k3-convergence-score.v1_and_its_empirical_5_95_bounds"
        ),
        "phase": "reference_scoring_after_caller_verified_frozen_execution",
        "pure_scoring_no_io": True,
        "cohort": {
            "kind": "full" if n == 170 else "pilot",
            "object_count": n,
            "execution_cell_count": len(rows),
            "object_ids": list(object_ids),
            "fold_counts": {
                str(fold): int(np.sum(folds == fold)) for fold in sorted(np.unique(folds).tolist())
            },
        },
        "factorial": {
            "arms": list(ARMS),
            "timing_modes": list(TIMING_MODES),
            "repeat_indices": list(REPEAT_INDICES),
            "complete_cartesian_grid_validated": True,
        },
        "endpoint_definitions": {
            "axis": "antipodal_minimum_arccos_absolute_dot_over_all_admissible_references",
            "repeat_recovery": "selected_fit_axial_error_degrees_less_than_or_equal_to_20",
            "object_recovery": "at_least_two_of_three_repeat_recoveries_failures_retained",
            "object_completion": "at_least_two_of_three_completed_repeats_failures_retained",
            "binary_noninferiority": (
                "paired_object_level_conservative_simultaneous_exact_cp_lower_alpha_0.025_each"
            ),
            "rms": "object_mean_log_ratio_over_jointly_completed_repeats_minimum_two_all_objects",
        },
        "thresholds": {
            "recovery_degrees_maximum_inclusive": RECOVERY_MAX_DEGREES,
            "completion_and_recovery_noninferiority_lower_minimum": NONINFERIORITY_MARGIN,
            "runtime_ratio_percentile95_lower_strictly_greater_than": 1.0,
            "rms_ratio_percentile95_upper_strictly_less_than": RMS_RATIO_MAXIMUM,
        },
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "percentiles": [2.5, 97.5],
            "unit": "object_cluster_stratified_by_fold_preserving_repeats",
        },
        "arm_summaries": arm_summaries,
        "selected_axial_errors_by_object": selected_errors,
        "analyses": analyses,
        "standard6_descriptive": {
            "matched_object_repeat_grid": True,
            "positive_hypothesis_test": False,
            "interpretation": "descriptive_only_no_unplanned_gate",
            "cold": arm_summaries["standard6"]["cold"],
            "warm": arm_summaries["standard6"]["warm"],
        },
        "decision": {
            "primary_analysis": "cold_step20_primary",
            "primary_verdict": (
                "pass" if primary["decision"]["passed_all_four_conditions"] else "fail"
            ),
            "primary_passed_all_four_conditions": primary["decision"]["passed_all_four_conditions"],
            "failed_primary_gates": primary["decision"]["failed_gates"],
            "warm_step20_separately_scoped_passed": warm["decision"]["passed_all_four_conditions"],
            "step15_sensitivity_can_rescue_primary": False,
            "standard6_can_support_unplanned_positive_test": False,
        },
    }
