"""Pure scoring utilities for the prespecified retrospective K3 reliability study.

The functions in this module consume already-sealed prediction/fit rows.  They
do not read references, choose conditions, fit curves, or write files.  In
particular, failed eligible predictions remain in the registered angular-error
denominator at 90 degrees, while ineligible rows remain visible but are not
treated as attempted predictions.
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

FAILURE_ERROR_DEG = 90.0
VALID_STATUSES = frozenset({"ok", "failed", "ineligible"})
SAMPLING_SEED_BY_REPEAT = {0: 20260915, 1: 20260916, 2: 20260917}


class ReliabilityScoringError(ValueError):
    """Raised when sealed reliability rows violate the scoring contract."""


def withheld_normalized_curve_rms(
    observed_curves: Sequence[Sequence[float] | np.ndarray],
    predicted_curves: Sequence[Sequence[float] | np.ndarray],
) -> float:
    """Return sqrt(mean(per-curve MSE)) for fixed withheld normalized flux.

    Each curve has equal weight regardless of its number of points.  Inputs are
    assumed to have been normalized before scoring; this function deliberately
    performs no fitting, rescaling, offset correction, or period selection.
    """

    if len(observed_curves) == 0 or len(observed_curves) != len(predicted_curves):
        raise ReliabilityScoringError(
            "observed and predicted withheld curve lists must have equal nonzero length"
        )
    curve_mse: list[float] = []
    for observed, predicted in zip(observed_curves, predicted_curves, strict=True):
        y = np.asarray(observed, dtype=np.float64)
        y_hat = np.asarray(predicted, dtype=np.float64)
        if (
            y.ndim != 1
            or y.size == 0
            or y.shape != y_hat.shape
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(y_hat))
        ):
            raise ReliabilityScoringError(
                "each withheld observed/predicted pair must be finite, nonempty, and shape matched"
            )
        curve_mse.append(float(np.mean(np.square(y - y_hat))))
    return float(math.sqrt(float(np.mean(curve_mse))))


def _as_nonempty_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ReliabilityScoringError(f"{key} must be a nonempty string")
    return value


def _as_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ReliabilityScoringError(f"{key} must be an integer")
    return int(value)


def _finite_number(value: Any, *, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ReliabilityScoringError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReliabilityScoringError(f"{key} must be finite")
    return result


def _validate_axes(value: Any) -> None:
    axes = np.asarray(value, dtype=np.float64)
    if axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
        raise ReliabilityScoringError("successful rows require finite axes with shape [3,3]")
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms <= 0):
        raise ReliabilityScoringError("successful candidate axes must be nonzero")


def _axes_are_absent(value: Any) -> bool:
    return value is None or (isinstance(value, (list, tuple)) and len(value) == 0)


def validate_reliability_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_objects: Mapping[str, int],
    expected_conditions: Sequence[str],
    expected_repeats: Sequence[int] = (0, 1, 2),
    expected_repeats_by_condition: Mapping[str, Sequence[int]] | None = None,
    expected_models: Sequence[str] = ("k3",),
    sampling_seed_by_repeat: Mapping[int, int] = SAMPLING_SEED_BY_REPEAT,
) -> list[dict[str, Any]]:
    """Validate and normalize a complete frozen ensemble-row factorial.

    The required unit is one ``object_id`` x ``condition`` x ``repeat`` row for
    each model.  ``model_id`` may be omitted and then defaults to ``"k3"``.
    All scheduled cells must exist exactly once, including explicitly
    ineligible cells. ``expected_repeats_by_condition`` overrides the fallback
    repeat schedule for deterministic one-pass conditions. The caller is
    responsible for binding the rows to the frozen input and prediction
    checksums before calling this pure scorer.
    """

    objects = dict(expected_objects)
    conditions = tuple(expected_conditions)
    repeats = tuple(int(value) for value in expected_repeats)
    models = tuple(expected_models)
    seed_lookup = {int(key): int(value) for key, value in sampling_seed_by_repeat.items()}
    repeat_overrides = expected_repeats_by_condition or {}
    unknown_repeat_conditions = set(repeat_overrides) - set(conditions)
    condition_repeats = {
        condition: tuple(int(value) for value in repeat_overrides.get(condition, repeats))
        for condition in conditions
    }
    if (
        not objects
        or not conditions
        or not repeats
        or not models
        or len(conditions) != len(set(conditions))
        or len(repeats) != len(set(repeats))
        or len(models) != len(set(models))
        or unknown_repeat_conditions
        or any(
            not values
            or len(values) != len(set(values))
            or any(repeat not in seed_lookup for repeat in values)
            for values in condition_repeats.values()
        )
    ):
        raise ReliabilityScoringError(
            "expected objects, conditions, repeats, and models must be unique and nonempty"
        )
    expected = {
        (object_id, condition, repeat, model)
        for object_id in objects
        for condition in conditions
        for repeat in condition_repeats[condition]
        for model in models
    }
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    eligibility: dict[tuple[str, str, str], bool] = {}
    for source in rows:
        row = dict(source)
        object_id = _as_nonempty_string(row, "object_id")
        condition = _as_nonempty_string(row, "condition")
        fold = _as_int(row, "fold")
        repeat = _as_int(row, "repeat")
        seed = _as_int(row, "seed")
        model = row.get("model_id", "k3")
        if not isinstance(model, str) or not model:
            raise ReliabilityScoringError("model_id must be a nonempty string")
        status = _as_nonempty_string(row, "status")
        key = (object_id, condition, repeat, model)
        if key not in expected:
            raise ReliabilityScoringError(f"unexpected reliability row: {key!r}")
        if key in seen:
            raise ReliabilityScoringError(f"duplicate reliability row: {key!r}")
        seen.add(key)
        if fold != objects[object_id]:
            raise ReliabilityScoringError(f"fold mismatch for {object_id}")
        if seed != seed_lookup[repeat]:
            raise ReliabilityScoringError(f"sampling seed mismatch for repeat {repeat}")
        if status not in VALID_STATUSES:
            raise ReliabilityScoringError(f"unknown reliability status: {status}")
        cell = (object_id, condition, model)
        eligible = status != "ineligible"
        if cell in eligibility and eligibility[cell] != eligible:
            raise ReliabilityScoringError("eligibility must not vary across repeats")
        eligibility[cell] = eligible

        error = row.get("error_deg")
        axes = row.get("axes")
        if status == "ok":
            error = _finite_number(error, key="error_deg")
            if not 0 <= error <= FAILURE_ERROR_DEG:
                raise ReliabilityScoringError("error_deg must lie in [0,90]")
            _validate_axes(axes)
        elif status == "failed":
            if error is not None and _finite_number(error, key="error_deg") != FAILURE_ERROR_DEG:
                raise ReliabilityScoringError("failed rows may omit error_deg or set it to 90")
            if not _axes_are_absent(axes):
                raise ReliabilityScoringError("failed rows must not contain candidate axes")
            error = FAILURE_ERROR_DEG
        else:
            if error is not None or not _axes_are_absent(axes):
                raise ReliabilityScoringError("ineligible rows must not contain axes or error_deg")
            error = None

        normalized_row = dict(row)
        normalized_row.update(
            {
                "object_id": object_id,
                "fold": fold,
                "condition": condition,
                "repeat": repeat,
                "seed": seed,
                "model_id": model,
                "status": status,
                "error_deg": error,
                "axes": axes,
            }
        )
        for metric in ("withheld_normalized_curve_rms", "training_rms"):
            if normalized_row.get(metric) is not None:
                if status == "ineligible":
                    raise ReliabilityScoringError(f"ineligible rows must not contain {metric}")
                value = _finite_number(normalized_row[metric], key=metric)
                if value < 0:
                    raise ReliabilityScoringError(f"{metric} must be nonnegative")
                normalized_row[metric] = value
        normalized.append(normalized_row)
    missing = expected - seen
    if missing:
        example = min(missing)
        raise ReliabilityScoringError(
            f"incomplete frozen reliability rows: {len(missing)} missing; first={example!r}"
        )
    return sorted(
        normalized,
        key=lambda row: (
            row["model_id"],
            row["condition"],
            row["fold"],
            row["object_id"],
            row["repeat"],
        ),
    )


def aggregate_object_repeats(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Average repeats within each asteroid before population summaries.

    Angular failures have already been assigned 90 degrees by validation.  The
    within-20 result is the mean of repeat-level indicators, not an indicator
    applied after averaging the errors.  Optional fit metrics are averaged only
    over rows that contain them, with their own denominators reported.
    """

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row.get("model_id", "k3")), str(row["condition"]), str(row["object_id"]))
        ].append(row)
    result: list[dict[str, Any]] = []
    for (model, condition, object_id), group in sorted(grouped.items()):
        folds = {int(row["fold"]) for row in group}
        statuses = [str(row["status"]) for row in group]
        if len(folds) != 1:
            raise ReliabilityScoringError(f"fold varies within object {object_id}")
        eligible = [row for row in group if row["status"] != "ineligible"]
        if eligible and len(eligible) != len(group):
            raise ReliabilityScoringError("eligibility must not vary across repeats")
        errors = [float(row["error_deg"]) for row in eligible]
        object_row: dict[str, Any] = {
            "object_id": object_id,
            "fold": folds.pop(),
            "condition": condition,
            "model_id": model,
            "eligible": bool(eligible),
            "n_repeats": len(group),
            "n_ok": statuses.count("ok"),
            "n_failed": statuses.count("failed"),
            "n_ineligible": statuses.count("ineligible"),
            "failure_rate": statuses.count("failed") / len(eligible) if eligible else None,
            "mean_error_deg": float(np.mean(errors)) if errors else None,
            "within_20_fraction": float(np.mean(np.asarray(errors) <= 20.0)) if errors else None,
        }
        for metric in ("withheld_normalized_curve_rms", "training_rms"):
            values = [float(row[metric]) for row in eligible if row.get(metric) is not None]
            object_row[metric] = float(np.mean(values)) if values else None
            object_row[f"n_{metric}"] = len(values)
        result.append(object_row)
    return result


def stratified_asteroid_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    statistic: str | Callable[[np.ndarray], float] = "mean",
    resamples: int = 10_000,
    seed: int = 20260915,
) -> dict[str, Any]:
    """Bootstrap unique asteroids within their frozen outer folds."""

    if resamples <= 0:
        raise ReliabilityScoringError("bootstrap resamples must be positive")
    identities = [str(row["object_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ReliabilityScoringError("bootstrap input must contain one row per asteroid")
    if not rows:
        raise ReliabilityScoringError("bootstrap input must not be empty")
    values = np.asarray([_finite_number(row.get(value_key), key=value_key) for row in rows])
    folds = np.asarray([_as_int(row, "fold") for row in rows])
    if statistic == "mean":

        def function(sample: np.ndarray) -> float:
            return float(np.mean(sample))

        statistic_name = "mean"
    elif statistic == "median":

        def function(sample: np.ndarray) -> float:
            return float(np.median(sample))

        statistic_name = "median"
    elif callable(statistic):
        function = statistic
        statistic_name = getattr(statistic, "__name__", "callable")
    else:
        raise ReliabilityScoringError("statistic must be 'mean', 'median', or callable")
    strata = [np.flatnonzero(folds == fold) for fold in sorted(set(folds.tolist()))]
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = np.concatenate(
            [indices[rng.integers(0, len(indices), size=len(indices))] for indices in strata]
        )
        estimates[index] = function(values[sampled])
    point = function(values)
    if not math.isfinite(point) or not np.all(np.isfinite(estimates)):
        raise ReliabilityScoringError("bootstrap statistic returned a non-finite value")
    return {
        "statistic": statistic_name,
        "estimate": point,
        "bootstrap_lower_95": float(np.quantile(estimates, 0.025)),
        "bootstrap_upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "n_objects": len(rows),
        "n_folds": len(strata),
        "resampling": "asteroids_within_frozen_outer_fold",
    }


def matched_condition_comparison(
    object_rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    baseline_condition: str,
    value_key: str = "mean_error_deg",
    favorable: str = "lower",
    resamples: int = 10_000,
    seed: int = 20260915,
) -> dict[str, Any]:
    """Compare a condition with its declared baseline on matched asteroids.

    ``degradation`` is positive when the condition is worse: condition minus
    baseline for lower-is-better metrics, and baseline minus condition for
    higher-is-better metrics.
    """

    if condition == baseline_condition:
        raise ReliabilityScoringError("condition and baseline_condition must differ")
    if favorable not in {"lower", "higher"}:
        raise ReliabilityScoringError("favorable must be 'lower' or 'higher'")
    models = {str(row.get("model_id", "k3")) for row in object_rows}
    if len(models) != 1:
        raise ReliabilityScoringError("matched comparison accepts exactly one model")
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in object_rows:
        key = (str(row["condition"]), str(row["object_id"]))
        if key in lookup:
            raise ReliabilityScoringError(f"duplicate object-condition aggregate: {key!r}")
        lookup[key] = row
    target_ids = {object_id for row_condition, object_id in lookup if row_condition == condition}
    baseline_ids = {
        object_id for row_condition, object_id in lookup if row_condition == baseline_condition
    }
    target_eligible = {
        object_id
        for object_id in target_ids
        if bool(lookup[(condition, object_id)].get("eligible"))
    }
    baseline_eligible = {
        object_id
        for object_id in baseline_ids
        if bool(lookup[(baseline_condition, object_id)].get("eligible"))
    }
    target_metric = {
        object_id
        for object_id in target_eligible
        if lookup[(condition, object_id)].get(value_key) is not None
    }
    baseline_metric = {
        object_id
        for object_id in baseline_eligible
        if lookup[(baseline_condition, object_id)].get(value_key) is not None
    }
    matched = sorted(target_metric & baseline_metric)
    paired_rows = []
    for object_id in matched:
        target = lookup[(condition, object_id)]
        baseline = lookup[(baseline_condition, object_id)]
        if int(target["fold"]) != int(baseline["fold"]):
            raise ReliabilityScoringError(f"matched fold mismatch for {object_id}")
        target_value = _finite_number(target.get(value_key), key=value_key)
        baseline_value = _finite_number(baseline.get(value_key), key=value_key)
        raw_delta = target_value - baseline_value
        degradation = raw_delta if favorable == "lower" else -raw_delta
        paired_rows.append(
            {"object_id": object_id, "fold": int(target["fold"]), "degradation": degradation}
        )
    if not paired_rows:
        raise ReliabilityScoringError("matched comparison has no jointly eligible asteroids")
    interval = stratified_asteroid_bootstrap(
        paired_rows,
        value_key="degradation",
        resamples=resamples,
        seed=seed,
    )
    return {
        "condition": condition,
        "baseline_condition": baseline_condition,
        "metric": value_key,
        "favorable": favorable,
        "mean_degradation": interval["estimate"],
        "bootstrap_lower_95": interval["bootstrap_lower_95"],
        "bootstrap_upper_95": interval["bootstrap_upper_95"],
        "n_condition_eligible": len(target_eligible),
        "n_baseline_eligible": len(baseline_eligible),
        "n_condition_with_metric": len(target_metric),
        "n_baseline_with_metric": len(baseline_metric),
        "n_matched": len(matched),
        "n_unmatched_condition": len(target_metric - baseline_metric),
        "n_unmatched_baseline": len(baseline_metric - target_metric),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "resampling": interval["resampling"],
    }


def score_reliability(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_objects: Mapping[str, int],
    expected_conditions: Sequence[str],
    expected_repeats: Sequence[int] = (0, 1, 2),
    expected_repeats_by_condition: Mapping[str, Sequence[int]] | None = None,
    expected_models: Sequence[str] = ("k3",),
    sampling_seed_by_repeat: Mapping[int, int] = SAMPLING_SEED_BY_REPEAT,
    baseline_by_condition: Mapping[str, str] | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260915,
) -> dict[str, Any]:
    """Return a JSON-serializable denominator-complete reliability summary."""

    validated = validate_reliability_rows(
        rows,
        expected_objects=expected_objects,
        expected_conditions=expected_conditions,
        expected_repeats=expected_repeats,
        expected_repeats_by_condition=expected_repeats_by_condition,
        expected_models=expected_models,
        sampling_seed_by_repeat=sampling_seed_by_repeat,
    )
    objects = aggregate_object_repeats(validated)
    summaries: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for model in expected_models:
        model_rows = [row for row in objects if row["model_id"] == model]
        model_summary: dict[str, Any] = {}
        for condition in expected_conditions:
            raw = [
                row
                for row in validated
                if row["model_id"] == model and row["condition"] == condition
            ]
            aggregated = [row for row in model_rows if row["condition"] == condition]
            eligible = [row for row in aggregated if row["eligible"]]
            n_attempted = sum(row["status"] != "ineligible" for row in raw)
            n_failed = sum(row["status"] == "failed" for row in raw)
            summary: dict[str, Any] = {
                "n_scheduled_objects": len(aggregated),
                "n_eligible_objects": len(eligible),
                "n_ineligible_objects": len(aggregated) - len(eligible),
                "n_attempted_repeats": n_attempted,
                "n_successful_repeats": sum(row["status"] == "ok" for row in raw),
                "n_failed_repeats": n_failed,
                "failure_rate": n_failed / n_attempted if n_attempted else None,
                "mean_error_deg": float(np.mean([row["mean_error_deg"] for row in eligible]))
                if eligible
                else None,
                "median_object_repeat_mean_error_deg": float(
                    np.median([row["mean_error_deg"] for row in eligible])
                )
                if eligible
                else None,
                "mean_repeat_within_20_fraction": float(
                    np.mean([row["within_20_fraction"] for row in eligible])
                )
                if eligible
                else None,
            }
            for metric in ("withheld_normalized_curve_rms", "training_rms"):
                metric_rows = [row for row in eligible if row[metric] is not None]
                summary[f"mean_object_{metric}"] = (
                    float(np.mean([row[metric] for row in metric_rows])) if metric_rows else None
                )
                summary[f"n_objects_with_{metric}"] = len(metric_rows)
            if eligible:
                interval = stratified_asteroid_bootstrap(
                    eligible,
                    value_key="mean_error_deg",
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                )
                summary["mean_error_bootstrap_lower_95"] = interval["bootstrap_lower_95"]
                summary["mean_error_bootstrap_upper_95"] = interval["bootstrap_upper_95"]
            else:
                summary["mean_error_bootstrap_lower_95"] = None
                summary["mean_error_bootstrap_upper_95"] = None
            model_summary[condition] = summary
        summaries[model] = model_summary
        if baseline_by_condition:
            comparisons[model] = {}
            for condition, baseline in baseline_by_condition.items():
                comparisons[model][condition] = matched_condition_comparison(
                    model_rows,
                    condition=condition,
                    baseline_condition=baseline,
                    value_key="mean_error_deg",
                    favorable="lower",
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                )
    return {
        "schema": "delphi.k3-reliability-score.v1",
        "failure_error_deg": FAILURE_ERROR_DEG,
        "repeat_aggregation": "within_asteroid_before_population_summary",
        "within_20_aggregation": "mean_of_repeat_indicators",
        "median_aggregation": "median_of_asteroid_repeat_mean_errors",
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "resampling": "asteroids_within_frozen_outer_fold",
        },
        "models": summaries,
        "matched_comparisons": comparisons,
    }


def reliability_table_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten score output into stable condition rows for CSV/TeX tables."""

    fields = (
        "n_scheduled_objects",
        "n_eligible_objects",
        "n_ineligible_objects",
        "n_attempted_repeats",
        "n_successful_repeats",
        "n_failed_repeats",
        "failure_rate",
        "mean_error_deg",
        "mean_error_bootstrap_lower_95",
        "mean_error_bootstrap_upper_95",
        "median_object_repeat_mean_error_deg",
        "mean_repeat_within_20_fraction",
        "mean_object_withheld_normalized_curve_rms",
        "n_objects_with_withheld_normalized_curve_rms",
    )
    rows = []
    comparisons = summary.get("matched_comparisons", {})
    for model, conditions in sorted(summary["models"].items()):
        for condition, values in sorted(conditions.items()):
            comparison = comparisons.get(model, {}).get(condition, {})
            rows.append(
                {
                    "model_id": model,
                    "condition": condition,
                    **{key: values.get(key) for key in fields},
                    "baseline_condition": comparison.get("baseline_condition"),
                    "n_matched": comparison.get("n_matched"),
                    "mean_degradation": comparison.get("mean_degradation"),
                    "degradation_bootstrap_lower_95": comparison.get("bootstrap_lower_95"),
                    "degradation_bootstrap_upper_95": comparison.get("bootstrap_upper_95"),
                }
            )
    return rows


def render_reliability_csv(summary: Mapping[str, Any]) -> str:
    """Render a denominator-explicit condition table as RFC-compatible CSV."""

    rows = reliability_table_rows(summary)
    if not rows:
        raise ReliabilityScoringError("summary contains no reliability table rows")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _tex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def render_reliability_tex(summary: Mapping[str, Any]) -> str:
    """Render a compact LaTeX tabular with escaped labels and denominators."""

    rows = reliability_table_rows(summary)
    if not rows:
        raise ReliabilityScoringError("summary contains no reliability table rows")
    lines = [
        r"\begin{tabular}{llrrrrrrlrr}",
        r"Model & Condition & Eligible & Attempted & Failed & Fail. rate & Mean error & Median error & Baseline & Matched & Degradation \\",
        r"\hline",
    ]
    for row in rows:
        values = (
            row["model_id"],
            row["condition"],
            row["n_eligible_objects"],
            row["n_attempted_repeats"],
            row["n_failed_repeats"],
            row["failure_rate"],
            row["mean_error_deg"],
            row["median_object_repeat_mean_error_deg"],
            row["baseline_condition"],
            row["n_matched"],
            row["mean_degradation"],
        )
        lines.append(" & ".join(_tex_escape(value) for value in values) + r" \\")
    lines.extend((r"\end{tabular}", ""))
    return "\n".join(lines)
