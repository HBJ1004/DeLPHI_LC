import csv
import io

import pytest

from lc_pipeline.k3.reliability_scoring import (
    ReliabilityScoringError,
    aggregate_object_repeats,
    matched_condition_comparison,
    render_reliability_csv,
    render_reliability_tex,
    score_reliability,
    stratified_asteroid_bootstrap,
    validate_reliability_rows,
    withheld_normalized_curve_rms,
)

AXES = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _rows():
    errors = {
        ("a", "full"): [10.0, 30.0, 50.0],
        ("a", "thin"): [20.0, 40.0, 60.0],
        ("b", "full"): [5.0, 15.0, 25.0],
        ("b", "thin"): [10.0, 20.0, 30.0],
    }
    rows = []
    for (object_id, condition), values in errors.items():
        fold = 0 if object_id == "a" else 1
        for repeat, error in enumerate(values):
            rows.append(
                {
                    "object_id": object_id,
                    "fold": fold,
                    "condition": condition,
                    "repeat": repeat,
                    "seed": 20260915 + repeat,
                    "status": "ok",
                    "axes": AXES,
                    "error_deg": error,
                    "withheld_normalized_curve_rms": error / 100,
                }
            )
    return rows


def test_withheld_curve_rms_weights_curves_not_points_and_does_not_refit():
    # Per-curve MSEs are 1 and 4, despite the unequal point counts.
    value = withheld_normalized_curve_rms([[0.0, 0.0], [0.0]], [[1.0, 1.0], [2.0]])
    assert value == pytest.approx((2.5) ** 0.5)
    with pytest.raises(ReliabilityScoringError, match="shape matched"):
        withheld_normalized_curve_rms([[0.0]], [[0.0, 1.0]])


def test_complete_contract_defaults_model_and_exposes_failures_at_90_degrees():
    rows = _rows()
    rows[0] = {
        **rows[0],
        "status": "failed",
        "axes": None,
        "error_deg": None,
        "failure_code": "inference_error",
    }
    validated = validate_reliability_rows(
        rows,
        expected_objects={"a": 0, "b": 1},
        expected_conditions=("full", "thin"),
    )
    failed = next(row for row in validated if row["status"] == "failed")
    assert failed["model_id"] == "k3"
    assert failed["error_deg"] == 90.0
    objects = aggregate_object_repeats(validated)
    full_a = next(row for row in objects if row["object_id"] == "a" and row["condition"] == "full")
    assert full_a["n_failed"] == 1
    assert full_a["failure_rate"] == pytest.approx(1 / 3)
    assert full_a["mean_error_deg"] == pytest.approx((90 + 30 + 50) / 3)
    assert full_a["within_20_fraction"] == 0.0


def test_incomplete_duplicate_or_repeat_varying_eligibility_is_rejected():
    args = {
        "expected_objects": {"a": 0, "b": 1},
        "expected_conditions": ("full", "thin"),
    }
    with pytest.raises(ReliabilityScoringError, match="missing"):
        validate_reliability_rows(_rows()[:-1], **args)
    with pytest.raises(ReliabilityScoringError, match="duplicate"):
        validate_reliability_rows(_rows() + [_rows()[0]], **args)
    rows = _rows()
    rows[0] = {
        **rows[0],
        "status": "ineligible",
        "axes": None,
        "error_deg": None,
        "withheld_normalized_curve_rms": None,
    }
    with pytest.raises(ReliabilityScoringError, match="eligibility must not vary"):
        validate_reliability_rows(rows, **args)


def test_summary_uses_repeat_indicator_mean_and_object_repeat_mean_median():
    result = score_reliability(
        _rows(),
        expected_objects={"a": 0, "b": 1},
        expected_conditions=("full", "thin"),
        baseline_by_condition={"thin": "full"},
        bootstrap_resamples=100,
    )
    full = result["models"]["k3"]["full"]
    # Object repeat means: a=30, b=15.  Do not take the median of six rows.
    assert full["median_object_repeat_mean_error_deg"] == pytest.approx(22.5)
    # Repeat indicators: [1,0,0] and [1,1,0], averaged within objects first.
    assert full["mean_repeat_within_20_fraction"] == pytest.approx(0.5)
    assert full["n_attempted_repeats"] == 6
    comparison = result["matched_comparisons"]["k3"]["thin"]
    assert comparison["mean_degradation"] == pytest.approx(7.5)
    assert comparison["n_matched"] == 2


def test_condition_specific_repeat_schedule_does_not_duplicate_deterministic_baseline():
    rows = [row for row in _rows() if row["condition"] != "full" or row["repeat"] == 0]
    result = score_reliability(
        rows,
        expected_objects={"a": 0, "b": 1},
        expected_conditions=("full", "thin"),
        expected_repeats_by_condition={"full": (0,)},
        baseline_by_condition={"thin": "full"},
        bootstrap_resamples=20,
    )
    full = result["models"]["k3"]["full"]
    assert full["n_attempted_repeats"] == 2
    assert full["n_failed_repeats"] == 0
    assert result["matched_comparisons"]["k3"]["thin"]["n_matched"] == 2


def test_stratified_bootstrap_is_deterministic_and_requires_unique_asteroids():
    rows = [
        {"object_id": "a", "fold": 0, "value": 0.0},
        {"object_id": "b", "fold": 0, "value": 2.0},
        {"object_id": "c", "fold": 1, "value": 10.0},
    ]
    first = stratified_asteroid_bootstrap(rows, value_key="value", resamples=200, seed=7)
    second = stratified_asteroid_bootstrap(rows, value_key="value", resamples=200, seed=7)
    assert first == second
    assert first["estimate"] == pytest.approx(4.0)
    assert first["n_folds"] == 2
    with pytest.raises(ReliabilityScoringError, match="one row per asteroid"):
        stratified_asteroid_bootstrap(rows + [rows[0]], value_key="value", resamples=10)


def test_matched_comparison_reports_eligibility_denominators():
    rows = [
        {
            "object_id": "a",
            "fold": 0,
            "condition": "grid-full",
            "eligible": True,
            "mean_error_deg": 10,
        },
        {
            "object_id": "a",
            "fold": 0,
            "condition": "grid-1",
            "eligible": True,
            "mean_error_deg": 14,
        },
        {
            "object_id": "b",
            "fold": 1,
            "condition": "grid-full",
            "eligible": True,
            "mean_error_deg": 20,
        },
        {
            "object_id": "b",
            "fold": 1,
            "condition": "grid-1",
            "eligible": False,
            "mean_error_deg": None,
        },
    ]
    result = matched_condition_comparison(
        rows,
        condition="grid-1",
        baseline_condition="grid-full",
        resamples=20,
    )
    assert result["mean_degradation"] == 4.0
    assert result["n_condition_eligible"] == 1
    assert result["n_baseline_eligible"] == 2
    assert result["n_matched"] == 1
    assert result["n_unmatched_baseline"] == 1


def test_csv_and_tex_exports_preserve_denominators_and_escape_labels():
    rows = _rows()
    for row in rows:
        row["condition"] = "full&base" if row["condition"] == "full" else "thin_10%"
    summary = score_reliability(
        rows,
        expected_objects={"a": 0, "b": 1},
        expected_conditions=("full&base", "thin_10%"),
        bootstrap_resamples=10,
    )
    csv_rows = list(csv.DictReader(io.StringIO(render_reliability_csv(summary))))
    header = render_reliability_csv(summary).splitlines()[0].split(",")
    assert csv_rows[0]["n_attempted_repeats"] == "6"
    assert csv_rows[0]["n_failed_repeats"] == "0"
    assert "mean_error_bootstrap_lower_95" in header
    assert "mean_error_bootstrap_upper_95" in header
    tex = render_reliability_tex(summary)
    assert r"full\&base" in tex
    assert r"thin\_10\%" in tex
