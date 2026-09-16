"""Contracts for the independent retrospective known-period grid scorer."""

from __future__ import annotations

import copy
import json

import pytest

from lc_pipeline.k3.grid_benchmark_scoring import (
    ARMS,
    TIMING_MODES,
    GridBenchmarkScoringError,
    score_rows,
)


def _grid(object_count: int = 5) -> tuple[list[dict], dict[str, list[list[float]]]]:
    rows = []
    references = {}
    for index in range(object_count):
        object_id = f"object-{index:03d}"
        references[object_id] = [[1.0, 0.0, 0.0]]
        for arm in ARMS:
            for mode in TIMING_MODES:
                for repeat in range(3):
                    classical = arm.startswith("classical")
                    guided = arm.startswith("guided")
                    seconds = 2.0 if classical else (1.0 if guided else 1.5)
                    rms = 1.0 if classical else (0.9 if guided else 0.95)
                    rows.append(
                        {
                            "object_id": object_id,
                            "fold": index % 5,
                            "repeat_index": repeat,
                            "arm": arm,
                            "timing_mode": mode,
                            "wall_seconds": seconds,
                            "completed": True,
                            "selected_fit": {
                                "axis": [1.0, 0.0, 0.0],
                                "final_relative_rms": rms,
                                "start_index": repeat,
                            },
                        }
                    )
    return rows, references


def _cell(rows: list[dict], object_id: str, arm: str, mode: str, repeat: int) -> dict:
    return next(
        row
        for row in rows
        if (row["object_id"], row["arm"], row["timing_mode"], row["repeat_index"])
        == (object_id, arm, mode, repeat)
    )


def test_antipodal_direction_sign_has_zero_error_and_json_is_strict() -> None:
    rows, references = _grid()
    references = {object_id: [[-3.0, 0.0, 0.0]] for object_id in references}

    result = score_rows(rows, references, resamples=17)

    error = result["selected_axial_errors_by_object"]["guided20"]["cold"][0]
    assert error["repeat_error_degrees"] == [0.0, 0.0, 0.0]
    json.dumps(result, allow_nan=False)


def test_reference_label_intervention_changes_only_axis_scoring() -> None:
    rows, references = _grid()
    changed = {object_id: [[0.0, 1.0, 0.0]] for object_id in references}

    x_score = score_rows(rows, references, resamples=19)
    y_score = score_rows(rows, changed, resamples=19)
    x_primary = x_score["analyses"]["cold_step20_primary"]
    y_primary = y_score["analyses"]["cold_step20_primary"]

    assert x_primary["runtime"] == y_primary["runtime"]
    assert x_primary["completion"] == y_primary["completion"]
    assert x_primary["rms"] == y_primary["rms"]
    assert x_primary["recovery"]["baseline_success_count"] == 5
    assert y_primary["recovery"]["baseline_success_count"] == 0


def test_one_failed_repeat_remains_in_denominator_but_majority_succeeds() -> None:
    rows, references = _grid()
    for object_id in references:
        for arm in ARMS:
            for mode in TIMING_MODES:
                row = _cell(rows, object_id, arm, mode, 0)
                row["completed"] = False
                row["selected_fit"] = None

    result = score_rows(rows, references, resamples=19)
    summary = result["arm_summaries"]["guided20"]["cold"]

    assert summary["completion"] == {
        "repeat_success_count": 10,
        "repeat_denominator": 15,
        "repeat_rate": pytest.approx(2 / 3),
        "object_success_count": 5,
        "object_denominator": 5,
        "object_rate": 1.0,
    }
    assert summary["recovery_within_20_degrees"]["repeat_denominator_including_failures"] == 15
    assert summary["recovery_within_20_degrees"]["object_success_count"] == 5
    assert result["analyses"]["cold_step20_primary"]["rms"]["object_support"] == 5


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "duplicate execution cell"),
        ("missing", "complete Cartesian grid"),
        ("fold", "inconsistent folds"),
        ("failed_with_fit", "completed and selected_fit"),
        ("completed_without_fit", "completed and selected_fit"),
        ("nonunit_axis", "unit vector"),
    ],
)
def test_duplicate_missing_and_tampered_rows_are_rejected(mutation: str, match: str) -> None:
    rows, references = _grid()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "fold":
        rows[1]["fold"] = 4 if rows[1]["fold"] != 4 else 3
    elif mutation == "failed_with_fit":
        rows[0]["completed"] = False
    elif mutation == "completed_without_fit":
        rows[0]["selected_fit"] = None
    else:
        rows[0]["selected_fit"]["axis"] = [2.0, 0.0, 0.0]

    with pytest.raises(GridBenchmarkScoringError, match=match):
        score_rows(rows, references, resamples=5)


def test_rms_absent_full_support_is_not_estimable_and_fails_closed() -> None:
    rows, references = _grid()
    for repeat in (0, 1):
        row = _cell(rows, "object-004", "classical20", "cold", repeat)
        row["completed"] = False
        row["selected_fit"] = None

    result = score_rows(rows, references, resamples=23)
    primary = result["analyses"]["cold_step20_primary"]

    assert primary["rms"]["object_support"] == 4
    assert primary["rms"]["required_object_support"] == 5
    assert primary["rms"]["estimable"] is False
    assert primary["rms"]["point_ratio"] is None
    assert "rms_full_predeclared_object_support" in primary["decision"]["failed_gates"]


def test_full_cohort_simple_favorable_case_passes_primary() -> None:
    rows, references = _grid(170)

    result = score_rows(rows, references, resamples=31)
    primary = result["analyses"]["cold_step20_primary"]

    assert primary["runtime"]["point_ratio"] == pytest.approx(2.0)
    assert primary["runtime"]["percentile_interval_95"] == {
        "lower": pytest.approx(2.0),
        "upper": pytest.approx(2.0),
    }
    assert primary["rms"]["point_ratio"] == pytest.approx(0.9)
    assert primary["decision"]["passed_all_four_conditions"] is True
    assert result["decision"]["primary_verdict"] == "pass"
    assert result["standard6_descriptive"]["positive_hypothesis_test"] is False


def test_full_cohort_simple_unfavorable_case_fails_primary() -> None:
    rows, references = _grid(170)
    for row in rows:
        if row["arm"] == "guided20" and row["timing_mode"] == "cold":
            row["wall_seconds"] = 3.0
            row["selected_fit"]["axis"] = [0.0, 1.0, 0.0]
            row["selected_fit"]["final_relative_rms"] = 1.2

    result = score_rows(rows, references, resamples=31)
    primary = result["analyses"]["cold_step20_primary"]

    assert primary["runtime"]["point_ratio"] == pytest.approx(2 / 3)
    assert primary["recovery"]["point_difference"] == -1.0
    assert primary["rms"]["point_ratio"] == pytest.approx(1.2)
    assert primary["decision"]["passed_all_four_conditions"] is False
    assert len(primary["decision"]["failed_gates"]) == 3
    assert result["decision"]["primary_verdict"] == "fail"
