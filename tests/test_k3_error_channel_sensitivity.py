from __future__ import annotations

import numpy as np

from repro.run_k3_error_channel_sensitivity import (
    _descriptive,
    _paired_interval,
    remove_measured_errors,
)


def test_remove_measured_errors_changes_no_other_observation_fields() -> None:
    source = {
        "object_id": "asteroid_1",
        "epochs": [
            {
                "epoch_id": "e0",
                "observations": [
                    {"time_jd": 1.0, "relative_brightness": 2.0, "measured_error": 0.1},
                    {"time_jd": 2.0, "relative_brightness": 3.0},
                ],
            }
        ],
    }
    result, observations, removed_fields, removed_values = remove_measured_errors(source)
    assert observations == 2
    assert removed_fields == 1
    assert removed_values == 1
    assert "measured_error" not in result["epochs"][0]["observations"][0]
    assert result["epochs"][0]["observations"][0]["relative_brightness"] == 2.0
    assert source["epochs"][0]["observations"][0]["measured_error"] == 0.1


def test_paired_interval_reports_the_mean_direction() -> None:
    result = _paired_interval(np.asarray([1.0, 2.0, 3.0]))
    assert result["mean_difference_deg"] == 2.0
    assert result["ci95_low_deg"] <= 2.0 <= result["ci95_high_deg"]


def test_descriptive_uses_inclusive_thresholds() -> None:
    result = _descriptive(np.asarray([10.0, 20.0, 30.0, 40.0]))
    assert result["median_error_deg"] == 25.0
    assert result["within_20_fraction"] == 0.5
    assert result["within_30_fraction"] == 0.75
