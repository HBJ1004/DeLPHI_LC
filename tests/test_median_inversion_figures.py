"""Portable checks for the inversion figure's time labels and censored costs."""

import pytest

from repro.own_period.median_figures import full_scan_costs, inversion_time_label


@pytest.mark.parametrize(("seconds", "label"), [
    (0, "0 s"), (2.618726634585445, "2.62 s"), (59, "59 s"), (60, "1 min"),
    (3600, "1 h"), (86400, "1 d"), (365.25 * 86400, "1 yr"),
])
def test_inversion_time_units(seconds, label):
    assert inversion_time_label(seconds) == label


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf")])
def test_inversion_time_rejects_invalid_durations(seconds):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        inversion_time_label(seconds)


def test_inversion_time_annotation_precision():
    assert inversion_time_label(5.793711620831358 * 3600, significant=2) == "5.8 h"
    assert inversion_time_label(72.32961673733482 * 86400, significant=2) == "72 d"


def test_full_scan_costs_keeps_bounds_and_excludes_validation_objects():
    def estimate(**cpu):
        return {"P_core": {"cpu_seconds": cpu}}

    summary = {
        "estimates": {
            "first": estimate(estimated_full=13 * 86400),
            "bounded_middle": estimate(estimated_full_at_least=70 * 86400),
            "middle": estimate(estimated_full=74 * 86400, estimated_full_at_least=1),
            "bounded_maximum": estimate(estimated_full_at_least=346 * 86400),
            "validation": estimate(estimated_full=1),
        },
        "full_scan_checks": {"validation": {}},
    }
    result = full_scan_costs(summary)
    assert result["n"] == 4
    assert result["object_ids"] == ["first", "bounded_middle", "middle", "bounded_maximum"]
    assert result["seconds"] == 72 * 86400
    assert result["range_seconds"] == [13 * 86400, 346 * 86400]
    assert result["range_upper_is_lower_bound"] is True
    assert [row["lower_bound"] for row in result["per_object"]] == [False, True, False, True]
