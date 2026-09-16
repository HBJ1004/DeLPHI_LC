"""Independent checks for capacities consumed by the native DAMIT solvers.

The C programs do not use only the input framing.  Both ``convexinv.c`` and
``period_scan.c`` append one synthetic three-point convexity lightcurve before
fitting, so their safe limits are input observations plus three, input
lightcurves plus one, and at least three points per lightcurve.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lc_pipeline.k3.solver_capacity import (
    LightcurveStructure,
    SolverCapacityError,
    capacity_violations,
    parse_lightcurve_structure,
    required_capacities,
)


def test_required_capacities_include_native_convexity_rows():
    structures = [
        LightcurveStructure(lightcurve_count=2, total_observations=6, max_points_per_lightcurve=1),
        LightcurveStructure(lightcurve_count=209, total_observations=26609, max_points_per_lightcurve=1549),
    ]
    assert required_capacities(structures) == {
        "POINTS_MAX": 1549,
        "MAX_N_OBS": 26612,
        "MAX_LC": 210,
    }


@pytest.mark.parametrize(
    ("name", "decrement"),
    [("POINTS_MAX", 1), ("MAX_N_OBS", 1), ("MAX_LC", 1)],
)
def test_capacity_violations_reject_one_below_safe_native_boundary(name: str, decrement: int):
    structure = LightcurveStructure(209, 26609, 1549)
    capacities = required_capacities([structure])
    capacities[name] -= decrement
    assert name in capacity_violations(structure, capacities)


def test_exact_safe_boundary_accepts_maximum_cohort_structure():
    structure = LightcurveStructure(209, 26609, 1549)
    capacities = required_capacities([structure])
    assert capacity_violations(structure, capacities) == ()


def test_small_input_with_two_points_still_requires_three_point_synthetic_row():
    structure = LightcurveStructure(1, 2, 2)
    assert required_capacities([structure]) == {
        "POINTS_MAX": 3,
        "MAX_N_OBS": 5,
        "MAX_LC": 2,
    }


def test_parser_keeps_input_counts_separate_from_native_extra_rows(tmp_path: Path):
    path = tmp_path / "lc.txt"
    path.write_text(
        "2\n"
        "2 1\n"
        "1 0\n"
        "2 0\n"
        "3 1\n"
        "4 0\n"
        "5 0\n"
        "6 0\n",
        encoding="ascii",
    )
    parsed = parse_lightcurve_structure(path)
    assert parsed == LightcurveStructure(2, 5, 3)
    assert required_capacities([parsed]) == {
        "POINTS_MAX": 3,
        "MAX_N_OBS": 8,
        "MAX_LC": 3,
    }


def test_parser_rejects_empty_or_truncated_lightcurve(tmp_path: Path):
    empty = tmp_path / "empty.lc"
    empty.write_text("\n", encoding="ascii")
    with pytest.raises(SolverCapacityError, match="empty"):
        parse_lightcurve_structure(empty)

    truncated = tmp_path / "truncated.lc"
    truncated.write_text("1\n3 1\n0 0\n", encoding="ascii")
    with pytest.raises(SolverCapacityError, match="truncated"):
        parse_lightcurve_structure(truncated)
