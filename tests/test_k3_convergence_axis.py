"""Contracts for interpreting unconstrained convexinv solver directions."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from lc_pipeline.k3.convergence_axis import (
    AXIS_INTERPRETATION_VERSION,
    ConvergenceAxisError,
    decode_convergence_axis,
)
from lc_pipeline.k3.convergence_benchmark import _completion, _selectable
from lc_pipeline.k3.convergence_study import _selected_pole_error, _selection_from_records


def _native_blmatrix_direction(lambda_deg: float, printed_beta_deg: float) -> np.ndarray:
    """C ``blmatrix`` third row after convexinv prints ``90 - theta``."""
    theta = math.radians(90.0 - printed_beta_deg)
    longitude = math.radians(lambda_deg)
    return np.asarray(
        (
            math.sin(theta) * math.cos(longitude),
            math.sin(theta) * math.sin(longitude),
            math.cos(theta),
        ),
        dtype=np.float64,
    )


def _valid_result(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "timed_out": False,
        "adapter_error": None,
        "return_code": 0,
        "iterations": 12,
        "chi2": 1.0,
        "deviation": 0.01,
        "final_lambda_deg": 10.0,
        "final_beta_deg": 20.0,
        "final_period_hours": 8.0,
        "relative_rms_from_output": 0.02,
        "output_validation_error": None,
        "output_lightcurve_sha256": "a" * 64,
        "output_parameter_sha256": "b" * 64,
        "output_area_sha256": "c" * 64,
    }
    result.update(updates)
    return result


def _record(start_index: int, **result_updates: object) -> dict[str, object]:
    result = _valid_result(**result_updates)
    return {
        "identity": {"start_index": start_index},
        "completion": _completion(result, expected_period_hours=8.0),
        "result": result,
    }


@pytest.mark.parametrize(
    ("raw", "standard"),
    [
        ((-355.0, -20.0), (5.0, -20.0)),
        ((725.0, 0.0), (5.0, 0.0)),
        ((10.0, 100.0), (190.0, 80.0)),
        ((10.0, -100.0), (190.0, -80.0)),
    ],
)
def test_decoder_standardizes_wrapped_and_unconstrained_coordinates(raw, standard):
    decoded = decode_convergence_axis(*raw)

    assert decoded.raw_lambda_deg == raw[0]
    assert decoded.raw_beta_deg == raw[1]
    assert decoded.interpretation_version == AXIS_INTERPRETATION_VERSION
    assert decoded.standard_lambda_deg == pytest.approx(standard[0], abs=1e-12)
    assert decoded.standard_beta_deg == pytest.approx(standard[1], abs=1e-12)
    assert 0.0 <= decoded.standard_lambda_deg < 360.0
    assert -90.0 <= decoded.standard_beta_deg <= 90.0
    np.testing.assert_allclose(
        decoded.directed_unit_vector,
        _native_blmatrix_direction(*raw),
        rtol=0.0,
        atol=1e-12,
    )


def test_decoder_preserves_vectors_at_poles_and_full_rotations():
    north = decode_convergence_axis(123.0, 90.0)
    south = decode_convergence_axis(-987.0, -90.0)
    full_turn = decode_convergence_axis(123.0 + 720.0, 90.0 + 720.0)

    np.testing.assert_allclose(north.directed_unit_vector, (0.0, 0.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(south.directed_unit_vector, (0.0, 0.0, -1.0), atol=1e-12)
    np.testing.assert_allclose(full_turn.directed_unit_vector, north.directed_unit_vector, atol=1e-12)
    assert np.linalg.norm(north.directed_unit_vector) == pytest.approx(1.0, abs=1e-15)
    assert np.linalg.norm(south.directed_unit_vector) == pytest.approx(1.0, abs=1e-15)


def test_decoder_matches_real_out_of_range_solver_example():
    decoded = decode_convergence_axis(92.878146, 101.510344)

    assert decoded.standard_lambda_deg == pytest.approx(272.878146, abs=1e-12)
    assert decoded.standard_beta_deg == pytest.approx(78.489656, abs=1e-12)
    np.testing.assert_allclose(
        decoded.directed_unit_vector,
        _native_blmatrix_direction(92.878146, 101.510344),
        rtol=0.0,
        atol=1e-12,
    )


def test_decoder_matches_native_rotation_direction_for_fixed_seed_random_angles():
    generator = np.random.default_rng(20260911)
    for raw_lambda, raw_beta in generator.uniform(-4 * 360.0, 4 * 360.0, size=(257, 2)):
        decoded = decode_convergence_axis(raw_lambda, raw_beta)
        np.testing.assert_allclose(
            decoded.directed_unit_vector,
            _native_blmatrix_direction(raw_lambda, raw_beta),
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("longitude", "latitude"),
    [
        (float("nan"), 0.0),
        (0.0, float("inf")),
        ("10", 0.0),
        (0.0, "20"),
        (True, 0.0),
        (0.0, False),
        (None, 0.0),
    ],
)
def test_decoder_rejects_nonfinite_or_non_numeric_angles(longitude, latitude):
    with pytest.raises(ConvergenceAxisError, match="raw_.*(finite|real)"):
        decode_convergence_axis(longitude, latitude)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"timed_out": True}, "timeout"),
        ({"adapter_error": "adapter failed"}, "adapter-error"),
        ({"return_code": 1}, "exit-nonzero"),
        ({"iterations": None}, "missing-iteration-log"),
        ({"iterations": 1000}, "iteration-cap"),
        ({"relative_rms_from_output": 0.0}, "numerical-output-failure"),
        ({"final_period_hours": 8.1}, "numerical-output-failure"),
        ({"final_beta_deg": float("nan")}, "numerical-output-failure"),
        ({"final_lambda_deg": True}, "numerical-output-failure"),
        ({"output_validation_error": "bad product"}, "numerical-output-failure"),
        ({"output_area_sha256": "not-a-hash"}, "numerical-output-failure"),
    ],
)
def test_completion_keeps_every_other_failure_rejection(updates, expected):
    assert _completion(_valid_result(**updates), expected_period_hours=8.0) == expected


def test_out_of_range_latitude_is_converged_and_selection_uses_standard_axis_without_mutation():
    preferred = _record(1, final_lambda_deg=10.0, final_beta_deg=100.0, relative_rms_from_output=0.01)
    other = _record(0, final_lambda_deg=0.0, final_beta_deg=0.0, relative_rms_from_output=0.02)
    records = [preferred, other]
    before = json.dumps(records, sort_keys=True, allow_nan=False)

    assert preferred["completion"] == "converged"
    assert _selectable(preferred)
    selected = _selection_from_records(records)

    assert selected is not None
    assert selected["start_index"] == 1
    assert selected["final_lambda_deg"] == pytest.approx(190.0)
    assert selected["final_beta_deg"] == pytest.approx(80.0)
    assert selected["axis_interpretation_version"] == AXIS_INTERPRETATION_VERSION
    assert json.dumps(records, sort_keys=True, allow_nan=False) == before


def test_selection_and_scoring_are_label_blind_and_use_the_same_decoded_direction():
    record = _record(2, final_lambda_deg=10.0, final_beta_deg=-100.0)
    selected = _selection_from_records([copy.deepcopy(record)])
    assert selected is not None
    targets = np.asarray([decode_convergence_axis(190.0, -80.0).directed_unit_vector])

    assert _selected_pole_error(selected, targets) == pytest.approx(0.0, abs=1e-10)
    record["unrelated_reference_axis"] = [1.0, 0.0, 0.0]
    record["unrelated_label"] = "changed"
    assert _selection_from_records([record]) == selected
