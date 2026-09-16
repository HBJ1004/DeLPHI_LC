"""Independent checks for the native convexinv pole-angle interpretation.

These tests deliberately do not import the implementation's vector formula.  The
formula below is derived from ``blmatrix.c``: the third row is
``(sin(beta)*cos(lambda), sin(beta)*sin(lambda), cos(beta))`` and the solver
sets ``beta = 90 - printed_latitude`` before converting to radians.

The decoder API is intentionally small: ``decode_convergence_axis`` accepts the
two printed angles in degrees and returns an object with ``raw_lambda_deg``,
``raw_beta_deg``, ``standard_lambda_deg``, ``standard_beta_deg``,
``directed_unit_vector``, and ``interpretation_version`` attributes.  A mapping
with these keys is also
accepted so the test does not prescribe a particular result container.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import numpy as np
import pytest

from lc_pipeline.k3.convergence_axis import decode_convergence_axis


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _decoded(longitude: float, latitude: float) -> object:
    return decode_convergence_axis(raw_lambda_deg=longitude, raw_beta_deg=latitude)


def _native_vector(longitude: float, latitude: float) -> np.ndarray:
    """Independent translation of the native C rotation-matrix third row."""
    lam = math.radians(longitude)
    beta = math.radians(90.0 - latitude)
    return np.asarray(
        [
            math.sin(beta) * math.cos(lam),
            math.sin(beta) * math.sin(lam),
            math.cos(beta),
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    ("raw_longitude", "raw_latitude", "expected_longitude", "expected_latitude"),
    [
        (10.0, 100.0, 190.0, 80.0),
        (10.0, -100.0, 190.0, -80.0),
        (725.0, 0.0, 5.0, 0.0),
        (92.878146, 101.510344, 272.878146, 78.489656),
        (-5.0, 20.0, 355.0, 20.0),
    ],
)
def test_directed_vector_and_standard_coordinates_are_preserved(
    raw_longitude: float,
    raw_latitude: float,
    expected_longitude: float,
    expected_latitude: float,
):
    decoded = _decoded(raw_longitude, raw_latitude)
    vector = np.asarray(_field(decoded, "directed_unit_vector"), dtype=np.float64)

    np.testing.assert_allclose(vector, _native_vector(raw_longitude, raw_latitude), atol=1e-12)
    np.testing.assert_allclose(
        vector,
        _native_vector(expected_longitude, expected_latitude),
        atol=1e-12,
    )
    assert _field(decoded, "raw_lambda_deg") == raw_longitude
    assert _field(decoded, "raw_beta_deg") == raw_latitude
    assert _field(decoded, "standard_lambda_deg") == pytest.approx(expected_longitude)
    assert _field(decoded, "standard_beta_deg") == pytest.approx(expected_latitude)
    assert 0.0 <= float(_field(decoded, "standard_lambda_deg")) < 360.0
    assert -90.0 <= float(_field(decoded, "standard_beta_deg")) <= 90.0
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("longitude", [0.0, 360.0, -360.0, 1440.0])
@pytest.mark.parametrize("latitude", [-90.0, 90.0])
def test_poles_compare_vectors_not_longitude(longitude: float, latitude: float):
    decoded = _decoded(longitude, latitude)
    vector = np.asarray(_field(decoded, "directed_unit_vector"), dtype=np.float64)
    np.testing.assert_allclose(vector, _native_vector(longitude, latitude), atol=1e-12)
    assert -90.0 <= float(_field(decoded, "standard_beta_deg")) <= 90.0


def test_random_full_turn_angles_match_independent_native_matrix():
    rng = np.random.default_rng(20260911)
    for longitude, latitude in rng.uniform(
        low=(-8.0 * 360.0, -8.0 * 360.0),
        high=(8.0 * 360.0, 8.0 * 360.0),
        size=(128, 2),
    ):
        decoded = _decoded(float(longitude), float(latitude))
        vector = np.asarray(_field(decoded, "directed_unit_vector"), dtype=np.float64)
        np.testing.assert_allclose(vector, _native_vector(longitude, latitude), atol=1e-12)


@pytest.mark.parametrize(
    ("longitude", "latitude"),
    [
        (float("nan"), 0.0),
        (0.0, float("inf")),
        ("10", 0.0),
        (0.0, "10"),
        (True, 0.0),
        (0.0, False),
    ],
)
def test_nonfinite_non_numeric_and_boolean_angles_are_rejected(longitude, latitude):
    with pytest.raises((TypeError, ValueError)):
        _decoded(longitude, latitude)


def test_decoder_does_not_mutate_a_record_containing_raw_angles():
    record = {
        "final_lambda_deg": 92.878146,
        "final_beta_deg": 101.510344,
        "iterations": 17,
        "relative_rms_from_output": 0.2,
    }
    before = copy.deepcopy(record)
    decoded = _decoded(record["final_lambda_deg"], record["final_beta_deg"])
    assert record == before
    assert _field(decoded, "interpretation_version")
