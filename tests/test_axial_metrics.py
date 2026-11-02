"""Truth-table and gradient tests for the preregistered axial metric."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lc_pipeline.physics.axial import (
    axial_angular_error_deg,
    axial_angular_error_deg_scalar,
    oracle_source_axis_match,
    pairwise_axial_angular_errors_deg,
    torch_axial_angular_error_deg,
)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.0),
        ([1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], 0.0),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 90.0),
        ([1.0, 0.0, 0.0], [-1.0, -1.0, 0.0], 45.0),
    ],
)
def test_axial_angle_truth_table(first, second, expected):
    assert axial_angular_error_deg(first, second) == pytest.approx(expected, abs=1e-12)
    assert axial_angular_error_deg_scalar(first, second) == pytest.approx(expected, abs=1e-12)


def test_axial_metric_is_sign_invariant_and_preserves_tiny_angles():
    angle = np.deg2rad(1.0e-10)
    first = np.array([1.0, 0.0, 0.0])
    second = np.array([np.cos(angle), np.sin(angle), 0.0])
    direct = axial_angular_error_deg(first, second)
    reflected = axial_angular_error_deg(-first, second)
    assert direct == pytest.approx(1.0e-10, rel=1e-12)
    assert reflected == pytest.approx(direct, abs=1e-20)


def test_axial_pairwise_oracle_honors_multiple_explicit_solutions():
    predictions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    sources = [[-1.0, 0.0, 0.0], [0.0, np.sqrt(3) / 2, 0.5]]
    errors = pairwise_axial_angular_errors_deg(predictions, sources)
    np.testing.assert_allclose(errors, [[0.0, 90.0], [90.0, 30.0], [90.0, 60.0]], atol=1e-12)
    match = oracle_source_axis_match(predictions, sources)
    assert match.error_deg == pytest.approx(0.0)
    assert (match.prediction_index, match.source_axis_index) == (0, 0)


@pytest.mark.parametrize(
    ("first", "second", "error"),
    [
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], "zero vector"),
        ([1.0, 0.0], [1.0, 0.0, 0.0], "shape"),
        ([1.0, np.nan, 0.0], [1.0, 0.0, 0.0], "finite"),
    ],
)
def test_axial_metric_rejects_invalid_vectors(first, second, error):
    with pytest.raises(ValueError, match=error):
        axial_angular_error_deg(first, second)
    with pytest.raises(ValueError, match=error):
        axial_angular_error_deg_scalar(first, second)


def test_torch_axial_metric_is_float64_and_sign_invariant():
    first = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    second = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    result = torch_axial_angular_error_deg(first, second)
    torch.testing.assert_close(result, torch.tensor([0.0, 90.0], dtype=torch.float64))
    assert result.dtype == torch.float64
