"""Truth-table and gradient tests for the preregistered axial metric."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lc_pipeline.physics.axial import (
    axial_angular_error_deg,
    axial_angular_error_deg_scalar,
    axial_oracle_training_loss,
    axial_separation_penalty,
    axial_sine_squared_loss,
    masked_axial_softmin_loss,
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


def test_axial_sine_squared_loss_has_no_angle_floor_and_finite_gradient():
    prediction = torch.tensor([[1.0, 1.0e-3, 0.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)
    loss = axial_sine_squared_loss(prediction, target)
    assert 0.0 < loss.item() < 2.0e-6
    loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert prediction.grad[0, 1].item() > 0.0


def test_masked_softmin_ignores_zero_padding_and_is_differentiable():
    predictions = torch.tensor(
        [[[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.1, 0.0, 1.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor(
        [[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64
    )
    mask = torch.tensor([[True, False]])
    loss = masked_axial_softmin_loss(predictions, targets, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert predictions.grad is not None
    assert bool(torch.isfinite(predictions.grad).all())


def test_separation_penalty_is_axis_aware():
    separated = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    collapsed = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    assert axial_separation_penalty(separated).item() == pytest.approx(0.0)
    assert axial_separation_penalty(collapsed).item() > 0.0


def test_combined_oracle_loss_reports_components():
    predictions = torch.tensor(
        [[[1.0, 0.01, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[[-1.0, 0.0, 0.0]]])
    total, components = axial_oracle_training_loss(
        predictions, targets, torch.tensor([[True]])
    )
    assert set(components) == {"axial_oracle_loss", "separation_penalty"}
    total.backward()
    assert predictions.grad is not None
