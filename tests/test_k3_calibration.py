"""Finite-sample calibration safeguards for K3 axial search cones."""

from __future__ import annotations

import numpy as np
import pytest

from lc_pipeline.k3.calibration import (
    K3CalibrationError,
    conformal_radius,
    normalized_entropy,
)


def test_conformal_radius_uses_exact_finite_sample_rank() -> None:
    errors = np.arange(1.0, 21.0)
    # ceil((20 + 1) * 0.90) = 19
    assert conformal_radius(errors, 0.90) == 19.0
    # ceil((20 + 1) * 0.95) = 20
    assert conformal_radius(errors, 0.95) == 20.0


def test_fourteen_object_95_percent_radius_is_bounded_full_hemisphere() -> None:
    errors = np.arange(1.0, 15.0)
    assert conformal_radius(errors, 0.90) == 14.0
    # ceil(15 * .95) = 15: no observed order statistic supports the target.
    assert conformal_radius(errors, 0.95) == 90.0


@pytest.mark.parametrize(
    "errors,coverage",
    (([1.0], 0.9), ([1.0, np.nan], 0.9), ([-1.0, 2.0], 0.9), ([1.0, 2.0], 1.0)),
)
def test_conformal_radius_rejects_invalid_inputs(errors, coverage) -> None:
    with pytest.raises(K3CalibrationError):
        conformal_radius(errors, coverage)


def test_normalized_entropy_extremes() -> None:
    assert normalized_entropy([1.0, 0.0, 0.0]) == 0.0
    assert normalized_entropy([0.25] * 4) == pytest.approx(1.0)
