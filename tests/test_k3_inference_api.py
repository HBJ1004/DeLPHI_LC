"""Grid, bounded refinement, and no-oracle K3 API tests."""

from __future__ import annotations

import inspect
import math
from copy import deepcopy

import numpy as np
import torch

from lc_pipeline.k3.api import DeLPHIK3Predictor, K3ModelBundle
from lc_pipeline.k3.calibration import K3Calibration
from lc_pipeline.k3.config import K3ScoreModelConfig
from lc_pipeline.k3.grid import axial_healpix_grid, extract_axial_modes
from lc_pipeline.k3.inference import (
    K3InferenceError,
    refine_axes,
    refine_ensemble_axes,
    score_axial_grid,
)
from lc_pipeline.k3.model import CandidateConditionedScorer
from lc_pipeline.k3.schema import K3PredictionStatus
from lc_pipeline.k3.tokenizer import pad_tokenized_objects, tokenize_epochs
from lc_pipeline.physics.axial import axial_angular_error_deg
from lc_pipeline.v2.preprocessing import KnownPeriod, Observation, ObservationEpoch


def _epochs() -> tuple[ObservationEpoch, ...]:
    observations = tuple(
        Observation(
            time_jd=2450000.0 + index / 48.0,
            relative_brightness=math.exp(0.1 * math.cos(index)),
            sun_asteroid_ecliptic_j2000_au=(1.0, 0.02 * index, 0.2),
            observer_asteroid_ecliptic_j2000_au=(0.2, 1.0, 0.01 * index + 0.1),
        )
        for index in range(12)
    )
    return (ObservationEpoch("source-epoch-0", observations),)


def _model_inputs(device: str = "cpu"):
    tokenized = tokenize_epochs(
        _epochs(), known_period=KnownPeriod(8.0, "external-fixed-period")
    )
    arrays = pad_tokenized_objects((tokenized,))
    return {
        key: torch.from_numpy(arrays[key]).to(device)
        for key in ("phase_features", "phase_mask", "geometry_features", "epoch_features", "epoch_mask")
    }


def test_axial_grid_has_exactly_6144_unique_projective_points() -> None:
    grid = axial_healpix_grid(32)
    assert grid.vectors.shape == (6144, 3)
    assert len(np.unique(np.round(grid.vectors, 12), axis=0)) == 6144
    assert np.array_equal(grid.full_to_axial[:], grid.full_to_axial[:])
    # Every full-sphere pixel and its antipodal representative map to one axis.
    counts = np.bincount(grid.full_to_axial, minlength=6144)
    assert np.all(counts == 2)


def test_mode_extraction_is_stable_and_separated() -> None:
    grid = axial_healpix_grid()
    scores = np.linspace(-1.0, 1.0, len(grid.vectors))
    scores[10], scores[2000], scores[5000] = 10.0, 9.0, 8.0
    modes = extract_axial_modes(scores, grid=grid)
    assert [mode.grid_index for mode in modes] == [10, 2000, 5000]
    pair_errors = [
        axial_angular_error_deg(modes[left].axis_xyz, modes[right].axis_xyz)
        for left in range(3)
        for right in range(left + 1, 3)
    ]
    assert min(pair_errors) >= 15.0


def test_chunked_grid_scoring_and_refinement_are_finite_and_bounded() -> None:
    torch.manual_seed(21)
    config = K3ScoreModelConfig(
        hidden_dim=32, attention_heads=4, convolution_blocks=1, set_encoder_layers=1,
        dropout=0.0, score_chunk_size=1024,
    )
    model = CandidateConditionedScorer(config).eval()
    inputs = _model_inputs()
    scores = score_axial_grid(model, inputs, chunk_size=1024)
    assert scores.shape == (6144,) and np.all(np.isfinite(scores))
    modes = extract_axial_modes(scores)
    starts = np.asarray([mode.axis_xyz for mode in modes])
    axes, refined_scores = refine_axes(model, inputs, starts, config=config)
    assert axes.shape == (3, 3) and np.all(np.isfinite(refined_scores))
    displacement = axial_angular_error_deg(axes, starts)
    assert np.all(displacement <= 5.0 + 1e-5)


def test_ensemble_refinement_uses_the_mean_scorer_without_averaging_axes() -> None:
    torch.manual_seed(23)
    config = K3ScoreModelConfig(
        hidden_dim=32, attention_heads=4, convolution_blocks=1, set_encoder_layers=1,
        dropout=0.0, score_chunk_size=1024,
    )
    model = CandidateConditionedScorer(config).eval()
    duplicate = deepcopy(model).eval()
    inputs = _model_inputs()
    grid_scores = score_axial_grid(model, inputs, chunk_size=1024)
    starts = np.asarray([mode.axis_xyz for mode in extract_axial_modes(grid_scores)])

    single_axes, single_scores = refine_axes(model, inputs, starts)
    ensemble_axes, ensemble_scores = refine_ensemble_axes((model, duplicate), inputs, starts)

    np.testing.assert_allclose(ensemble_axes, single_axes, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(ensemble_scores, single_scores, rtol=0.0, atol=1e-6)
    assert np.all(axial_angular_error_deg(ensemble_axes, starts) <= 5.0 + 1e-5)


def test_ensemble_refinement_rejects_invalid_membership() -> None:
    config = K3ScoreModelConfig(
        hidden_dim=32, attention_heads=4, convolution_blocks=1, set_encoder_layers=1,
        dropout=0.0,
    )
    model = CandidateConditionedScorer(config).eval()
    inputs = _model_inputs()
    starts = axial_healpix_grid().vectors[:3]
    with np.testing.assert_raises_regex(K3InferenceError, "at least two models"):
        refine_ensemble_axes((model,), inputs, starts)

    different = CandidateConditionedScorer(
        K3ScoreModelConfig(
            hidden_dim=64, attention_heads=4, convolution_blocks=1, set_encoder_layers=1,
            dropout=0.0,
        )
    ).eval()
    with np.testing.assert_raises_regex(K3InferenceError, "identical configurations"):
        refine_ensemble_axes((model, different), inputs, starts)


def test_public_api_has_no_ground_truth_and_returns_structured_failure_or_k3() -> None:
    assert "target" not in inspect.signature(DeLPHIK3Predictor.predict).parameters
    config = K3ScoreModelConfig(
        hidden_dim=32, attention_heads=4, convolution_blocks=1, set_encoder_layers=1,
        dropout=0.0, score_chunk_size=2048,
    )
    model = CandidateConditionedScorer(config)
    calibration = K3Calibration(
        calibration_id="unit-test",
        temperature=1.0,
        cone90_deg=30.0,
        cone95_deg=40.0,
        risk_raw_knots=(0.0, 1.0),
        risk_deg_knots=(0.0, 60.0),
    )
    predictor = DeLPHIK3Predictor(
        K3ModelBundle(model, "untrained-unit-test", calibration, config)
    )
    failed = predictor.predict(_epochs(), known_period=None)
    assert failed.status is K3PredictionStatus.FAILED and not failed.axes
    result = predictor.predict(
        _epochs(), known_period=KnownPeriod(8.0, "external-fixed-period")
    )
    assert result.status is not K3PredictionStatus.FAILED
    assert len(result.axes) == 3
    assert all(np.isfinite(axis.axis_xyz).all() for axis in result.axes)
