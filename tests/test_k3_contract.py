"""Prospective contract tests for the candidate-conditioned K3 arm."""

from __future__ import annotations

import pytest

from lc_pipeline.k3.config import K3ConfigError, K3ScoreModelConfig, K3TrainingConfig
from lc_pipeline.k3.protocol import K3ProtocolError, load_protocol, validate_synthetic_donors
from lc_pipeline.k3.schema import K3AxialPrediction, K3AxisCandidate, K3PredictionStatus


def _axis() -> K3AxisCandidate:
    return K3AxisCandidate((1.0, 0.0, 0.0), 2.0, 0.1, 20.0, 30.0, 7)


def test_hash_bound_protocol_and_locked_primary_gates() -> None:
    protocol = load_protocol()
    assert protocol["scientific_question"]["known_period_scope"] == "primary"
    assert protocol["stage_gates"]["retrospective_oof"]["mean_error_deg_maximum"] == 22.0
    assert protocol["stage_gates"]["order_of_magnitude_claim"]["minimum_speed_ratio"] == 10.0


def test_locked_model_and_training_configuration_fail_closed() -> None:
    assert K3ScoreModelConfig().candidate_count == 3
    assert K3TrainingConfig(seed=17).synthetic_to_real_ratio == (3, 1)
    with pytest.raises(K3ConfigError, match="nside=32"):
        K3ScoreModelConfig(nside=16)
    with pytest.raises(K3ConfigError, match="seed"):
        K3TrainingConfig(seed=1)


def test_donor_partitions_reject_protected_or_overlapping_ids() -> None:
    arguments = dict(
        shape_train=["s1"], shape_validation=["s2"], shape_test=["s3"],
        geometry_train=["g1"], geometry_validation=["g2"], geometry_test=["g3"],
        retrospective_test_ids=["held-out"], temporal_ids=["future"],
    )
    validate_synthetic_donors(**arguments)
    with pytest.raises(K3ProtocolError, match="overlaps"):
        validate_synthetic_donors(**{**arguments, "shape_test": ["s1"]})
    with pytest.raises(K3ProtocolError, match="protected"):
        validate_synthetic_donors(**{**arguments, "geometry_train": ["future"]})


def test_prediction_is_exactly_three_axes_or_explicit_failure() -> None:
    digest = "a" * 64
    prediction = K3AxialPrediction(
        status=K3PredictionStatus.OK,
        axes=(_axis(), _axis(), _axis()),
        risk_deg=12.0,
        model_id="unit-test",
        period_provenance="measured-period",
        tokenizer_schema_sha256=digest,
    )
    assert len(prediction.as_mapping()["axes"]) == 3
    failed = K3AxialPrediction.failed(
        model_id="unit-test", tokenizer_schema_sha256=digest, reason="non-finite score grid"
    )
    assert failed.status is K3PredictionStatus.FAILED and not failed.axes
    with pytest.raises(ValueError, match="exactly three"):
        K3AxialPrediction(
            status=K3PredictionStatus.OK,
            axes=(_axis(),),
            risk_deg=12.0,
            model_id="unit-test",
            period_provenance=None,
            tokenizer_schema_sha256=digest,
        )
