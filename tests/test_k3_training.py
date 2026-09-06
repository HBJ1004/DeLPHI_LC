"""Smoke tests for K3 ratio losses, negative sampling, and deterministic fitting."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from lc_pipeline.k3.config import K3ScoreModelConfig, K3TrainingConfig
from lc_pipeline.k3.losses import sample_local_hard_negative_axes, sample_uniform_negative_axes
from lc_pipeline.k3.model import CandidateConditionedScorer
from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
from lc_pipeline.k3.tokenizer import tokenize_epochs
from lc_pipeline.k3.training import (
    K3TrainingError,
    K3TrainingExample,
    collate_examples,
    deterministic_derangement_indices,
    fit_k3,
    interleave_synthetic_and_real,
    load_k3_checkpoint,
    shuffle_training_labels,
    swap_input_examples,
)
from lc_pipeline.v2.preprocessing import KnownPeriod, Observation, ObservationEpoch


def test_gpu_determinism_workspace_is_declared_before_torch_use() -> None:
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"}


def test_control_derangement_is_deterministic_bijective_and_has_no_fixed_points() -> None:
    first = deterministic_derangement_indices(17, seed=20260901)
    second = deterministic_derangement_indices(17, seed=20260901)
    assert np.array_equal(first, second)
    assert set(first.tolist()) == set(range(17))
    assert np.all(first != np.arange(17))
    assert not first.flags.writeable
    with pytest.raises(K3TrainingError, match="at least two"):
        deterministic_derangement_indices(1, seed=20260901)


def _example(object_id: str, offset: float) -> K3TrainingExample:
    observations = tuple(
        Observation(
            time_jd=2450000.0 + offset + index / 24.0,
            relative_brightness=math.exp(0.2 * math.sin(index + offset)),
            sun_asteroid_ecliptic_j2000_au=(1.0, 0.1, 0.2),
            observer_asteroid_ecliptic_j2000_au=(0.2, 1.0, 0.1),
        )
        for index in range(12)
    )
    tokenized = tokenize_epochs(
        (ObservationEpoch(object_id + "-epoch", observations),),
        known_period=KnownPeriod(8.0, "synthetic-known-period"),
    )
    axis = np.asarray((math.cos(offset), math.sin(offset), 0.6), dtype=np.float32)
    axis /= np.linalg.norm(axis)
    return K3TrainingExample(object_id, tokenized, np.stack((axis, -axis)))


def test_negative_samplers_respect_exclusion_and_annulus() -> None:
    targets = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    mask = torch.tensor([[True, True]])
    generator = torch.Generator().manual_seed(2)
    uniform = sample_uniform_negative_axes(targets, mask, count=32, generator=generator)
    errors = torch.rad2deg(torch.acos(torch.abs(torch.einsum("bnc,bsc->bns", uniform, targets))))
    assert float(errors.min()) >= 10.0 - 1e-5
    hard = sample_local_hard_negative_axes(targets, mask, count=32, generator=generator)
    hard_errors = torch.rad2deg(torch.acos(torch.abs(torch.einsum("bnc,bsc->bns", hard, targets))))
    assert float(hard_errors.min()) >= 10.0 - 1e-5
    assert float(hard_errors.min()) <= 45.0 + 1e-5


def test_collate_and_cpu_smoke_fit_are_finite_and_checkpointed(tmp_path) -> None:
    values = (_example("a", 0.0), _example("b", 0.2), _example("c", 0.4), _example("d", 0.6))
    batch = collate_examples(values[:2])
    assert batch["target_axes"].shape == (2, 2, 3)
    config = K3TrainingConfig(
        seed=17, batch_size=2, synthetic_max_epochs=2, synthetic_patience=1,
        real_max_epochs=2, real_patience=1, uniform_negatives=4, hard_negatives=2,
    )
    model = CandidateConditionedScorer(
        K3ScoreModelConfig(hidden_dim=32, attention_heads=4, convolution_blocks=1, set_encoder_layers=1, dropout=0.0)
    )
    provenance = {
        "protocol_sha256": K3_PROTOCOL_SHA256,
        "implementation_commit": "a" * 40,
    }
    result = fit_k3(
        model,
        values[:3],
        values[3:],
        config=config,
        device="cpu",
        checkpoint_path=tmp_path / "best.pt",
        run_provenance=provenance,
    )
    assert result.epochs_completed >= 1 and np.isfinite(result.best_validation_loss)
    assert result.checkpoint_sha256 and (tmp_path / "best.pt").is_file()
    swapped = swap_input_examples(values, seed=3)
    shuffled = shuffle_training_labels(values, seed=3)
    assert swapped[0].tokenized is not values[0].tokenized
    assert shuffled[0].target_axes is not values[0].target_axes
    mixed = interleave_synthetic_and_real(values[:3], (values[3],), seed=3)
    assert len(mixed) == 4 and mixed[-1].object_id == "d"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loaded = load_k3_checkpoint(
        tmp_path / "best.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        stage="synthetic",
        run_provenance=provenance,
    )
    assert loaded["stage"] == "synthetic"
    assert loaded["schema"] == "delphi.k3-checkpoint.v2"
    assert loaded["run_provenance"] == provenance
    with pytest.raises(K3TrainingError, match="run provenance"):
        load_k3_checkpoint(
            tmp_path / "best.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            stage="synthetic",
            run_provenance={**provenance, "implementation_commit": "b" * 40},
        )
    bad_config = K3TrainingConfig(
        seed=17, batch_size=2, synthetic_max_epochs=3, synthetic_patience=1,
        real_max_epochs=2, real_patience=1, uniform_negatives=4, hard_negatives=2,
    )
    with pytest.raises(K3TrainingError, match="configuration"):
        load_k3_checkpoint(
            tmp_path / "best.pt", model=model, optimizer=optimizer, config=bad_config, stage="synthetic"
        )
