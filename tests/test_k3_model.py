"""Symmetry and dependency tests for the candidate-conditioned scorer."""

from __future__ import annotations

import torch

from lc_pipeline.k3.config import K3ScoreModelConfig
from lc_pipeline.k3.model import CandidateConditionedScorer


def _inputs(batch: int = 2, epochs: int = 3, candidates: int = 5):
    generator = torch.Generator().manual_seed(91)
    phase = torch.randn(batch, epochs, 64, 6, generator=generator)
    phase_mask = torch.rand(batch, epochs, 64, generator=generator) > 0.25
    epoch_mask = torch.ones(batch, epochs, dtype=torch.bool)
    geometry = torch.zeros(batch, epochs, 64, 8)
    sun = torch.randn(batch, epochs, 64, 3, generator=generator)
    observer = torch.randn(batch, epochs, 64, 3, generator=generator)
    geometry[..., :3] = torch.nn.functional.normalize(sun, dim=-1)
    geometry[..., 3] = 1.0 + torch.rand(batch, epochs, 64, generator=generator)
    geometry[..., 4:7] = torch.nn.functional.normalize(observer, dim=-1)
    geometry[..., 7] = 0.5 + torch.rand(batch, epochs, 64, generator=generator)
    geometry = geometry * phase_mask[..., None]
    epoch_features = torch.randn(batch, epochs, 22, generator=generator)
    candidate = torch.randn(batch, candidates, 3, generator=generator)
    return phase, phase_mask, geometry, epoch_features, epoch_mask, candidate


def _random_rotation(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.linalg.det(orthogonal) < 0:
        orthogonal[:, 0] *= -1
    return orthogonal


def test_scores_are_antipode_and_shared_rotation_invariant() -> None:
    torch.manual_seed(7)
    model = CandidateConditionedScorer(K3ScoreModelConfig(dropout=0.0)).eval()
    values = _inputs()
    with torch.no_grad():
        score = model(*values).scores
        antipode = model(*values[:-1], -values[-1]).scores
        # A generic SO(3) matrix exercises all axes, not only an azimuthal turn.
        rotation = _random_rotation(731)
        geometry = values[2].clone()
        geometry[..., :3] = geometry[..., :3] @ rotation.T
        geometry[..., 4:7] = geometry[..., 4:7] @ rotation.T
        rotated = model(
            values[0], values[1], geometry, values[3], values[4], values[5] @ rotation.T
        ).scores
    assert torch.allclose(score, antipode, atol=1e-6, rtol=1e-6)
    assert torch.allclose(score, rotated, atol=2e-6, rtol=2e-6)


def test_epoch_set_is_permutation_invariant() -> None:
    torch.manual_seed(8)
    model = CandidateConditionedScorer(K3ScoreModelConfig(dropout=0.0)).eval()
    values = _inputs(batch=1)
    order = torch.tensor([2, 0, 1])
    with torch.no_grad():
        expected = model(*values).scores
        actual = model(
            values[0][:, order],
            values[1][:, order],
            values[2][:, order],
            values[3][:, order],
            values[4][:, order],
            values[5],
        ).scores
    assert torch.allclose(expected, actual, atol=2e-6, rtol=2e-6)


def test_scores_depend_on_input_and_have_candidate_gradients() -> None:
    torch.manual_seed(9)
    model = CandidateConditionedScorer(K3ScoreModelConfig(dropout=0.0)).eval()
    values = list(_inputs(batch=1, candidates=3))
    candidates = values[-1].requires_grad_(True)
    first = model(*values[:-1], candidates).scores
    changed_phase = values[0].clone()
    changed_phase[..., 0] *= -2.0
    second = model(changed_phase, *values[1:-1], candidates).scores
    first.sum().backward()
    assert not torch.allclose(first, second)
    assert candidates.grad is not None and torch.isfinite(candidates.grad).all()
    assert float(torch.linalg.vector_norm(candidates.grad)) > 0


def test_default_model_has_no_raw_candidate_projection() -> None:
    model = CandidateConditionedScorer()
    parameter_names = tuple(name for name, _ in model.named_parameters())
    assert not any("candidate_projection" in name for name in parameter_names)
    output = model(*_inputs(batch=1, epochs=2, candidates=4))
    assert output.scores.shape == (1, 4)
