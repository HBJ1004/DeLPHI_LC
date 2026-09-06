"""Chunked score-grid evaluation and bounded candidate refinement."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .config import K3ScoreModelConfig
from .grid import AxialGrid, axial_healpix_grid
from .model import CandidateConditionedScorer


class K3InferenceError(ValueError):
    """Raised when deterministic K3 inference cannot produce finite axes."""


_INPUT_KEYS = (
    "phase_features",
    "phase_mask",
    "geometry_features",
    "epoch_features",
    "epoch_mask",
)


def _validated_inputs(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    missing = [key for key in _INPUT_KEYS if key not in inputs]
    if missing:
        raise K3InferenceError("missing scorer inputs: " + ", ".join(missing))
    values = {key: inputs[key] for key in _INPUT_KEYS}
    if not all(isinstance(value, torch.Tensor) for value in values.values()):
        raise K3InferenceError("all scorer inputs must be Torch tensors")
    batch_sizes = {value.shape[0] for value in values.values()}
    if len(batch_sizes) != 1:
        raise K3InferenceError("scorer input batch dimensions must align")
    return values


def score_axial_grid(
    model: CandidateConditionedScorer,
    inputs: Mapping[str, torch.Tensor],
    *,
    grid: AxialGrid | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Evaluate every unique axial HEALPix point in bounded memory."""
    values = _validated_inputs(inputs)
    selected_grid = axial_healpix_grid(model.config.nside) if grid is None else grid
    if selected_grid.nside != model.config.nside:
        raise K3InferenceError("model and axial grid nside differ")
    batch_size = next(iter(values.values())).shape[0]
    size = model.config.score_chunk_size if chunk_size is None else chunk_size
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise K3InferenceError("score chunk size must be a positive integer")
    device = next(model.parameters()).device
    if any(value.device != device for value in values.values()):
        raise K3InferenceError("model and scorer inputs must share one device")
    grid_tensor = torch.as_tensor(
        np.array(selected_grid.vectors, copy=True), dtype=values["phase_features"].dtype, device=device
    )
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        phase_interactions, epoch_latent = model._encode_photometry(
            values["phase_features"],
            values["phase_mask"],
            values["epoch_features"],
            values["epoch_mask"],
        )
        for start in range(0, len(grid_tensor), size):
            candidates = grid_tensor[start : start + size].unsqueeze(0).expand(batch_size, -1, -1)
            chunks.append(
                model.score_encoded(
                    phase_interactions,
                    epoch_latent,
                    values["phase_mask"],
                    values["geometry_features"],
                    values["epoch_mask"],
                    candidates,
                ).scores.detach().cpu()
            )
    scores = torch.cat(chunks, dim=1).numpy().astype(np.float64, copy=False)
    if scores.shape != (batch_size, len(selected_grid.vectors)) or not np.all(np.isfinite(scores)):
        raise K3InferenceError("model produced an invalid axial score grid")
    return scores[0] if batch_size == 1 else scores


def _cap_axial_displacement(
    candidates: torch.Tensor, starts: torch.Tensor, maximum_radians: float
) -> torch.Tensor:
    aligned = torch.where(
        torch.sum(candidates * starts, dim=-1, keepdim=True) < 0,
        -candidates,
        candidates,
    )
    dot = torch.sum(aligned * starts, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    tangent = aligned - dot * starts
    tangent = tangent / torch.clamp(torch.linalg.vector_norm(tangent, dim=-1, keepdim=True), min=1e-12)
    capped = math.cos(maximum_radians) * starts + math.sin(maximum_radians) * tangent
    return torch.where(angle > maximum_radians, capped, aligned)


def refine_axes(
    model: CandidateConditionedScorer,
    inputs: Mapping[str, torch.Tensor],
    starts: np.ndarray | torch.Tensor,
    *,
    config: K3ScoreModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run eight projected ascent steps, capped at five degrees from each start."""
    values = _validated_inputs(inputs)
    selected_config = model.config if config is None else config
    if next(iter(values.values())).shape[0] != 1:
        raise K3InferenceError("local refinement currently requires one object")
    device = next(model.parameters()).device
    dtype = values["phase_features"].dtype
    candidates = torch.as_tensor(starts, dtype=dtype, device=device)
    if candidates.shape != (selected_config.candidate_count, 3) or not bool(torch.isfinite(candidates).all()):
        raise K3InferenceError("refinement starts must contain three finite axes")
    starts_tensor = F.normalize(candidates, dim=-1)
    candidates = starts_tensor.clone()
    maximum = math.radians(selected_config.refinement_max_displacement_deg)
    with torch.no_grad():
        phase_interactions, epoch_latent = model._encode_photometry(
            values["phase_features"], values["phase_mask"], values["epoch_features"], values["epoch_mask"]
        )
    for _ in range(selected_config.refinement_steps):
        candidates = candidates.detach().requires_grad_(True)
        scores = model.score_encoded(
            phase_interactions, epoch_latent, values["phase_mask"], values["geometry_features"],
            values["epoch_mask"], candidates.unsqueeze(0)
        ).scores[0]
        gradient = torch.autograd.grad(scores.sum(), candidates, only_inputs=True)[0]
        if not bool(torch.isfinite(gradient).all()):
            raise K3InferenceError("candidate refinement produced non-finite gradients")
        tangent_gradient = gradient - torch.sum(gradient * candidates, dim=-1, keepdim=True) * candidates
        updated = F.normalize(
            candidates + selected_config.refinement_learning_rate * tangent_gradient, dim=-1
        )
        candidates = _cap_axial_displacement(updated, starts_tensor, maximum)
        candidates = F.normalize(candidates, dim=-1)
    with torch.no_grad():
        final_scores = model.score_encoded(
            phase_interactions, epoch_latent, values["phase_mask"], values["geometry_features"],
            values["epoch_mask"], candidates.unsqueeze(0)
        ).scores[0]
    axes = candidates.detach().cpu().numpy().astype(np.float64, copy=False)
    scores = final_scores.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(axes)) or not np.all(np.isfinite(scores)):
        raise K3InferenceError("candidate refinement did not return finite outputs")
    return axes, scores


def refine_ensemble_axes(
    models: Sequence[CandidateConditionedScorer],
    inputs: Mapping[str, torch.Tensor],
    starts: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine K=3 axes against the arithmetic mean score of multiple models.

    The candidates remain shared throughout optimization.  In particular, this
    function never refines each model independently and never averages axes.
    """
    values = _validated_inputs(inputs)
    if len(models) < 2:
        raise K3InferenceError("ensemble refinement requires at least two models")
    config = models[0].config
    if any(model.config != config for model in models[1:]):
        raise K3InferenceError("ensemble models must have identical configurations")
    if next(iter(values.values())).shape[0] != 1:
        raise K3InferenceError("local refinement currently requires one object")

    devices = {next(model.parameters()).device for model in models}
    if len(devices) != 1:
        raise K3InferenceError("ensemble models must share one device")
    device = next(iter(devices))
    if any(value.device != device for value in values.values()):
        raise K3InferenceError("ensemble models and scorer inputs must share one device")

    dtype = values["phase_features"].dtype
    candidates = torch.as_tensor(starts, dtype=dtype, device=device)
    if candidates.shape != (config.candidate_count, 3) or not bool(torch.isfinite(candidates).all()):
        raise K3InferenceError("refinement starts must contain three finite axes")
    starts_tensor = F.normalize(candidates, dim=-1)
    candidates = starts_tensor.clone()
    maximum = math.radians(config.refinement_max_displacement_deg)

    encoded: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for model in models:
            encoded.append(
                model._encode_photometry(
                    values["phase_features"],
                    values["phase_mask"],
                    values["epoch_features"],
                    values["epoch_mask"],
                )
            )

    def mean_scores(candidate_axes: torch.Tensor) -> torch.Tensor:
        outputs = [
            model.score_encoded(
                phase_interactions,
                epoch_latent,
                values["phase_mask"],
                values["geometry_features"],
                values["epoch_mask"],
                candidate_axes.unsqueeze(0),
            ).scores[0]
            for model, (phase_interactions, epoch_latent) in zip(models, encoded, strict=True)
        ]
        return torch.stack(outputs, dim=0).mean(dim=0)

    for _ in range(config.refinement_steps):
        candidates = candidates.detach().requires_grad_(True)
        scores = mean_scores(candidates)
        gradient = torch.autograd.grad(scores.sum(), candidates, only_inputs=True)[0]
        if not bool(torch.isfinite(gradient).all()):
            raise K3InferenceError("ensemble candidate refinement produced non-finite gradients")
        tangent_gradient = gradient - torch.sum(gradient * candidates, dim=-1, keepdim=True) * candidates
        candidates = F.normalize(
            candidates + config.refinement_learning_rate * tangent_gradient, dim=-1
        )
        candidates = _cap_axial_displacement(candidates, starts_tensor, maximum)
        candidates = F.normalize(candidates, dim=-1)

    with torch.no_grad():
        final_scores = mean_scores(candidates)
    axes = candidates.detach().cpu().numpy().astype(np.float64, copy=False)
    scores = final_scores.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(axes)) or not np.all(np.isfinite(scores)):
        raise K3InferenceError("ensemble candidate refinement did not return finite outputs")
    return axes, scores
