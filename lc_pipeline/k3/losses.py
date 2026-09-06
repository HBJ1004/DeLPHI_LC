"""Density-ratio and hard-negative objectives for candidate-conditioned K3."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch.nn import functional as F

from .config import K3TrainingConfig
from .model import CandidateConditionedScorer


class K3LossError(ValueError):
    """Raised for invalid positive/negative candidate sets."""


def _normalize_axes(axes: torch.Tensor) -> torch.Tensor:
    if axes.ndim != 3 or axes.shape[-1] != 3 or not torch.is_floating_point(axes):
        raise K3LossError("axes must have shape [B,S,3] and floating dtype")
    norms = torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    if bool(torch.any(norms <= 1e-12)) or not bool(torch.isfinite(axes).all()):
        raise K3LossError("axes must be finite and nonzero")
    return F.normalize(axes, dim=-1)


def _orthogonal_basis(axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reference = torch.zeros_like(axis)
    reference[..., 2] = 1.0
    use_x = torch.abs(axis[..., 2]) > 0.9
    reference = torch.where(use_x[..., None], torch.tensor((1.0, 0.0, 0.0), device=axis.device, dtype=axis.dtype), reference)
    first = F.normalize(torch.linalg.cross(reference, axis, dim=-1), dim=-1)
    second = torch.linalg.cross(axis, first, dim=-1)
    return first, second


def _canonicalize(axes: torch.Tensor) -> torch.Tensor:
    z, y, x = axes[..., 2], axes[..., 1], axes[..., 0]
    flip = (z < -1e-12) | ((torch.abs(z) <= 1e-12) & ((y < -1e-12) | ((torch.abs(y) <= 1e-12) & (x < 0))))
    return torch.where(flip[..., None], -axes, axes)


def _valid_against_targets(candidate: torch.Tensor, targets: torch.Tensor, exclusion_rad: float) -> torch.Tensor:
    dots = torch.abs(torch.einsum("bnc,bsc->bns", candidate, targets)).clamp(0.0, 1.0)
    distances = torch.acos(dots)
    return torch.all(distances >= exclusion_rad, dim=-1)


def sample_uniform_negative_axes(
    target_axes: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
    exclusion_deg: float = 10.0,
) -> torch.Tensor:
    """Sample uniform projective negatives, rejecting a 10-degree target tube."""
    targets = _normalize_axes(target_axes)
    if target_mask.shape != targets.shape[:2] or target_mask.dtype is not torch.bool:
        raise K3LossError("target_mask must be boolean and align with target_axes")
    if count <= 0 or not 0 < exclusion_deg < 90:
        raise K3LossError("negative count and exclusion angle are invalid")
    batch, slots = targets.shape[:2]
    chosen = torch.empty(batch, count, 3, dtype=targets.dtype, device=targets.device)
    remaining = torch.ones(batch, count, dtype=torch.bool, device=targets.device)
    exclusion_rad = math.radians(exclusion_deg)
    for _ in range(100):
        proposal = torch.randn(batch, count, 3, generator=generator, dtype=targets.dtype, device=targets.device)
        proposal = _canonicalize(F.normalize(proposal, dim=-1))
        valid_targets = torch.where(target_mask[..., None], targets, targets[:, :1])
        valid = _valid_against_targets(proposal, valid_targets, exclusion_rad)
        take = remaining & valid
        chosen = torch.where(take[..., None], proposal, chosen)
        remaining = remaining & ~take
        if not bool(remaining.any()):
            return chosen
    if bool(remaining.any()):
        raise K3LossError("uniform negative sampler could not satisfy the exclusion tube")
    return chosen


def sample_local_hard_negative_axes(
    target_axes: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
    minimum_deg: float = 5.0,
    maximum_deg: float = 45.0,
    exclusion_deg: float = 10.0,
) -> torch.Tensor:
    """Sample local annulus negatives around explicit source axes."""
    targets = _normalize_axes(target_axes)
    if target_mask.shape != targets.shape[:2] or target_mask.dtype is not torch.bool:
        raise K3LossError("target_mask must align with target_axes")
    if not 0 < minimum_deg < exclusion_deg < maximum_deg <= 90 or count <= 0:
        raise K3LossError("hard-negative angular ranges are invalid")
    batch, slots = targets.shape[:2]
    selected_target = torch.argmax(target_mask.to(torch.int64), dim=1)
    centers = targets[torch.arange(batch, device=targets.device), selected_target]
    first, second = _orthogonal_basis(centers)
    # The annulus may be declared as 5--45 degrees while the exclusion gate
    # removes everything below 10 degrees.  Sampling from the effective
    # interval prevents sampling axes that the exclusion gate must reject.
    effective_minimum = max(minimum_deg, exclusion_deg)
    low, high = math.cos(math.radians(maximum_deg)), math.cos(math.radians(effective_minimum))
    cosine = low + (high - low) * torch.rand(batch, count, generator=generator, dtype=targets.dtype, device=targets.device)
    sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0))
    azimuth = 2.0 * math.pi * torch.rand(batch, count, generator=generator, dtype=targets.dtype, device=targets.device)
    proposal = cosine[..., None] * centers[:, None, :]
    proposal = proposal + sine[..., None] * (
        torch.cos(azimuth)[..., None] * first[:, None, :]
        + torch.sin(azimuth)[..., None] * second[:, None, :]
    )
    proposal = _canonicalize(F.normalize(proposal, dim=-1))
    valid_targets = torch.where(target_mask[..., None], targets, targets[:, :1])
    valid = _valid_against_targets(proposal, valid_targets, math.radians(exclusion_deg))
    if not bool(valid.all()):
        # Resample recursively with a deterministic shifted generator state.
        return sample_local_hard_negative_axes(
            targets,
            target_mask,
            count=count,
            generator=generator,
            minimum_deg=minimum_deg,
            maximum_deg=maximum_deg,
            exclusion_deg=exclusion_deg,
        )
    return proposal


def density_ratio_training_loss(
    model: CandidateConditionedScorer,
    inputs: Mapping[str, torch.Tensor],
    target_axes: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    config: K3TrainingConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """BCE on positive/sampled-negative pairs plus a local ranking penalty.

    Target-dependent exclusions and hard negatives mean this objective does
    not by itself estimate a calibrated joint-to-product density ratio.
    """
    if target_axes.ndim != 3 or target_axes.shape[-1] != 3 or not torch.is_floating_point(target_axes):
        raise K3LossError("target axes must have shape [B,S,3] and floating dtype")
    if target_mask.shape != target_axes.shape[:2] or target_mask.dtype is not torch.bool:
        raise K3LossError("target_mask must be boolean and align with target_axes")
    if not bool(target_mask.any(dim=1).all()):
        raise K3LossError("each object requires at least one positive source axis")
    if not bool(torch.isfinite(target_axes).all()):
        raise K3LossError("target axes and padding must be finite")
    placeholder = torch.zeros_like(target_axes)
    placeholder[..., 0] = 1.0
    targets = _normalize_axes(torch.where(target_mask[..., None], target_axes, placeholder))
    uniform = sample_uniform_negative_axes(
        targets,
        target_mask,
        count=config.uniform_negatives,
        generator=generator,
        exclusion_deg=config.negative_exclusion_deg,
    )
    hard = sample_local_hard_negative_axes(
        targets,
        target_mask,
        count=config.hard_negatives,
        generator=generator,
        minimum_deg=config.hard_negative_min_deg,
        maximum_deg=config.hard_negative_max_deg,
        exclusion_deg=config.negative_exclusion_deg,
    )
    positive_count = targets.shape[1]
    candidates = torch.cat((targets, uniform, hard), dim=1)
    scores = model(**inputs, candidates=candidates).scores
    labels = torch.zeros_like(scores)
    labels[:, :positive_count] = target_mask.to(scores.dtype)
    valid = torch.ones_like(scores, dtype=torch.bool)
    valid[:, :positive_count] = target_mask
    bce = F.binary_cross_entropy_with_logits(scores[valid], labels[valid])
    positive_scores = scores[:, :positive_count]
    hard_scores = scores[:, positive_count + uniform.shape[1] :]
    masked_positive = positive_scores.masked_fill(~target_mask, float("-inf"))
    best_positive = masked_positive.max(dim=1).values
    ranking = F.softplus(hard_scores - best_positive[:, None]).mean()
    total = bce + config.rank_loss_weight * ranking
    return total, {"bce": bce.detach(), "rank": ranking.detach()}
