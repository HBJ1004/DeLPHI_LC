"""Candidate-conditioned, antipode-invariant K3 compatibility scorer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import K3ScoreModelConfig

CANDIDATE_INVARIANT_NAMES = (
    "dot_p_s_squared",
    "dot_p_e_squared",
    "dot_p_s_times_dot_p_e",
    "dot_s_e",
    "phase_angle_over_pi",
    "log1p_sun_distance_au",
    "log1p_observer_distance_au",
)
INTERACTION_DIM = 8


class K3ModelError(ValueError):
    """Raised when scorer tensors violate the model contract."""


@dataclass(frozen=True)
class K3ScoreOutput:
    scores: torch.Tensor


class CircularResidualBlock(nn.Module):
    """Mask-aware circular phase convolution with no absolute phase position."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.convolution_1 = nn.Conv1d(
            width, width, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.normalization_1 = nn.GroupNorm(groups, width)
        self.convolution_2 = nn.Conv1d(
            width, width, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.normalization_2 = nn.GroupNorm(groups, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.convolution_1(values)
        values = F.gelu(self.normalization_1(values))
        values = self.dropout(values)
        values = self.normalization_2(self.convolution_2(values))
        values = F.gelu(values + residual)
        return values * mask


class CandidateConditionedScorer(nn.Module):
    """Score proposed spin axes from phase curves and relative geometry.

    Candidate coordinates never enter directly. Only dot-product invariants are
    exposed, making the score unchanged under ``p -> -p`` and under a shared
    3-D rotation of candidate, Sun, and observer vectors.
    """

    def __init__(self, config: K3ScoreModelConfig = K3ScoreModelConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_dim
        self.phase_projection = nn.Conv1d(config.phase_feature_count, width, kernel_size=1)
        self.phase_blocks = nn.ModuleList(
            CircularResidualBlock(width, config.dropout)
            for _ in range(config.convolution_blocks)
        )
        self.interaction_projection = nn.Linear(width, INTERACTION_DIM)
        self.epoch_projection = nn.Sequential(
            nn.Linear(width + config.epoch_feature_count, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.attention_heads,
            dim_feedforward=2 * width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.epoch_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.set_encoder_layers,
            enable_nested_tensor=False,
        )
        invariant_count = len(CANDIDATE_INVARIANT_NAMES)
        candidate_summary = 2 * invariant_count + INTERACTION_DIM * invariant_count
        self.epoch_evidence = nn.Sequential(
            nn.Linear(width + candidate_summary, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width, 1),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encoder_parameters(self):
        """Yield photometry/set parameters for the lower fine-tune LR group."""
        for module in (
            self.phase_projection,
            self.phase_blocks,
            self.interaction_projection,
            self.epoch_projection,
            self.epoch_encoder,
        ):
            yield from module.parameters()

    def scorer_parameters(self):
        for module in (self.epoch_evidence, self.score_head):
            yield from module.parameters()

    def _validate_inputs(
        self,
        phase_features: torch.Tensor,
        phase_mask: torch.Tensor,
        geometry_features: torch.Tensor,
        epoch_features: torch.Tensor,
        epoch_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> None:
        if phase_features.ndim != 4 or phase_features.shape[-2:] != (
            64,
            self.config.phase_feature_count,
        ):
            raise K3ModelError("phase_features must have shape [B,E,64,F]")
        batch, epochs = phase_features.shape[:2]
        if phase_mask.shape != (batch, epochs, 64) or phase_mask.dtype is not torch.bool:
            raise K3ModelError("phase_mask must be boolean with shape [B,E,64]")
        if geometry_features.shape != (batch, epochs, 64, 8):
            raise K3ModelError("geometry_features must have shape [B,E,64,8]")
        if epoch_features.shape != (batch, epochs, self.config.epoch_feature_count):
            raise K3ModelError("epoch_features have the wrong shape")
        if epoch_mask.shape != (batch, epochs) or epoch_mask.dtype is not torch.bool:
            raise K3ModelError("epoch_mask must be boolean with shape [B,E]")
        if candidates.ndim != 3 or candidates.shape[0] != batch or candidates.shape[2] != 3:
            raise K3ModelError("candidates must have shape [B,C,3]")
        tensors = (phase_features, geometry_features, epoch_features, candidates)
        if any(not torch.is_floating_point(value) for value in tensors):
            raise K3ModelError("model features and candidates must be floating point")
        if any(value.device != phase_features.device for value in (*tensors, phase_mask, epoch_mask)):
            raise K3ModelError("all scorer inputs must share one device")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise K3ModelError("scorer inputs must be finite")
        if not bool(epoch_mask.any(dim=1).all()):
            raise K3ModelError("each object requires at least one original epoch")
        if bool(torch.any(phase_mask & ~epoch_mask[..., None])):
            raise K3ModelError("padded epochs may not contain occupied phase bins")
        occupied_by_epoch = phase_mask.any(dim=-1)
        if not bool(torch.equal(occupied_by_epoch, epoch_mask)):
            raise K3ModelError("every real epoch must contain an occupied phase bin")
        candidate_norms = torch.linalg.vector_norm(candidates, dim=-1)
        if bool(torch.any(candidate_norms <= 1e-12)):
            raise K3ModelError("candidate axes must be nonzero")

    def _encode_photometry(
        self,
        phase_features: torch.Tensor,
        phase_mask: torch.Tensor,
        epoch_features: torch.Tensor,
        epoch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, epochs, phase_bins, _ = phase_features.shape
        mask = phase_mask.reshape(batch * epochs, 1, phase_bins).to(phase_features.dtype)
        values = phase_features.reshape(batch * epochs, phase_bins, -1).transpose(1, 2)
        values = self.phase_projection(values) * mask
        for block in self.phase_blocks:
            values = block(values, mask)
        phase_latent = values.transpose(1, 2).reshape(
            batch, epochs, phase_bins, self.config.hidden_dim
        )
        counts = phase_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(phase_features.dtype)
        pooled = torch.sum(phase_latent * phase_mask[..., None], dim=2) / counts
        epoch_latent = self.epoch_projection(torch.cat((pooled, epoch_features), dim=-1))
        epoch_latent = self.epoch_encoder(
            epoch_latent, src_key_padding_mask=~epoch_mask
        )
        epoch_latent = epoch_latent * epoch_mask[..., None]
        phase_interactions = torch.tanh(self.interaction_projection(phase_latent))
        return phase_interactions, epoch_latent

    @staticmethod
    def _candidate_invariants(
        geometry_features: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        sun = geometry_features[..., :3]
        observer = geometry_features[..., 4:7]
        dot_p_s = torch.einsum("beld,bcd->belc", sun, candidates)
        dot_p_e = torch.einsum("beld,bcd->belc", observer, candidates)
        dot_s_e = torch.sum(sun * observer, dim=-1).clamp(-1.0, 1.0)
        phase_angle = torch.acos(dot_s_e) / torch.pi
        sun_distance = torch.log1p(torch.clamp(geometry_features[..., 3], min=0.0))
        observer_distance = torch.log1p(torch.clamp(geometry_features[..., 7], min=0.0))
        shared = (dot_s_e, phase_angle, sun_distance, observer_distance)
        return torch.stack(
            (
                dot_p_s.square(),
                dot_p_e.square(),
                dot_p_s * dot_p_e,
                *(value[..., None].expand_as(dot_p_s) for value in shared),
            ),
            dim=-1,
        )

    def forward(
        self,
        phase_features: torch.Tensor,
        phase_mask: torch.Tensor,
        geometry_features: torch.Tensor,
        epoch_features: torch.Tensor,
        epoch_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> K3ScoreOutput:
        self._validate_inputs(
            phase_features,
            phase_mask,
            geometry_features,
            epoch_features,
            epoch_mask,
            candidates,
        )
        candidates = F.normalize(candidates, dim=-1)
        phase_interactions, epoch_latent = self._encode_photometry(
            phase_features, phase_mask, epoch_features, epoch_mask
        )
        return self.score_encoded(
            phase_interactions,
            epoch_latent,
            phase_mask,
            geometry_features,
            epoch_mask,
            candidates,
        )

    def score_encoded(
        self,
        phase_interactions: torch.Tensor,
        epoch_latent: torch.Tensor,
        phase_mask: torch.Tensor,
        geometry_features: torch.Tensor,
        epoch_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> K3ScoreOutput:
        """Score candidates from one reusable photometric object encoding."""
        candidates = F.normalize(candidates, dim=-1)
        invariants = self._candidate_invariants(geometry_features, candidates)
        weights = phase_mask.to(phase_interactions.dtype)
        counts = weights.sum(dim=-1).clamp_min(1.0)
        mean = torch.einsum("belcj,bel->becj", invariants, weights) / counts[..., None, None]
        second = torch.einsum("belcj,bel->becj", invariants.square(), weights) / counts[
            ..., None, None
        ]
        standard_deviation = torch.sqrt(torch.clamp(second - mean.square(), min=0.0) + 1e-8)
        interaction = torch.einsum(
            "belr,belcj,bel->becrj", phase_interactions, invariants, weights
        ) / counts[..., None, None, None]
        candidate_summary = torch.cat(
            (mean, standard_deviation, interaction.flatten(start_dim=-2)), dim=-1
        )
        expanded_epoch = epoch_latent[:, :, None, :].expand(
            -1, -1, candidates.shape[1], -1
        )
        evidence = self.epoch_evidence(torch.cat((expanded_epoch, candidate_summary), dim=-1))
        epoch_weights = epoch_mask.to(epoch_latent.dtype)
        object_evidence = torch.einsum("bech,be->bch", evidence, epoch_weights) / epoch_weights.sum(
            dim=1
        )[:, None, None]
        object_context = torch.einsum("beh,be->bh", epoch_latent, epoch_weights) / epoch_weights.sum(
            dim=1, keepdim=True
        )
        expanded_context = object_context[:, None, :].expand(-1, candidates.shape[1], -1)
        scores = self.score_head(torch.cat((object_evidence, expanded_context), dim=-1)).squeeze(-1)
        if not bool(torch.isfinite(scores).all()):
            raise K3ModelError("scorer produced non-finite compatibility scores")
        return K3ScoreOutput(scores=scores)
