"""Geometry-aware hierarchical HEALPix-density model for DeLPHI V2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .density import npix_for_nside
from .preprocessing import PHASE_FEATURE_SLICE, TOKEN_FEATURE_NAMES, TOKENIZER_SCHEMA_HASH

if TYPE_CHECKING:
    from .atlas import AxisAtlas


@dataclass(frozen=True)
class V2ModelConfig:
    nside: int = 32
    d_model: int = 128
    token_layers: int = 2
    epoch_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    tokenizer_schema_sha256: str = TOKENIZER_SCHEMA_HASH
    coordinate_frame: str = "asteroid_centric_ecliptic_j2000"
    geometry_required: bool = True
    phase_enabled: bool = True
    hierarchical_density: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.token_layers <= 0 or self.epoch_layers <= 0:
            raise ValueError("token_layers and epoch_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.phase_enabled, bool):
            raise ValueError("phase_enabled must be boolean")
        npix_for_nside(self.nside)
        if self.hierarchical_density and self.nside != 32:
            raise ValueError(
                "hierarchical_density currently requires the canonical nside=32 output"
            )
        if self.tokenizer_schema_sha256 != TOKENIZER_SCHEMA_HASH:
            raise ValueError("model tokenizer schema does not match the V2 canonical builder")


@dataclass(frozen=True)
class V2AxisModelConfig:
    """Frozen architecture for either comparison-study V2 axis head."""

    d_model: int = 128
    token_layers: int = 2
    epoch_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    candidate_count: int = 3
    head_type: Literal["plain", "residual"] = "plain"
    maximum_residual_deg: float = 45.0
    residual_regularization_weight: float = 0.01
    tokenizer_schema_sha256: str = TOKENIZER_SCHEMA_HASH
    coordinate_frame: str = "asteroid_centric_ecliptic_j2000"
    geometry_required: bool = True
    phase_enabled: bool = True
    atlas_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.token_layers <= 0 or self.epoch_layers <= 0:
            raise ValueError("token_layers and epoch_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.candidate_count != 3:
            raise ValueError("the comparison contract fixes candidate_count=3")
        if self.head_type not in {"plain", "residual"}:
            raise ValueError("head_type must be 'plain' or 'residual'")
        if not math.isfinite(self.maximum_residual_deg) or not 0 < self.maximum_residual_deg <= 90:
            raise ValueError("maximum_residual_deg must lie in (0, 90]")
        if (
            not math.isfinite(self.residual_regularization_weight)
            or self.residual_regularization_weight < 0
        ):
            raise ValueError("residual regularization weight must be finite and nonnegative")
        if self.tokenizer_schema_sha256 != TOKENIZER_SCHEMA_HASH:
            raise ValueError("model tokenizer schema does not match the V2 canonical builder")
        if self.head_type == "residual" and not self.atlas_sha256:
            raise ValueError("residual models require a content-addressed training atlas")
        if self.head_type == "plain" and self.atlas_sha256 is not None:
            raise ValueError("plain models must not declare an atlas")


@dataclass(frozen=True)
class DensityModelOutput:
    logits: torch.Tensor
    raw_risk: torch.Tensor
    nside16_logits: torch.Tensor | None = None
    nside8_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class AxisModelOutput:
    axes: torch.Tensor
    residual_angles_rad: torch.Tensor | None = None


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    total = (values * weights).sum(dim=dim)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return total / count


class GeometryHierarchicalEncoder(nn.Module):
    """Reusable observation-then-epoch encoder with strict mask validation."""

    def __init__(self, config: V2ModelConfig | V2AxisModelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.token_projection = nn.Sequential(
            nn.Linear(len(TOKEN_FEATURE_NAMES), d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        token_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * d_model,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(
            token_layer, num_layers=config.token_layers, enable_nested_tensor=False
        )
        # Three epoch descriptors plus period value/mask context (five values).
        self.epoch_context = nn.Sequential(
            nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        epoch_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * d_model,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.epoch_encoder = nn.TransformerEncoder(
            epoch_layer, num_layers=config.epoch_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(d_model)

    def encode(
        self,
        tokens: torch.Tensor,
        observation_mask: torch.Tensor,
        epoch_descriptors: torch.Tensor,
        epoch_mask: torch.Tensor,
        period_values: torch.Tensor,
        period_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[-1] != len(TOKEN_FEATURE_NAMES):
            raise ValueError("tokens must have shape [batch, epoch, observation, feature]")
        batch, epochs, observations, _ = tokens.shape
        if observation_mask.shape != (batch, epochs, observations):
            raise ValueError("observation_mask shape must match the token axes")
        if epoch_descriptors.shape != (batch, epochs, 3):
            raise ValueError("epoch_descriptors must have shape [batch, epoch, 3]")
        if epoch_mask.shape != (batch, epochs):
            raise ValueError("epoch_mask must have shape [batch, epoch]")
        if period_values.shape != (batch, 2) or period_mask.shape != (batch, 2):
            raise ValueError("period tensors must have shape [batch, 2]")
        if not bool(epoch_mask.any(dim=1).all()):
            raise ValueError("each object requires at least one valid epoch")
        if torch.any(observation_mask.sum(dim=-1)[epoch_mask] == 0):
            raise ValueError("valid epochs require at least one valid observation")

        if not self.config.phase_enabled:
            tokens = tokens.clone()
            tokens[..., PHASE_FEATURE_SLICE] = 0.0
        flat_tokens = tokens.reshape(batch * epochs, observations, -1)
        flat_observation_mask = observation_mask.reshape(batch * epochs, observations).bool()
        # Transformer attention cannot process an entirely masked sequence.
        # Invalid epoch slots are excluded by epoch_mask later, but receive one
        # deterministic zero token here to avoid NaNs contaminating arithmetic.
        empty_sequences = ~flat_observation_mask.any(dim=1)
        if bool(empty_sequences.any()):
            flat_tokens = flat_tokens.clone()
            flat_observation_mask = flat_observation_mask.clone()
            flat_tokens[empty_sequences, 0] = 0.0
            flat_observation_mask[empty_sequences, 0] = True
        encoded = self.token_encoder(
            self.token_projection(flat_tokens), src_key_padding_mask=~flat_observation_mask
        )
        pooled = _masked_mean(encoded, flat_observation_mask, dim=1).reshape(batch, epochs, -1)
        period_context = torch.cat(
            [period_values, period_mask.to(dtype=period_values.dtype)], dim=-1
        )
        period_context = torch.cat(
            [period_context, period_mask.any(dim=-1, keepdim=True).to(dtype=period_values.dtype)],
            dim=-1,
        )
        epoch_context = torch.cat(
            [epoch_descriptors, period_context.unsqueeze(1).expand(-1, epochs, -1)], dim=-1
        )
        encoded_epochs = self.epoch_encoder(
            pooled + self.epoch_context(epoch_context), src_key_padding_mask=~epoch_mask.bool()
        )
        return self.output_norm(_masked_mean(encoded_epochs, epoch_mask.bool(), dim=1))


class GeometryHierarchicalDensityModel(GeometryHierarchicalEncoder):
    """Historical experimental density head retained without checkpoint-key drift."""

    def __init__(self, config: V2ModelConfig = V2ModelConfig()) -> None:
        super().__init__(config)
        d_model = config.d_model
        self.density_head = nn.Linear(d_model, npix_for_nside(config.nside))
        self.nside16_head = (
            nn.Linear(d_model, npix_for_nside(16)) if config.hierarchical_density else None
        )
        self.nside8_head = (
            nn.Linear(d_model, npix_for_nside(8)) if config.hierarchical_density else None
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        observation_mask: torch.Tensor,
        epoch_descriptors: torch.Tensor,
        epoch_mask: torch.Tensor,
        period_values: torch.Tensor,
        period_mask: torch.Tensor,
    ) -> DensityModelOutput:
        object_embedding = self.encode(
            tokens,
            observation_mask,
            epoch_descriptors,
            epoch_mask,
            period_values,
            period_mask,
        )
        return DensityModelOutput(
            logits=self.density_head(object_embedding),
            raw_risk=self.risk_head(object_embedding).squeeze(-1),
            nside16_logits=None
            if self.nside16_head is None
            else self.nside16_head(object_embedding),
            nside8_logits=None if self.nside8_head is None else self.nside8_head(object_embedding),
        )


class GeometryHierarchicalAxisModel(GeometryHierarchicalEncoder):
    """V2-plain: end-to-end encoder with one shared unrestricted K=3 head."""

    def __init__(self, config: V2AxisModelConfig = V2AxisModelConfig()) -> None:
        if config.head_type != "plain":
            raise ValueError("GeometryHierarchicalAxisModel requires head_type='plain'")
        super().__init__(config)
        self.axis_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.candidate_count * 3),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        observation_mask: torch.Tensor,
        epoch_descriptors: torch.Tensor,
        epoch_mask: torch.Tensor,
        period_values: torch.Tensor,
        period_mask: torch.Tensor,
    ) -> AxisModelOutput:
        embedding = self.encode(
            tokens,
            observation_mask,
            epoch_descriptors,
            epoch_mask,
            period_values,
            period_mask,
        )
        raw_axes = self.axis_head(embedding).reshape(-1, self.config.candidate_count, 3)
        return AxisModelOutput(axes=F.normalize(raw_axes, dim=-1))


def _tangent_basis(anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct stable right-handed orthonormal bases at unit anchors."""
    z_reference = torch.tensor([0.0, 0.0, 1.0], dtype=anchors.dtype, device=anchors.device)
    x_reference = torch.tensor([1.0, 0.0, 0.0], dtype=anchors.dtype, device=anchors.device)
    references = torch.where((anchors[:, 2].abs() > 0.9)[:, None], x_reference, z_reference)
    first = F.normalize(
        references - (references * anchors).sum(dim=1, keepdim=True) * anchors, dim=1
    )
    second = F.normalize(torch.linalg.cross(anchors, first, dim=1), dim=1)
    return first, second


def bounded_tangent_exponential_map(
    anchors: torch.Tensor, tangent_coordinates: torch.Tensor, maximum_angle_deg: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map two tangent coordinates per anchor to unit axes within a hard cap."""
    if anchors.ndim != 2 or anchors.shape[-1] != 3:
        raise ValueError("anchors must have shape [candidate, 3]")
    if tangent_coordinates.ndim != 3 or tangent_coordinates.shape[1:] != (anchors.shape[0], 2):
        raise ValueError("tangent_coordinates must have shape [batch, candidate, 2]")
    if not math.isfinite(maximum_angle_deg) or not 0 < maximum_angle_deg <= 90:
        raise ValueError("maximum_angle_deg must lie in (0, 90]")
    unit_anchors = F.normalize(anchors, dim=-1)
    first, second = _tangent_basis(unit_anchors)
    raw_squared_norm = tangent_coordinates.square().sum(dim=-1, keepdim=True)
    raw_norm = torch.sqrt(raw_squared_norm)
    stable_ratio = torch.where(
        raw_squared_norm > 1.0e-12,
        torch.tanh(raw_norm) / raw_norm.clamp_min(1.0e-12),
        1.0 - raw_squared_norm / 3.0,
    )
    maximum_angle_rad = math.radians(maximum_angle_deg)
    bounded_coordinates = tangent_coordinates * (maximum_angle_rad * stable_ratio)
    tangent = (
        bounded_coordinates[..., 0, None] * first[None, :, :]
        + bounded_coordinates[..., 1, None] * second[None, :, :]
    )
    angles = torch.linalg.vector_norm(tangent, dim=-1)
    # torch.sinc(x/pi) is sin(x)/x and has the correct analytic value at zero.
    mapped = (
        torch.cos(angles)[..., None] * unit_anchors[None, :, :]
        + torch.sinc(angles / math.pi)[..., None] * tangent
    )
    return F.normalize(mapped, dim=-1), angles


class GeometryHierarchicalResidualAxisModel(GeometryHierarchicalEncoder):
    """V2-residual: bounded tangent corrections around a train-only atlas."""

    def __init__(self, config: V2AxisModelConfig, atlas: AxisAtlas) -> None:
        if config.head_type != "residual":
            raise ValueError("residual model requires head_type='residual'")
        if config.atlas_sha256 != atlas.atlas_sha256:
            raise ValueError("axis model configuration does not match the supplied atlas hash")
        super().__init__(config)
        axes = torch.as_tensor(atlas.axes, dtype=torch.float32)
        if axes.shape != (config.candidate_count, 3):
            raise ValueError("training atlas must contain exactly three axes")
        self.register_buffer("atlas_axes", F.normalize(axes, dim=-1))
        self.residual_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.candidate_count * 2),
        )
        final = self.residual_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        observation_mask: torch.Tensor,
        epoch_descriptors: torch.Tensor,
        epoch_mask: torch.Tensor,
        period_values: torch.Tensor,
        period_mask: torch.Tensor,
    ) -> AxisModelOutput:
        embedding = self.encode(
            tokens,
            observation_mask,
            epoch_descriptors,
            epoch_mask,
            period_values,
            period_mask,
        )
        coordinates = self.residual_head(embedding).reshape(-1, self.config.candidate_count, 2)
        axes, angles = bounded_tangent_exponential_map(
            self.atlas_axes, coordinates, self.config.maximum_residual_deg
        )
        return AxisModelOutput(axes=axes, residual_angles_rad=angles)


def residual_angle_regularization(output: AxisModelOutput) -> torch.Tensor:
    """Mean squared tangent angle for the residual head only."""
    if output.residual_angles_rad is None:
        return output.axes.sum() * 0.0
    return output.residual_angles_rad.square().mean()


def multi_solution_density_nll(
    logits: torch.Tensor, target_pixels: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    """Negative log mass of any explicit source solution for each object."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, pixel]")
    if target_pixels.ndim != 2 or target_mask.shape != target_pixels.shape:
        raise ValueError("target_pixels and target_mask must share shape [batch, solution]")
    if target_pixels.shape[0] != logits.shape[0]:
        raise ValueError("target batch size must match logits")
    if not bool(target_mask.any(dim=1).all()):
        raise ValueError("each object requires at least one explicit source solution")
    if torch.any(target_pixels[target_mask] < 0) or torch.any(
        target_pixels[target_mask] >= logits.shape[1]
    ):
        raise ValueError("target pixel is outside the configured HEALPix grid")
    # Explicit source solutions define a *set* of valid pixels. Two nearby
    # solutions may quantize to the same HEALPix cell; gathering them as a list
    # would count that cell twice and create artificial probability mass.
    membership_count = torch.zeros_like(logits, dtype=torch.int32)
    membership_count.scatter_add_(
        1,
        target_pixels.clamp(min=0, max=logits.shape[1] - 1),
        target_mask.to(dtype=torch.int32),
    )
    source_pixel_union = membership_count > 0
    log_probability = torch.log_softmax(logits, dim=-1)
    selected = log_probability.masked_fill(~source_pixel_union, float("-inf"))
    return -torch.logsumexp(selected, dim=1).mean()


def hierarchical_multi_solution_density_nll(
    output: DensityModelOutput, target_pixels: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    """Fixed multi-resolution spherical supervision for the canonical grid.

    HEALPix nested indices map a child at nside 32 to its nside 16 and nside 8
    parents by integer division by four and sixteen respectively.  The final
    public density remains the nside-32 head; coarse heads improve sample
    efficiency without changing the deployed prediction interface.
    """
    fine = multi_solution_density_nll(output.logits, target_pixels, target_mask)
    if output.nside16_logits is None or output.nside8_logits is None:
        return fine
    mid = multi_solution_density_nll(output.nside16_logits, target_pixels // 4, target_mask)
    coarse = multi_solution_density_nll(output.nside8_logits, target_pixels // 16, target_mask)
    return 0.40 * fine + 0.35 * mid + 0.25 * coarse
