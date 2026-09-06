"""Controlled V1 and amplitude-summary baselines for the axial comparison."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..inference.tokenizer import N_FEATURES as V1_FEATURE_COUNT
from ..inference.tokenizer import tokenize_lightcurve
from ..models.geo_hier_k3_transformer import GeoHierK3Transformer
from ..physics.axial import axial_separation_penalty
from .data import PreparedObject, canonical_json
from .preprocessing import ObservationEpoch

V1_TOKENIZER_SCHEMA = "delphi.v1-faithful-tokenizer.v1"
V1_TOKENIZER_SCHEMA_SHA256 = hashlib.sha256(
    canonical_json(
        {
            "schema": V1_TOKENIZER_SCHEMA,
            "features": [
                "window_time_0_1",
                "window_delta_over_mean_abs_delta",
                "window_brightness_mad_normalized",
                "six_zero_geometry_slots",
                "log1p_window_local_rotation_count",
                "normalized_log_observation_inferred_period",
                "window_local_phase_sin",
                "window_local_phase_cos",
            ],
            "gap_days": 1.0,
            "maximum_windows": 8,
            "maximum_tokens_per_window": 256,
            "short_window_minimum_observations": 3,
            "padding": "repeat_largest_window_until_eight_then_zero_token_padding",
        }
    ).encode("utf-8")
).hexdigest()

AMPLITUDE_FEATURE_NAMES = (
    "median_epoch_peak_to_peak",
    "mean_epoch_peak_to_peak",
    "std_epoch_peak_to_peak",
    "maximum_epoch_peak_to_peak",
    "minimum_epoch_peak_to_peak",
    "range_epoch_peak_to_peak",
    "median_epoch_brightness_std",
    "mean_epoch_brightness_std",
    "std_epoch_brightness_std",
)


class BaselineContractError(ValueError):
    """Raised when a baseline could differ from its declared control."""


@dataclass(frozen=True)
class LegacyAxisObject:
    object_id: str
    tokens: np.ndarray
    mask: np.ndarray
    target_vectors: np.ndarray
    period_hours_targets: np.ndarray
    inferred_period_hours: float | None

    def validate(self) -> None:
        if not self.object_id:
            raise BaselineContractError("legacy baseline object ID is required")
        if (
            self.tokens.ndim != 3
            or self.tokens.shape[0] != 8
            or self.tokens.shape[2] != V1_FEATURE_COUNT
        ):
            raise BaselineContractError("legacy tokens must have shape [8, observation, 13]")
        if self.mask.shape != self.tokens.shape[:2]:
            raise BaselineContractError("legacy mask must align with tokens")
        if (
            self.target_vectors.ndim != 2
            or self.target_vectors.shape[1] != 3
            or not self.target_vectors.size
        ):
            raise BaselineContractError("legacy target vectors must have shape [solution, 3]")
        if self.period_hours_targets.shape != (self.target_vectors.shape[0],):
            raise BaselineContractError("legacy period targets must align with source axes")
        arrays = (self.tokens, self.mask, self.target_vectors, self.period_hours_targets)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise BaselineContractError("legacy baseline arrays must be finite")


@dataclass(frozen=True)
class AmplitudeAxisObject:
    object_id: str
    features: np.ndarray
    target_vectors: np.ndarray
    period_hours_targets: np.ndarray

    def validate(self) -> None:
        if not self.object_id:
            raise BaselineContractError("amplitude baseline object ID is required")
        if self.features.shape != (len(AMPLITUDE_FEATURE_NAMES),):
            raise BaselineContractError("amplitude features must have shape [9]")
        if (
            self.target_vectors.ndim != 2
            or self.target_vectors.shape[1] != 3
            or not self.target_vectors.size
        ):
            raise BaselineContractError("amplitude targets must have shape [solution, 3]")
        if self.period_hours_targets.shape != (self.target_vectors.shape[0],):
            raise BaselineContractError("amplitude period targets must align with source axes")
        if not all(
            np.all(np.isfinite(value))
            for value in (self.features, self.target_vectors, self.period_hours_targets)
        ):
            raise BaselineContractError("amplitude baseline arrays must be finite")


def _epochs_as_damit_arrays(epochs: Sequence[ObservationEpoch]) -> list[np.ndarray]:
    if not epochs:
        raise BaselineContractError("baseline preprocessing requires source epochs")
    arrays: list[np.ndarray] = []
    for epoch in epochs:
        rows = [
            (
                observation.time_jd,
                observation.relative_brightness,
                *observation.sun_asteroid_ecliptic_j2000_au,
                *observation.observer_asteroid_ecliptic_j2000_au,
            )
            for observation in epoch.observations
        ]
        arrays.append(np.asarray(rows, dtype=np.float64))
    return arrays


def observation_inferred_period(prepared: PreparedObject) -> float | None:
    """Read only the deployable period estimate already frozen in the V2 cache."""
    prepared.validate()
    if not bool(prepared.period_mask[0]):
        return None
    value = float(prepared.period_values[0])
    if not math.isfinite(value) or value <= 0:
        raise BaselineContractError("cached inferred period must be finite and positive")
    return value


def prepare_v1_baseline_object(
    prepared: PreparedObject, epochs: Sequence[ObservationEpoch]
) -> LegacyAxisObject:
    """Apply the historical tokenizer while reusing V2's inferred period only."""
    period_hours = observation_inferred_period(prepared)
    tokens, mask = tokenize_lightcurve(
        _epochs_as_damit_arrays(epochs),
        period_hours=period_hours,
        n_windows=8,
        tokens_per_window=256,
        use_geometry=False,
    )
    result = LegacyAxisObject(
        object_id=prepared.object_id,
        tokens=tokens,
        mask=mask,
        target_vectors=np.asarray(prepared.target_vectors, dtype=np.float64).copy(),
        period_hours_targets=np.asarray(prepared.period_hours_targets, dtype=np.float64).copy(),
        inferred_period_hours=period_hours,
    )
    result.validate()
    return result


def extract_amplitude_features(epochs: Sequence[ObservationEpoch]) -> np.ndarray:
    """Reproduce the nine-feature morphology baseline from the prior audit."""
    if not epochs:
        raise BaselineContractError("amplitude features require source epochs")
    amplitudes: list[float] = []
    standard_deviations: list[float] = []
    for epoch in epochs:
        brightness = np.asarray(
            [observation.relative_brightness for observation in epoch.observations],
            dtype=np.float64,
        )
        if not brightness.size or not np.all(np.isfinite(brightness)):
            raise BaselineContractError("epoch brightness must be nonempty and finite")
        centered = brightness - np.median(brightness)
        amplitudes.append(float(np.ptp(centered)))
        standard_deviations.append(float(np.std(centered)))
    amplitude = np.asarray(amplitudes, dtype=np.float64)
    brightness_std = np.asarray(standard_deviations, dtype=np.float64)
    return np.asarray(
        [
            np.median(amplitude),
            np.mean(amplitude),
            np.std(amplitude),
            np.max(amplitude),
            np.min(amplitude),
            np.max(amplitude) - np.min(amplitude),
            np.median(brightness_std),
            np.mean(brightness_std),
            np.std(brightness_std),
        ],
        dtype=np.float32,
    )


def prepare_amplitude_baseline_object(
    prepared: PreparedObject, epochs: Sequence[ObservationEpoch]
) -> AmplitudeAxisObject:
    result = AmplitudeAxisObject(
        object_id=prepared.object_id,
        features=extract_amplitude_features(epochs),
        target_vectors=np.asarray(prepared.target_vectors, dtype=np.float64).copy(),
        period_hours_targets=np.asarray(prepared.period_hours_targets, dtype=np.float64).copy(),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    train_object_ids_sha256: str

    @classmethod
    def fit(
        cls,
        values: Sequence[AmplitudeAxisObject],
        *,
        partition_role: str,
        forbidden_object_ids: Sequence[str] = (),
    ) -> "FeatureStandardizer":
        if partition_role != "train":
            raise BaselineContractError(
                "feature scaling may only be fit on a partition named 'train'"
            )
        if not values:
            raise BaselineContractError("feature scaling requires training objects")
        sorted_values = sorted(values, key=lambda item: item.object_id)
        identifiers = [item.object_id for item in sorted_values]
        if len(identifiers) != len(set(identifiers)):
            raise BaselineContractError("feature scaling requires unique object IDs")
        if set(identifiers) & set(forbidden_object_ids):
            raise BaselineContractError("feature-scaling training IDs overlap evaluation IDs")
        for value in sorted_values:
            value.validate()
        features = np.stack([value.features for value in sorted_values]).astype(np.float64)
        scale = np.std(features, axis=0)
        scale = np.where(scale < 1.0e-6, 1.0, scale)
        return cls(
            mean=tuple(float(item) for item in np.mean(features, axis=0)),
            scale=tuple(float(item) for item in scale),
            train_object_ids_sha256=hashlib.sha256(
                canonical_json(identifiers).encode("utf-8")
            ).hexdigest(),
        )

    def transform(self, values: Sequence[AmplitudeAxisObject]) -> np.ndarray:
        if not values:
            raise BaselineContractError("cannot transform an empty object collection")
        for value in values:
            value.validate()
        features = np.stack([value.features for value in values]).astype(np.float32)
        return (features - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.scale, dtype=np.float32
        )


class AmplitudeFeatureMLP(nn.Module):
    """Published nine-summary-feature baseline with three unit-axis outputs."""

    def __init__(self, n_features: int = 9, hidden: int = 128, candidates: int = 3) -> None:
        super().__init__()
        if n_features != 9 or candidates != 3 or hidden <= 0:
            raise BaselineContractError("amplitude comparison fixes 9 inputs and 3 candidates")
        self.candidates = candidates
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, candidates * 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 9:
            raise ValueError("amplitude features must have shape [batch, 9]")
        return F.normalize(self.network(features).reshape(-1, self.candidates, 3), dim=-1)


def make_v1_model(*, seed: int) -> GeoHierK3Transformer:
    """Instantiate the historical architecture after deterministic seeding."""
    torch.manual_seed(seed)
    return GeoHierK3Transformer(
        d_model=128,
        n_heads=4,
        n_layers=4,
        n_feature_input=13,
        include_quality_head=False,
        dropout=0.1,
    )


def historical_floor_axial_softmin_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    tau_deg: float = 5.0,
) -> torch.Tensor:
    """Faithfully preserve V1's clamped antipode-angle oracle loss.

    This intentionally retains the ``acos(sqrt(clamp(dot,-.99,.99)^2))``
    floor. It exists only as a controlled historical arm and must not be used
    by V2 or interpreted as a precise evaluation metric.
    """
    if predictions.ndim != 3 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [batch, candidate, 3]")
    if targets.ndim != 3 or targets.shape[-1] != 3 or target_mask.shape != targets.shape[:2]:
        raise ValueError("targets and target_mask must have shapes [B,S,3] and [B,S]")
    if predictions.shape[0] != targets.shape[0] or not bool(target_mask.any(dim=1).all()):
        raise ValueError("every prediction batch item requires a source solution")
    if target_mask.dtype is not torch.bool:
        raise TypeError("target_mask must be boolean")
    if not math.isfinite(tau_deg) or tau_deg <= 0:
        raise ValueError("tau_deg must be finite and positive")
    safe_targets = torch.where(target_mask[..., None], targets, torch.ones_like(targets))
    unit_predictions = F.normalize(predictions, dim=-1)
    unit_targets = F.normalize(safe_targets, dim=-1)
    dots = torch.einsum("bkc,bsc->bks", unit_predictions, unit_targets)
    clamped = torch.clamp(dots, -0.99, 0.99)
    absolute_cosine = torch.sqrt(torch.clamp(clamped.square(), 1.0e-6, 1.0))
    angles = torch.rad2deg(torch.acos(absolute_cosine))
    angles = angles.masked_fill(~target_mask[:, None, :], float("inf"))
    per_candidate = -tau_deg * torch.logsumexp(-angles / tau_deg, dim=2)
    return (-tau_deg * torch.logsumexp(-per_candidate / tau_deg, dim=1)).mean()


def v1_training_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    mode: Literal["faithful", "corrected"],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply exactly the predeclared loss for either controlled V1 arm."""
    from ..physics.axial import masked_axial_softmin_loss

    if mode == "faithful":
        oracle = historical_floor_axial_softmin_loss(predictions, targets, target_mask)
    elif mode == "corrected":
        oracle = masked_axial_softmin_loss(predictions, targets, target_mask)
    else:
        raise ValueError("V1 loss mode must be 'faithful' or 'corrected'")
    separation = axial_separation_penalty(predictions)
    return oracle + 0.1 * separation, {
        "axial_oracle_loss": oracle,
        "separation_penalty": separation,
    }


__all__ = [
    "AMPLITUDE_FEATURE_NAMES",
    "V1_TOKENIZER_SCHEMA",
    "V1_TOKENIZER_SCHEMA_SHA256",
    "AmplitudeAxisObject",
    "AmplitudeFeatureMLP",
    "BaselineContractError",
    "FeatureStandardizer",
    "LegacyAxisObject",
    "extract_amplitude_features",
    "historical_floor_axial_softmin_loss",
    "make_v1_model",
    "observation_inferred_period",
    "prepare_amplitude_baseline_object",
    "prepare_v1_baseline_object",
    "v1_training_loss",
]
