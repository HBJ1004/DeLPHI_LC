"""Validated configuration for the prospective K=3 scorer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

K3_DEVELOPMENT_SEEDS = (17, 42, 137)
K3_OOF_SEEDS = (17, 42, 137, 777, 2027)


class K3ConfigError(ValueError):
    """Raised when a K3 configuration violates the locked contract."""


@dataclass(frozen=True)
class K3TokenizerConfig:
    phase_bins: int = 64
    fourier_harmonics: int = 8
    minimum_epoch_observations: int = 2

    def __post_init__(self) -> None:
        if self.phase_bins != 64:
            raise K3ConfigError("the locked tokenizer requires exactly 64 phase bins")
        if self.fourier_harmonics != 8:
            raise K3ConfigError("the locked tokenizer requires harmonics 1 through 8")
        if self.minimum_epoch_observations != 2:
            raise K3ConfigError("the locked tokenizer requires two observations per epoch")


@dataclass(frozen=True)
class K3ScoreModelConfig:
    """Architecture and deterministic inference grid for a K3 scorer."""

    phase_feature_count: int = 6
    epoch_feature_count: int = 22
    hidden_dim: int = 128
    convolution_blocks: int = 3
    set_encoder_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    nside: int = 32
    candidate_count: int = 3
    score_chunk_size: int = 512
    nms_separation_deg: float = 15.0
    refinement_steps: int = 8
    refinement_learning_rate: float = 0.08
    refinement_max_displacement_deg: float = 5.0

    def __post_init__(self) -> None:
        integer_positive = (
            self.phase_feature_count,
            self.epoch_feature_count,
            self.hidden_dim,
            self.convolution_blocks,
            self.set_encoder_layers,
            self.attention_heads,
            self.nside,
            self.candidate_count,
            self.score_chunk_size,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integer_positive):
            raise K3ConfigError("model dimensions must be positive integers")
        if self.hidden_dim % self.attention_heads:
            raise K3ConfigError("hidden_dim must be divisible by attention_heads")
        if self.nside != 32 or self.candidate_count != 3:
            raise K3ConfigError("the locked inference contract requires nside=32 and K=3")
        finite = (
            self.dropout,
            self.nms_separation_deg,
            self.refinement_learning_rate,
            self.refinement_max_displacement_deg,
        )
        if not all(math.isfinite(value) for value in finite):
            raise K3ConfigError("floating-point configuration values must be finite")
        if not 0 <= self.dropout < 1:
            raise K3ConfigError("dropout must lie in [0, 1)")
        if not 0 < self.nms_separation_deg < 90:
            raise K3ConfigError("NMS separation must lie in (0, 90) degrees")
        if self.refinement_steps != 8 or self.refinement_max_displacement_deg != 5.0:
            raise K3ConfigError("the locked refinement uses eight steps and a five-degree cap")
        if self.refinement_learning_rate <= 0:
            raise K3ConfigError("refinement learning rate must be positive")

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class K3TrainingConfig:
    seed: int
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    synthetic_max_epochs: int = 100
    synthetic_patience: int = 15
    real_max_epochs: int = 30
    real_patience: int = 8
    uniform_negatives: int = 48
    hard_negatives: int = 16
    hard_negative_min_deg: float = 5.0
    hard_negative_max_deg: float = 45.0
    negative_exclusion_deg: float = 10.0
    rank_loss_weight: float = 0.25
    encoder_finetune_lr_multiplier: float = 0.25
    synthetic_to_real_ratio: tuple[int, int] = (3, 1)
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.seed not in K3_OOF_SEEDS:
            raise K3ConfigError(f"seed must be one of {K3_OOF_SEEDS}")
        integers = (
            self.batch_size,
            self.synthetic_max_epochs,
            self.synthetic_patience,
            self.real_max_epochs,
            self.real_patience,
            self.uniform_negatives,
            self.hard_negatives,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integers):
            raise K3ConfigError("training counts must be positive integers")
        if self.synthetic_patience >= self.synthetic_max_epochs or self.real_patience >= self.real_max_epochs:
            raise K3ConfigError("patience must be smaller than the corresponding epoch limit")
        positives = (
            self.learning_rate,
            self.gradient_clip_norm,
            self.hard_negative_min_deg,
            self.hard_negative_max_deg,
            self.negative_exclusion_deg,
            self.encoder_finetune_lr_multiplier,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positives):
            raise K3ConfigError("training scales must be finite and positive")
        if not 0 <= self.weight_decay or not math.isfinite(self.weight_decay):
            raise K3ConfigError("weight_decay must be finite and nonnegative")
        if not self.hard_negative_min_deg < self.negative_exclusion_deg < self.hard_negative_max_deg <= 90:
            raise K3ConfigError("hard-negative and exclusion angles are inconsistent")
        if not math.isfinite(self.rank_loss_weight) or self.rank_loss_weight < 0:
            raise K3ConfigError("rank_loss_weight must be finite and nonnegative")
        if self.synthetic_to_real_ratio != (3, 1):
            raise K3ConfigError("the locked real fine-tune ratio is 3 synthetic to 1 real")
        if self.num_workers != 0:
            raise K3ConfigError("definitive deterministic runs require num_workers=0")

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)
