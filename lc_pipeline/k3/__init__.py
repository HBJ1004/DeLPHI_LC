"""K3 candidate scoring, source-epoch tokenization, and synthetic rendering."""

from .config import K3ScoreModelConfig, K3TokenizerConfig, K3TrainingConfig
from .damit import DAMITLoadError, DAMITObject, load_damit_object
from .damit_synthetic import (
    discover_damit_donors,
    generate_damit_synthetic_shards,
    partition_damit_donors,
)
from .schema import K3AxialPrediction, K3AxisCandidate, K3PredictionStatus
from .synthetic import ResidualNoiseBank, verify_synthetic_shard

__all__ = [
    "K3AxialPrediction",
    "K3AxisCandidate",
    "K3PredictionStatus",
    "K3ScoreModelConfig",
    "K3TokenizerConfig",
    "K3TrainingConfig",
    "DAMITLoadError",
    "DAMITObject",
    "load_damit_object",
    "generate_damit_synthetic_shards",
    "discover_damit_donors",
    "partition_damit_donors",
    "ResidualNoiseBank",
    "verify_synthetic_shard",
]
