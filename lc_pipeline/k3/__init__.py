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
    "load_predictor",
    "read_observations",
]


def __getattr__(name: str):
    """Load prediction helpers when requested so module CLIs can start normally."""
    if name == "load_predictor":
        from .bundle import load_predictor

        return load_predictor
    if name == "read_observations":
        from .predict import read_observations

        return read_observations
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
