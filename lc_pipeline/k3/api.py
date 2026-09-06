"""Deployable fixed-period K3 inference API with no oracle inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..v2.preprocessing import KnownPeriod, ObservationEpoch
from .calibration import K3Calibration, normalized_entropy
from .config import K3ScoreModelConfig, K3TokenizerConfig
from .grid import compatibility_mass, extract_axial_modes
from .inference import K3InferenceError, refine_axes, score_axial_grid
from .model import CandidateConditionedScorer, K3ModelError
from .schema import K3AxialPrediction, K3AxisCandidate, K3PredictionStatus
from .tokenizer import (
    K3_TOKENIZER_SCHEMA_SHA256,
    K3TokenizationError,
    K3TokenizedObject,
    pad_tokenized_objects,
    tokenize_epochs,
)


@dataclass(frozen=True)
class K3ModelBundle:
    model: CandidateConditionedScorer
    model_id: str
    calibration: K3Calibration
    model_config: K3ScoreModelConfig
    tokenizer_config: K3TokenizerConfig = K3TokenizerConfig()
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if self.model.config != self.model_config:
            raise ValueError("K3 bundle model/config mismatch")
        if self.checkpoint_sha256 is not None and (
            len(self.checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")


class DeLPHIK3Predictor:
    """Candidate scorer wrapper returning three axes or a structured failure."""

    def __init__(self, bundle: K3ModelBundle, *, device: str | torch.device = "cpu") -> None:
        self.bundle = bundle
        self.device = torch.device(device)
        self.bundle.model.to(self.device).eval()

    def _inputs(self, tokenized: K3TokenizedObject) -> dict[str, torch.Tensor]:
        arrays = pad_tokenized_objects((tokenized,))
        return {
            "phase_features": torch.from_numpy(arrays["phase_features"]).to(self.device),
            "phase_mask": torch.from_numpy(arrays["phase_mask"]).to(self.device),
            "geometry_features": torch.from_numpy(arrays["geometry_features"]).to(self.device),
            "epoch_features": torch.from_numpy(arrays["epoch_features"]).to(self.device),
            "epoch_mask": torch.from_numpy(arrays["epoch_mask"]).to(self.device),
        }

    def score_grid(
        self, epochs: Sequence[ObservationEpoch], *, known_period: KnownPeriod
    ) -> tuple[K3TokenizedObject, np.ndarray]:
        tokenized = tokenize_epochs(
            epochs, known_period=known_period, config=self.bundle.tokenizer_config
        )
        scores = score_axial_grid(self.bundle.model, self._inputs(tokenized))
        return tokenized, scores

    def predict(
        self, epochs: Sequence[ObservationEpoch], *, known_period: KnownPeriod | None
    ) -> K3AxialPrediction:
        provenance = known_period.provenance if isinstance(known_period, KnownPeriod) else None
        try:
            tokenized, scores = self.score_grid(epochs, known_period=known_period)  # type: ignore[arg-type]
            mass = compatibility_mass(scores, temperature=self.bundle.calibration.temperature)
            modes = extract_axial_modes(
                scores,
                candidate_count=self.bundle.model_config.candidate_count,
                minimum_separation_deg=self.bundle.model_config.nms_separation_deg,
            )
            starts = np.asarray([mode.axis_xyz for mode in modes], dtype=np.float64)
            refined_axes, refined_scores = refine_axes(
                self.bundle.model,
                self._inputs(tokenized),
                starts,
                config=self.bundle.model_config,
            )
            order = np.lexsort(
                (np.asarray([mode.grid_index for mode in modes]), -refined_scores)
            )
            axes = tuple(
                K3AxisCandidate(
                    axis_xyz=tuple(float(value) for value in refined_axes[index]),
                    score_logit=float(refined_scores[index]),
                    compatibility_mass=float(mass[modes[index].grid_index]),
                    cone90_deg=self.bundle.calibration.cone90_deg,
                    cone95_deg=self.bundle.calibration.cone95_deg,
                    grid_index=modes[index].grid_index,
                )
                for index in order
            )
            raw_risk = normalized_entropy(mass)
            risk = self.bundle.calibration.calibrated_risk_deg(raw_risk)
            if raw_risk >= self.bundle.calibration.uninformative_entropy:
                status = K3PredictionStatus.UNINFORMATIVE
            elif risk >= self.bundle.calibration.abstain_risk_deg:
                status = K3PredictionStatus.ABSTAIN
            elif risk >= self.bundle.calibration.ambiguous_risk_deg:
                status = K3PredictionStatus.AMBIGUOUS
            else:
                status = K3PredictionStatus.OK
            return K3AxialPrediction(
                status=status,
                axes=axes,
                risk_deg=risk,
                model_id=self.bundle.model_id,
                period_provenance=tokenized.period_provenance,
                tokenizer_schema_sha256=K3_TOKENIZER_SCHEMA_SHA256,
            )
        except (K3TokenizationError, K3ModelError, K3InferenceError) as exc:
            return K3AxialPrediction.failed(
                model_id=self.bundle.model_id,
                tokenizer_schema_sha256=K3_TOKENIZER_SCHEMA_SHA256,
                reason=str(exc),
                period_provenance=provenance,
            )
