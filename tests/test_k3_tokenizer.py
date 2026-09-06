"""Tests for global-phase, source-epoch-preserving K3 tokenization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lc_pipeline.k3.tokenizer import K3TokenizationError, pad_tokenized_objects, tokenize_epochs
from lc_pipeline.v2.preprocessing import KnownPeriod, Observation, ObservationEpoch


def _epoch(epoch_id: str, start: float, scale: float, count: int = 9) -> ObservationEpoch:
    observations = tuple(
        Observation(
            time_jd=start + index / 24.0,
            relative_brightness=scale * math.exp(0.15 * math.sin(index)),
            measured_error=0.01 * scale,
            sun_asteroid_ecliptic_j2000_au=(1.5, 0.1 * index, 0.2),
            observer_asteroid_ecliptic_j2000_au=(0.2, 1.2, 0.05 * index + 0.1),
        )
        for index in range(count)
    )
    return ObservationEpoch(epoch_id, observations)


def test_tokenizer_preserves_epoch_boundaries_counts_and_amplitude() -> None:
    epochs = (_epoch("late", 2450100.0, 1e6), _epoch("early", 2450000.0, 1.0))
    tokenized = tokenize_epochs(
        epochs, known_period=KnownPeriod(8.0, "external-fixed-period")
    )
    assert tokenized.epoch_ids == ("late", "early")
    assert tokenized.observation_counts == (9, 9)
    assert tokenized.observation_time_origin_jd == 2450000.0
    assert int(np.sum(np.expm1(tokenized.phase_features[..., 2]).round())) == 18
    # Arbitrary epoch flux scales are centered, but within-epoch log amplitude is retained.
    assert np.allclose(tokenized.epoch_features[:, 16], tokenized.epoch_features[0, 16])
    assert np.max(np.abs(tokenized.phase_features[..., 0])) < 0.3


def test_object_global_phase_reference_differs_across_separated_epochs() -> None:
    first = _epoch("a", 2450000.0, 1.0, count=7)
    second = _epoch("b", 2450000.125, 1.0, count=7)
    tokenized = tokenize_epochs(
        (first, second), known_period=KnownPeriod(12.0, "external-fixed-period")
    )
    occupied_first = np.flatnonzero(tokenized.phase_mask[0])
    occupied_second = np.flatnonzero(tokenized.phase_mask[1])
    assert occupied_first[0] != occupied_second[0]


def test_tokenizer_requires_known_period_and_does_not_reconstruct_epochs() -> None:
    with pytest.raises(K3TokenizationError, match="KnownPeriod"):
        tokenize_epochs((_epoch("a", 1.0, 1.0),), known_period=None)  # type: ignore[arg-type]
    with pytest.raises(K3TokenizationError, match="at least 2"):
        tokenize_epochs(
            (_epoch("a", 1.0, 1.0, count=1),),
            known_period=KnownPeriod(5.0, "external-fixed-period"),
        )


def test_collation_pads_epochs_without_repeating_data() -> None:
    period = KnownPeriod(8.0, "external-fixed-period")
    one = tokenize_epochs((_epoch("a", 2450000.0, 1.0),), known_period=period)
    two = tokenize_epochs(
        (_epoch("b", 2450000.0, 1.0), _epoch("c", 2450010.0, 1.0)), known_period=period
    )
    batch = pad_tokenized_objects((one, two))
    assert batch["epoch_mask"].tolist() == [[True, False], [True, True]]
    assert not np.any(batch["phase_mask"][0, 1])
    assert np.all(batch["phase_features"][0, 1] == 0)
