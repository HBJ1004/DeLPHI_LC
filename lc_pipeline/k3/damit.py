"""Minimal, auditable DAMIT dump adapter for the definitive K3 run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..v2.preprocessing import Observation, ObservationEpoch
from .renderer import ConvexFacets, parse_damit_shape


class DAMITLoadError(ValueError):
    """Raised when a local DAMIT object is incomplete or malformed."""


@dataclass(frozen=True)
class DAMITObject:
    object_id: str
    epochs: tuple[ObservationEpoch, ...]
    shape: ConvexFacets
    model_id: str
    period_hours: float
    pole_ecliptic: tuple[float, float, float]


def _pole(lambda_deg: float, beta_deg: float) -> tuple[float, float, float]:
    lam, bet = np.radians([lambda_deg, beta_deg])
    return (float(np.cos(bet) * np.cos(lam)), float(np.cos(bet) * np.sin(lam)), float(np.sin(bet)))


def _read_spin(path: Path) -> tuple[float, float, tuple[float, float, float]]:
    try:
        lines = [line.split() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
        lam, bet, period = map(float, lines[0][:3])
    except (OSError, UnicodeDecodeError, IndexError, ValueError) as exc:
        raise DAMITLoadError(f"cannot parse spin file {path}: {exc}") from exc
    return period, lam, _pole(lam, bet)


def _read_lc(path: Path) -> tuple[ObservationEpoch, ...]:
    try:
        tokens = path.read_text(encoding="ascii").split()
        cursor, count = 0, int(tokens[0])
        cursor = 1
        epochs: list[ObservationEpoch] = []
        for epoch_index in range(count):
            n_points, _calibrated = int(tokens[cursor]), int(tokens[cursor + 1])
            cursor += 2
            raw = np.asarray(tokens[cursor : cursor + 8 * n_points], dtype=np.float64).reshape(n_points, 8)
            cursor += 8 * n_points
            observations = tuple(
                Observation(
                    time_jd=float(row[0]),
                    relative_brightness=float(row[1]),
                    sun_asteroid_ecliptic_j2000_au=tuple(row[2:5]),
                    observer_asteroid_ecliptic_j2000_au=tuple(row[5:8]),
                )
                for row in raw
            )
            epochs.append(ObservationEpoch(f"damit-epoch-{epoch_index:04d}", observations))
    except (OSError, UnicodeDecodeError, IndexError, ValueError) as exc:
        raise DAMITLoadError(f"cannot parse lightcurve file {path}: {exc}") from exc
    if cursor != len(tokens):
        raise DAMITLoadError(f"trailing tokens in lightcurve file {path}")
    return tuple(epochs)


def load_damit_object(dump_root: str | Path, object_id: str, *, model_id: str | int | None = None) -> DAMITObject:
    """Load one object using only local files; never downloads or mutates data."""
    root = Path(dump_root)
    object_dir = root / "files" / object_id
    if not object_dir.is_dir():
        raise DAMITLoadError(f"missing DAMIT object directory: {object_dir}")
    models = sorted(object_dir.glob("model_*/"), key=lambda path: path.name)
    if model_id is not None:
        models = [path for path in models if path.name == f"model_{model_id}"]
    if not models:
        raise DAMITLoadError(f"no selected model found for {object_id}")
    # Synthetic shape donation does not consume the pole label. When no model
    # is requested, use the lexicographically first immutable model directory
    # and record its ID in every generated shard. Real labelled examples must
    # pass their catalogue model IDs explicitly.
    model = models[0]
    shape_path, spin_path = model / "shape.txt", model / "spin.txt"
    period, _lambda, pole = _read_spin(spin_path)
    return DAMITObject(
        object_id=object_id,
        epochs=_read_lc(object_dir / "lc.txt"),
        shape=parse_damit_shape(shape_path, source_id=f"{object_id}/{model.name}"),
        model_id=model.name.removeprefix("model_"),
        period_hours=period,
        pole_ecliptic=pole,
    )
