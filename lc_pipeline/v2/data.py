"""DAMIT catalogue loading and prepared-object validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .preprocessing import (
    TOKEN_FEATURE_NAMES,
    Observation,
    ObservationEpoch,
)
from .serialization import canonical_json as canonical_json
from .serialization import sha256_file as sha256_file


class DataContractError(ValueError):
    """Raised when source data or a prepared cache violates the V2 contract."""


@dataclass(frozen=True)
class CatalogObject:
    object_id: str
    lightcurve_path: str
    lightcurve_sha256: str
    solution_vectors: tuple[tuple[float, float, float], ...]
    solution_periods_hours: tuple[float, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CatalogObject":
        if value.get("eligible") is not True:
            raise DataContractError("only eligible catalog objects may be prepared")
        object_id = value.get("object_id")
        lightcurve = value.get("lightcurve")
        solutions = value.get("solutions")
        if not isinstance(object_id, str) or not object_id:
            raise DataContractError("catalog object_id must be nonempty")
        if not isinstance(lightcurve, Mapping) or not isinstance(solutions, list) or not solutions:
            raise DataContractError(f"catalog object {object_id} lacks lightcurve or solutions")
        vectors: list[tuple[float, float, float]] = []
        periods: list[float] = []
        for solution in solutions:
            if not isinstance(solution, Mapping):
                raise DataContractError(f"invalid solution for {object_id}")
            vector = solution.get("vector")
            if not isinstance(vector, list) or len(vector) != 3:
                raise DataContractError(f"invalid solution vector for {object_id}")
            components = tuple(float(component) for component in vector)
            period = float(solution.get("period_hours"))
            if (
                not all(math.isfinite(component) for component in components)
                or not math.isfinite(period)
                or period <= 0
            ):
                raise DataContractError(f"non-finite solution for {object_id}")
            vectors.append(components)
            periods.append(period)
        source_path = lightcurve.get("source_path")
        source_hash = lightcurve.get("source_sha256")
        if (
            not isinstance(source_path, str)
            or Path(source_path).is_absolute()
            or ".." in Path(source_path).parts
        ):
            raise DataContractError(f"invalid logical lightcurve path for {object_id}")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise DataContractError(f"invalid lightcurve hash for {object_id}")
        return cls(object_id, source_path, source_hash, tuple(vectors), tuple(periods))

    @property
    def source_period_hours(self) -> float:
        reference = self.solution_periods_hours[0]
        if any(
            abs(period - reference) / reference > 1e-3 for period in self.solution_periods_hours[1:]
        ):
            raise DataContractError(f"{self.object_id} has inconsistent explicit source periods")
        return float(np.median(self.solution_periods_hours))


def load_catalog(path: str | Path) -> dict[str, CatalogObject]:
    records: dict[str, CatalogObject] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DataContractError(f"cannot read catalog: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataContractError(f"invalid catalog JSON on line {line_number}: {exc}") from exc
        if raw.get("eligible") is not True:
            continue
        record = CatalogObject.from_mapping(raw)
        if record.object_id in records:
            raise DataContractError(f"duplicate catalog object_id: {record.object_id}")
        records[record.object_id] = record
    if not records:
        raise DataContractError("catalog contains no eligible objects")
    return records


def parse_damit_lightcurve(path: str | Path) -> tuple[ObservationEpoch, ...]:
    """Parse source epoch boundaries and all eight DAMIT observation columns."""
    source = Path(path)
    try:
        tokens = source.read_text(encoding="utf-8").split()
        n_epochs = int(tokens[0])
    except (OSError, IndexError, ValueError) as exc:
        raise DataContractError(f"cannot parse DAMIT lightcurve {source}: {exc}") from exc
    if n_epochs <= 0:
        raise DataContractError(f"DAMIT lightcurve has no epochs: {source}")
    position = 1
    epochs: list[ObservationEpoch] = []
    for epoch_index in range(n_epochs):
        try:
            n_points = int(tokens[position])
            calibrated = int(tokens[position + 1])
        except (IndexError, ValueError) as exc:
            raise DataContractError(f"malformed epoch {epoch_index}: {source}") from exc
        position += 2
        if n_points <= 0 or calibrated not in (0, 1):
            raise DataContractError(f"invalid epoch header {epoch_index}: {source}")
        observations: list[Observation] = []
        for row_index in range(n_points):
            try:
                row = tuple(float(value) for value in tokens[position : position + 8])
            except ValueError as exc:
                raise DataContractError(
                    f"non-numeric epoch {epoch_index} row {row_index}: {source}"
                ) from exc
            position += 8
            if len(row) != 8:
                raise DataContractError(f"truncated epoch {epoch_index} row {row_index}: {source}")
            observations.append(
                Observation(
                    time_jd=row[0],
                    relative_brightness=row[1],
                    sun_asteroid_ecliptic_j2000_au=row[2:5],
                    observer_asteroid_ecliptic_j2000_au=row[5:8],
                )
            )
        epochs.append(ObservationEpoch(f"source-epoch-{epoch_index:04d}", tuple(observations)))
    if position != len(tokens):
        raise DataContractError(f"unexpected trailing tokens in {source}")
    return tuple(epochs)


@dataclass(frozen=True)
class PreparedObject:
    object_id: str
    tokens: np.ndarray
    observation_mask: np.ndarray
    epoch_descriptors: np.ndarray
    epoch_mask: np.ndarray
    period_values: np.ndarray
    period_mask: np.ndarray
    target_pixels: np.ndarray
    target_vectors: np.ndarray
    period_hours_targets: np.ndarray

    @property
    def epoch_slots(self) -> int:
        return int(self.tokens.shape[0])

    def validate(self) -> None:
        if not self.object_id:
            raise DataContractError("prepared object_id is required")
        slots, observations, features = self.tokens.shape
        if features != len(TOKEN_FEATURE_NAMES) or self.observation_mask.shape != (
            slots,
            observations,
        ):
            raise DataContractError(f"invalid prepared token shapes for {self.object_id}")
        if self.epoch_descriptors.shape != (slots, 3) or self.epoch_mask.shape != (slots,):
            raise DataContractError(f"invalid prepared epoch shapes for {self.object_id}")
        if self.period_values.shape != (2,) or self.period_mask.shape != (2,):
            raise DataContractError(f"invalid prepared period shapes for {self.object_id}")
        if self.target_pixels.ndim != 1 or self.target_vectors.shape != (
            self.target_pixels.size,
            3,
        ):
            raise DataContractError(f"invalid prepared targets for {self.object_id}")
        if (
            self.period_hours_targets.ndim != 1
            or self.period_hours_targets.size != self.target_pixels.size
        ):
            raise DataContractError(
                f"period targets must align with source solutions for {self.object_id}"
            )
        if not self.epoch_mask.any() or not self.target_pixels.size:
            raise DataContractError(f"empty prepared object: {self.object_id}")
        arrays = (
            self.tokens,
            self.epoch_descriptors,
            self.period_values,
            self.target_vectors,
            self.period_hours_targets,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise DataContractError(f"non-finite prepared array for {self.object_id}")
