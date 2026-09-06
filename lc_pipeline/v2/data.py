"""Publication-grade DAMIT loading and variable-length V2 tensor caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .density import vector_to_pixel
from .period import PeriodSearchConfig, PeriodStatus, infer_period
from .preprocessing import (
    TOKEN_FEATURE_NAMES,
    TOKENIZER_SCHEMA_HASH,
    GeometryMode,
    KnownPeriod,
    Observation,
    ObservationEpoch,
    PreprocessingConfig,
    build_observation_tensors,
)

CACHE_SCHEMA = "delphi.prepared-object.v2"
CACHE_MANIFEST_SCHEMA = "delphi.tensor-cache-manifest.v2"


class DataContractError(ValueError):
    """Raised when source data or a prepared cache violates the V2 contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def load_fold(path: str | Path, fold: int) -> dict[str, tuple[str, ...]]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        selected = next(item for item in document["folds"] if item["fold"] == fold)
    except (OSError, json.JSONDecodeError, KeyError, StopIteration, TypeError) as exc:
        raise DataContractError(f"cannot load split fold {fold}: {exc}") from exc
    result: dict[str, tuple[str, ...]] = {}
    for role in ("train_ids", "validation_ids", "calibration_ids", "test_ids"):
        values = selected.get(role)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise DataContractError(f"fold {fold} has invalid {role}")
        if len(values) != len(set(values)):
            raise DataContractError(f"fold {fold} has duplicate IDs in {role}")
        result[role] = tuple(values)
    all_ids = [item for values in result.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise DataContractError(f"fold {fold} roles overlap")
    return result


def load_development_split(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load the deliberately test-free development partition."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        roles = document["roles"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataContractError(f"cannot load development split: {exc}") from exc
    if document.get("schema") != "delphi.development-split.v2":
        raise DataContractError("development split schema mismatch")
    result: dict[str, tuple[str, ...]] = {}
    for role in ("train_ids", "validation_ids", "calibration_ids"):
        values = roles.get(role)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise DataContractError(f"development split has invalid {role}")
        if len(values) != len(set(values)):
            raise DataContractError(f"development split has duplicate IDs in {role}")
        result[role] = tuple(values)
    all_ids = [item for values in result.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise DataContractError("development split roles overlap")
    return result


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


def required_epoch_slots(epochs: Sequence[ObservationEpoch], chunk_size: int) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise DataContractError("chunk_size must be a positive integer")
    return sum(math.ceil(len(epoch.observations) / chunk_size) for epoch in epochs)


def _period_for_features(
    record: CatalogObject,
    epochs: Sequence[ObservationEpoch],
    mode: str,
    config: PeriodSearchConfig,
) -> KnownPeriod | None:
    if mode == "source":
        return KnownPeriod(record.source_period_hours, "damit-explicit-source-period")
    if mode != "inferred":
        raise DataContractError("period mode must be 'inferred' or 'source'")
    result = infer_period(epochs, config=config)
    if result.status is PeriodStatus.UNINFORMATIVE or result.period_hours is None:
        return None
    uncertainty = None
    if result.ci68_hours is not None:
        uncertainty = (result.ci68_hours[1] - result.ci68_hours[0]) / 2.0
        if not math.isfinite(uncertainty) or uncertainty <= 0:
            uncertainty = None
    return KnownPeriod(
        result.period_hours,
        f"delphi-v2-period:{result.status.value}",
        uncertainty_hours=uncertainty,
    )


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


def prepare_object(
    record: CatalogObject,
    dump_root: str | Path,
    *,
    nside: int,
    chunk_size: int = 256,
    geometry_mode: GeometryMode = GeometryMode.ENABLED,
    period_mode: str = "inferred",
    period_config: PeriodSearchConfig = PeriodSearchConfig(),
    verify_source_hash: bool = True,
) -> PreparedObject:
    source = Path(dump_root) / record.lightcurve_path
    if not source.is_file():
        raise DataContractError(f"missing source lightcurve for {record.object_id}: {source}")
    if verify_source_hash and sha256_file(source) != record.lightcurve_sha256:
        raise DataContractError(f"source lightcurve hash mismatch for {record.object_id}")
    epochs = parse_damit_lightcurve(source)
    slots = required_epoch_slots(epochs, chunk_size)
    known_period = _period_for_features(record, epochs, period_mode, period_config)
    prepared = build_observation_tensors(
        epochs,
        config=PreprocessingConfig(slots, chunk_size, geometry_mode),
        known_period=known_period,
    )
    target_vectors = np.asarray(record.solution_vectors, dtype=np.float64)
    result = PreparedObject(
        object_id=record.object_id,
        tokens=prepared.tokens,
        observation_mask=prepared.observation_mask,
        epoch_descriptors=prepared.epoch_descriptors,
        epoch_mask=prepared.epoch_mask,
        period_values=prepared.period_values,
        period_mask=prepared.period_mask,
        target_pixels=np.asarray(
            [vector_to_pixel(vector, nside) for vector in target_vectors], dtype=np.int64
        ),
        target_vectors=target_vectors,
        period_hours_targets=np.asarray(record.solution_periods_hours, dtype=np.float64),
    )
    result.validate()
    return result


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_prepared_object(path: str | Path, value: PreparedObject) -> str:
    value.validate()
    metadata = {
        "schema": CACHE_SCHEMA,
        "object_id": value.object_id,
        "tokenizer_schema_sha256": TOKENIZER_SCHEMA_HASH,
    }
    _atomic_save_npz(
        Path(path),
        {
            "metadata": np.asarray(canonical_json(metadata)),
            "tokens": value.tokens,
            "observation_mask": value.observation_mask,
            "epoch_descriptors": value.epoch_descriptors,
            "epoch_mask": value.epoch_mask,
            "period_values": value.period_values,
            "period_mask": value.period_mask,
            "target_pixels": value.target_pixels,
            "target_vectors": value.target_vectors,
            "period_hours_targets": value.period_hours_targets,
        },
    )
    return sha256_file(path)


def materialize_tensor_cache(
    *,
    catalog_path: str | Path,
    dump_root: str | Path,
    cache_root: str | Path,
    nside: int = 32,
    chunk_size: int = 256,
    geometry_mode: GeometryMode = GeometryMode.ENABLED,
    period_mode: str = "inferred",
    period_config: PeriodSearchConfig = PeriodSearchConfig(),
    object_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create or resume a hash-bound external cache with no silent omissions."""
    catalog = load_catalog(catalog_path)
    selected_ids = tuple(sorted(catalog) if object_ids is None else object_ids)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise DataContractError("cache object IDs must be nonempty and unique")
    missing = sorted(set(selected_ids) - catalog.keys())
    if missing:
        raise DataContractError(f"cache requested unknown objects: {missing[:10]}")
    destination = Path(cache_root)
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for object_id in selected_ids:
        output_path = destination / f"{object_id}.npz"
        prepared = prepare_object(
            catalog[object_id],
            dump_root,
            nside=nside,
            chunk_size=chunk_size,
            geometry_mode=geometry_mode,
            period_mode=period_mode,
            period_config=period_config,
        )
        digest = write_prepared_object(output_path, prepared)
        entries.append(
            {
                "object_id": object_id,
                "logical_path": output_path.name,
                "sha256": digest,
                "size_bytes": output_path.stat().st_size,
                "epoch_slots": prepared.epoch_slots,
                "n_observations": int(prepared.observation_mask.sum()),
                "n_solutions": int(prepared.target_pixels.size),
            }
        )
    configuration = {
        "nside": nside,
        "chunk_size": chunk_size,
        "geometry_mode": geometry_mode.value,
        "period_mode": period_mode,
        "period_config": asdict(period_config),
        "tokenizer_schema_sha256": TOKENIZER_SCHEMA_HASH,
    }
    manifest = {
        "schema": CACHE_MANIFEST_SCHEMA,
        "catalog_sha256": sha256_file(catalog_path),
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(canonical_json(configuration).encode()).hexdigest(),
        "object_count": len(entries),
        "entries": entries,
    }
    _atomic_json(destination / "cache-manifest.json", manifest)
    return manifest


def load_cache_manifest(
    path: str | Path, *, verify_files: bool = True
) -> tuple[dict[str, Any], dict[str, str]]:
    source = Path(path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"cannot read cache manifest: {exc}") from exc
    if manifest.get("schema") != CACHE_MANIFEST_SCHEMA or manifest.get("object_count") != len(
        manifest.get("entries", [])
    ):
        raise DataContractError("cache manifest schema/count mismatch")
    configuration = manifest.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("tokenizer_schema_sha256") != TOKENIZER_SCHEMA_HASH
    ):
        raise DataContractError("cache tokenizer schema mismatch")
    expected_configuration_hash = hashlib.sha256(canonical_json(configuration).encode()).hexdigest()
    if manifest.get("configuration_sha256") != expected_configuration_hash:
        raise DataContractError("cache configuration hash mismatch")
    entries: dict[str, str] = {}
    for entry in manifest["entries"]:
        try:
            object_id = entry["object_id"]
            logical_path = entry["logical_path"]
            digest = entry["sha256"]
            size = entry["size_bytes"]
        except (KeyError, TypeError) as exc:
            raise DataContractError(f"malformed cache entry: {exc}") from exc
        if (
            not isinstance(object_id, str)
            or object_id in entries
            or logical_path != f"{object_id}.npz"
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(size, int)
            or size <= 0
        ):
            raise DataContractError(f"invalid cache entry for {object_id!r}")
        artifact = source.parent / logical_path
        if verify_files and (
            not artifact.is_file()
            or artifact.stat().st_size != size
            or sha256_file(artifact) != digest
        ):
            raise DataContractError(f"cache artifact verification failed: {logical_path}")
        entries[object_id] = digest
    return manifest, entries


def read_prepared_object(path: str | Path, *, expected_sha256: str | None = None) -> PreparedObject:
    source = Path(path)
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise DataContractError(f"prepared cache hash mismatch: {source.name}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata != {
                "schema": CACHE_SCHEMA,
                "object_id": metadata.get("object_id"),
                "tokenizer_schema_sha256": TOKENIZER_SCHEMA_HASH,
            }:
                raise DataContractError(f"prepared cache metadata mismatch: {source.name}")
            value = PreparedObject(
                object_id=metadata["object_id"],
                tokens=archive["tokens"].copy(),
                observation_mask=archive["observation_mask"].copy(),
                epoch_descriptors=archive["epoch_descriptors"].copy(),
                epoch_mask=archive["epoch_mask"].copy(),
                period_values=archive["period_values"].copy(),
                period_mask=archive["period_mask"].copy(),
                target_pixels=archive["target_pixels"].copy(),
                target_vectors=archive["target_vectors"].copy(),
                period_hours_targets=archive["period_hours_targets"].copy(),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, DataContractError):
            raise
        raise DataContractError(f"cannot read prepared cache {source}: {exc}") from exc
    value.validate()
    return value


def collate_prepared_objects(values: Sequence[PreparedObject]) -> dict[str, object]:
    if not values:
        raise DataContractError("cannot collate an empty object batch")
    for value in values:
        value.validate()
    batch = len(values)
    max_slots = max(value.epoch_slots for value in values)
    observations = values[0].tokens.shape[1]
    max_solutions = max(value.target_pixels.size for value in values)
    if any(value.tokens.shape[1] != observations for value in values):
        raise DataContractError("all cached objects in a batch require the same chunk size")
    tokens = np.zeros((batch, max_slots, observations, len(TOKEN_FEATURE_NAMES)), dtype=np.float32)
    observation_mask = np.zeros((batch, max_slots, observations), dtype=bool)
    descriptors = np.zeros((batch, max_slots, 3), dtype=np.float32)
    epoch_mask = np.zeros((batch, max_slots), dtype=bool)
    period_values = np.zeros((batch, 2), dtype=np.float32)
    period_mask = np.zeros((batch, 2), dtype=bool)
    target_pixels = np.zeros((batch, max_solutions), dtype=np.int64)
    target_mask = np.zeros((batch, max_solutions), dtype=bool)
    target_vectors = np.zeros((batch, max_solutions, 3), dtype=np.float64)
    period_targets = np.zeros((batch, max_solutions), dtype=np.float64)
    for index, value in enumerate(values):
        slots = value.epoch_slots
        solutions = value.target_pixels.size
        tokens[index, :slots] = value.tokens
        observation_mask[index, :slots] = value.observation_mask
        descriptors[index, :slots] = value.epoch_descriptors
        epoch_mask[index, :slots] = value.epoch_mask
        period_values[index] = value.period_values
        period_mask[index] = value.period_mask
        target_pixels[index, :solutions] = value.target_pixels
        target_mask[index, :solutions] = True
        target_vectors[index, :solutions] = value.target_vectors
        period_targets[index, :solutions] = value.period_hours_targets
    return {
        "object_ids": tuple(value.object_id for value in values),
        "tokens": torch.from_numpy(tokens),
        "observation_mask": torch.from_numpy(observation_mask),
        "epoch_descriptors": torch.from_numpy(descriptors),
        "epoch_mask": torch.from_numpy(epoch_mask),
        "period_values": torch.from_numpy(period_values),
        "period_mask": torch.from_numpy(period_mask),
        "target_pixels": torch.from_numpy(target_pixels),
        "target_mask": torch.from_numpy(target_mask),
        "target_vectors": torch.from_numpy(target_vectors),
        "period_hours_targets": torch.from_numpy(period_targets),
    }


class PreparedObjectDataset(torch.utils.data.Dataset[PreparedObject]):
    """Lazy, hash-verifying dataset over an external prepared-object cache."""

    def __init__(
        self, cache_root: str | Path, entries: Mapping[str, str], object_ids: Iterable[str]
    ) -> None:
        self.cache_root = Path(cache_root)
        self.entries = dict(entries)
        self.object_ids = tuple(object_ids)
        missing = sorted(set(self.object_ids) - self.entries.keys())
        if missing:
            raise DataContractError(f"cache manifest lacks object IDs: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, index: int) -> PreparedObject:
        object_id = self.object_ids[index]
        return read_prepared_object(
            self.cache_root / f"{object_id}.npz", expected_sha256=self.entries[object_id]
        )
