"""Hash-bound execution of the axial V1-versus-V2 comparison protocol."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from ..physics.axial import nearest_source_axis_errors_deg, oracle_source_axis_match
from .atlas import AxisAtlas, fit_axis_atlas, read_axis_atlas, write_axis_atlas
from .axial_training import (
    AxialTrainingConfig,
    AxisObject,
    AxisPrediction,
    ablate_v2_inputs,
    exhaustive_deranged_axis_errors,
    fit_axis_model,
    predict_axes,
    zero_axis_inputs,
)
from .baselines import (
    AmplitudeAxisObject,
    AmplitudeFeatureMLP,
    FeatureStandardizer,
    LegacyAxisObject,
    make_v1_model,
    prepare_amplitude_baseline_object,
    prepare_v1_baseline_object,
)
from .comparison_contract import AXIAL_COMPARISON_SPEC_SHA256, load_axial_comparison_spec
from .data import (
    PreparedObject,
    canonical_json,
    load_cache_manifest,
    load_catalog,
    load_development_split,
    load_fold,
    parse_damit_lightcurve,
    read_prepared_object,
    sha256_file,
)
from .model import (
    GeometryHierarchicalAxisModel,
    GeometryHierarchicalResidualAxisModel,
    V2AxisModelConfig,
)
from .training import configure_determinism

AXIAL_EXPERIMENT_SCHEMA = "delphi.axial-experiment-config.v1"
AXIAL_EXPERIMENT_SUMMARY_SCHEMA = "delphi.axial-experiment-summary.v1"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
AxisStage = Literal["development", "confirmation"]


class AxialExperimentError(ValueError):
    """Raised when an axial run is not fully bound to the frozen protocol."""


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _ids_hash(values: Sequence[str]) -> str:
    return _sha256_json(sorted(values))


@dataclass(frozen=True)
class AxialExperimentConfig:
    run_id: str
    stage: AxisStage
    fold: int | None
    model_kind: str
    training: AxialTrainingConfig
    catalog_sha256: str
    splits_sha256: str
    cache_configuration_sha256: str
    comparison_spec_sha256: str
    axis_model: V2AxisModelConfig | None = None
    atlas_path: str | None = None
    schema: str = AXIAL_EXPERIMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != AXIAL_EXPERIMENT_SCHEMA or RUN_ID_RE.fullmatch(self.run_id) is None:
            raise AxialExperimentError("invalid axial experiment schema or run_id")
        if self.stage not in {"development", "confirmation"}:
            raise AxialExperimentError("stage must be development or confirmation")
        if self.stage == "development" and self.fold is not None:
            raise AxialExperimentError("development runs must not declare a publication fold")
        if self.stage == "confirmation" and self.fold not in range(5):
            raise AxialExperimentError("confirmation fold must lie in [0, 4]")
        if self.model_kind != self.training.model_kind:
            raise AxialExperimentError("model_kind and training.model_kind differ")
        for name, digest in (
            ("catalog_sha256", self.catalog_sha256),
            ("splits_sha256", self.splits_sha256),
            ("cache_configuration_sha256", self.cache_configuration_sha256),
            ("comparison_spec_sha256", self.comparison_spec_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise AxialExperimentError(f"{name} must be a lowercase SHA-256 digest")
        if self.comparison_spec_sha256 != AXIAL_COMPARISON_SPEC_SHA256:
            raise AxialExperimentError("comparison specification hash mismatch")
        is_v2 = self.model_kind.startswith("v2_")
        if is_v2 != (self.axis_model is not None):
            raise AxialExperimentError("only V2 arms may declare an axis_model")
        if (
            self.model_kind == "v2_plain"
            and self.axis_model
            and self.axis_model.head_type != "plain"
        ):
            raise AxialExperimentError("v2_plain requires the plain axis head")
        if self.model_kind == "v2_residual":
            if (
                self.axis_model is None
                or self.axis_model.head_type != "residual"
                or not self.atlas_path
            ):
                raise AxialExperimentError(
                    "v2_residual requires residual configuration and atlas_path"
                )
        if self.axis_model is not None:
            expected_geometry = self.training.control != "geometry_disabled"
            if self.axis_model.geometry_required != expected_geometry:
                raise AxialExperimentError(
                    "axis model geometry contract mismatches training control"
                )
        frozen = AxialTrainingConfig.frozen(
            self.model_kind, self.training.seed, control=self.training.control
        )
        if asdict(self.training) != asdict(frozen):
            raise AxialExperimentError(
                "training configuration differs from the frozen comparison schedule"
            )

    @property
    def config_sha256(self) -> str:
        return _sha256_json(self.as_mapping())

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "stage": self.stage,
            "fold": self.fold,
            "model_kind": self.model_kind,
            "training": asdict(self.training),
            "catalog_sha256": self.catalog_sha256,
            "splits_sha256": self.splits_sha256,
            "cache_configuration_sha256": self.cache_configuration_sha256,
            "comparison_spec_sha256": self.comparison_spec_sha256,
            "axis_model": None if self.axis_model is None else asdict(self.axis_model),
            "atlas_path": self.atlas_path,
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "AxialExperimentConfig":
        try:
            document = json.loads(
                Path(path).read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs
            )
            axis_raw = document.pop("axis_model", None)
            training = AxialTrainingConfig(**document.pop("training"))
            axis_model = None if axis_raw is None else V2AxisModelConfig(**axis_raw)
            return cls(training=training, axis_model=axis_model, **document)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, AxialExperimentError):
                raise
            raise AxialExperimentError(
                f"cannot load axial experiment configuration: {exc}"
            ) from exc


def _assert_clean_repository(repository_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise AxialExperimentError("axial comparison runs require a clean Git worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise AxialExperimentError("axial comparison requires a full Git revision")
    return commit


def _partition_values(
    cache_root: Path, entries: Mapping[str, str], identifiers: Sequence[str]
) -> tuple[PreparedObject, ...]:
    return tuple(
        read_prepared_object(cache_root / f"{object_id}.npz", expected_sha256=entries[object_id])
        for object_id in identifiers
    )


def _baseline_objects(
    prepared: Sequence[PreparedObject],
    *,
    catalog: Mapping[str, object],
    dump_root: Path,
    model_kind: str,
) -> tuple[LegacyAxisObject | AmplitudeAxisObject, ...]:
    values: list[LegacyAxisObject | AmplitudeAxisObject] = []
    for value in prepared:
        record = catalog[value.object_id]
        epochs = parse_damit_lightcurve(dump_root / record.lightcurve_path)
        if model_kind.startswith("v1_"):
            values.append(prepare_v1_baseline_object(value, epochs))
        else:
            values.append(prepare_amplitude_baseline_object(value, epochs))
    return tuple(values)


def _build_model(config: AxialExperimentConfig, atlas: AxisAtlas | None) -> torch.nn.Module:
    configure_determinism(config.training.seed)
    if config.model_kind == "v1_faithful" or config.model_kind == "v1_corrected":
        return make_v1_model(seed=config.training.seed)
    if config.model_kind == "amplitude_mlp":
        return AmplitudeFeatureMLP()
    assert config.axis_model is not None
    if config.model_kind == "v2_plain":
        return GeometryHierarchicalAxisModel(config.axis_model)
    if atlas is None:
        raise AxialExperimentError("residual V2 run requires a verified atlas")
    return GeometryHierarchicalResidualAxisModel(config.axis_model, atlas)


def _prediction_rows(
    predictions: Sequence[AxisPrediction],
    *,
    config: AxialExperimentConfig,
    condition: str,
    override_errors: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in predictions:
        error = (
            item.axis_oracle_at3_error_deg
            if override_errors is None
            else override_errors[item.object_id]
        )
        rows.append(
            {
                "object_id": item.object_id,
                "model_kind": config.model_kind,
                "condition": condition,
                "seed": config.training.seed,
                "axis_oracle_at3_error_deg": float(error),
                "axis_top1_error_deg": item.axis_top1_error_deg,
                "directed_oracle_at3_error_deg": item.directed_oracle_at3_error_deg,
                "axes": [list(axis) for axis in item.axes],
                "axis_valid": list(item.axis_valid),
                "has_valid_axis": bool(any(item.axis_valid)),
                "target_vectors": [list(axis) for axis in item.target_vectors],
                "period_hours_targets": list(item.period_hours_targets),
            }
        )
    return rows


def _atlas_predictions(
    values: Sequence[AxisObject], atlas: AxisAtlas, config: AxialExperimentConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    axes = np.asarray(atlas.axes, dtype=np.float64)
    for value in values:
        target = np.asarray(value.target_vectors, dtype=np.float64)
        rows.append(
            {
                "object_id": value.object_id,
                "model_kind": config.model_kind,
                "condition": "atlas",
                "seed": config.training.seed,
                "axis_oracle_at3_error_deg": float(
                    oracle_source_axis_match(axes, target).error_deg
                ),
                "axis_top1_error_deg": float(nearest_source_axis_errors_deg(axes[:1], target)[0]),
                "directed_oracle_at3_error_deg": None,
                "axes": [list(axis) for axis in atlas.axes],
                "target_vectors": target.tolist(),
                "period_hours_targets": np.asarray(value.period_hours_targets).tolist(),
            }
        )
    return rows


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def fit_train_only_atlas(
    *,
    cache_manifest_path: str | Path,
    splits_path: str | Path,
    catalog_path: str | Path,
    stage: AxisStage,
    fold: int | None,
    output_path: str | Path,
) -> AxisAtlas:
    """Materialize a training-only atlas for one declared development/fold split."""
    load_axial_comparison_spec()
    manifest, entries = load_cache_manifest(cache_manifest_path, verify_files=True)
    if sha256_file(catalog_path) != manifest["catalog_sha256"]:
        raise AxialExperimentError("catalog does not match the prepared cache")
    roles = (
        load_development_split(splits_path)
        if stage == "development"
        else load_fold(splits_path, int(fold))
    )
    train = _partition_values(Path(cache_manifest_path).parent, entries, roles["train_ids"])
    forbidden = tuple(
        object_id for role, values in roles.items() if role != "train_ids" for object_id in values
    )
    atlas = fit_axis_atlas(train, partition_role="train", forbidden_object_ids=forbidden)
    write_axis_atlas(output_path, atlas)
    return atlas


def run_axial_experiment(
    config: AxialExperimentConfig,
    *,
    repository_root: str | Path,
    catalog_path: str | Path,
    splits_path: str | Path,
    cache_manifest_path: str | Path,
    output_root: str | Path,
    dump_root: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute one arm/seed/fold and write reviewable evaluation JSONL rows."""
    load_axial_comparison_spec()
    repository = Path(repository_root).resolve()
    commit = _assert_clean_repository(repository)
    if sha256_file(catalog_path) != config.catalog_sha256:
        raise AxialExperimentError("catalog hash does not match run configuration")
    if sha256_file(splits_path) != config.splits_sha256:
        raise AxialExperimentError("split hash does not match run configuration")
    manifest, entries = load_cache_manifest(cache_manifest_path, verify_files=True)
    if manifest["catalog_sha256"] != config.catalog_sha256:
        raise AxialExperimentError("prepared cache is from another catalog")
    if manifest["configuration_sha256"] != config.cache_configuration_sha256:
        raise AxialExperimentError("prepared cache configuration hash mismatch")
    roles = (
        load_development_split(splits_path)
        if config.stage == "development"
        else load_fold(splits_path, int(config.fold))
    )
    catalog = load_catalog(catalog_path)
    expected_ids = set(item for role in roles.values() for item in role)
    if expected_ids != set(catalog) or not expected_ids <= entries.keys():
        raise AxialExperimentError("catalog, split, and cache object sets do not agree")
    cache_root = Path(cache_manifest_path).parent
    train_prepared = _partition_values(cache_root, entries, roles["train_ids"])
    validation_prepared = _partition_values(cache_root, entries, roles["validation_ids"])
    calibration_prepared = _partition_values(cache_root, entries, roles["calibration_ids"])
    evaluation_prepared = (
        validation_prepared
        if config.stage == "development"
        else _partition_values(cache_root, entries, roles["test_ids"])
    )
    forbidden_ids = tuple(
        object_id
        for role, identifiers in roles.items()
        if role != "train_ids"
        for object_id in identifiers
    )
    run_directory = Path(output_root) / config.run_id
    if run_directory.exists() and any(run_directory.iterdir()):
        raise AxialExperimentError("axial run directory already exists")
    run_directory.mkdir(parents=True, exist_ok=False)
    resolved_path = run_directory / "axial-experiment-config.json"
    resolved_path.write_text(canonical_json(config.as_mapping()) + "\n", encoding="utf-8")

    generated_atlas_path = run_directory / "train-only-atlas.json"
    if config.atlas_path is not None:
        atlas_source = Path(config.atlas_path)
        if not atlas_source.is_absolute():
            atlas_source = repository / atlas_source
        atlas = read_axis_atlas(atlas_source)
        if atlas.train_object_ids_sha256 != _ids_hash(roles["train_ids"]):
            raise AxialExperimentError("declared atlas was not fit on this exact training fold")
        write_axis_atlas(generated_atlas_path, atlas)
    else:
        atlas = fit_axis_atlas(
            train_prepared, partition_role="train", forbidden_object_ids=forbidden_ids
        )
        write_axis_atlas(generated_atlas_path, atlas)

    if config.model_kind.startswith("v2_"):
        train: tuple[AxisObject, ...] = train_prepared
        validation: tuple[AxisObject, ...] = validation_prepared
        evaluation: tuple[AxisObject, ...] = evaluation_prepared
        standardizer = None
    else:
        if dump_root is None:
            raise AxialExperimentError("V1/amplitude comparison arms require --dump-root")
        raw_root = Path(dump_root)
        train = _baseline_objects(
            train_prepared, catalog=catalog, dump_root=raw_root, model_kind=config.model_kind
        )
        validation = _baseline_objects(
            validation_prepared, catalog=catalog, dump_root=raw_root, model_kind=config.model_kind
        )
        evaluation = _baseline_objects(
            evaluation_prepared, catalog=catalog, dump_root=raw_root, model_kind=config.model_kind
        )
        standardizer = (
            FeatureStandardizer.fit(
                train, partition_role="train", forbidden_object_ids=forbidden_ids
            )
            if config.model_kind == "amplitude_mlp"
            else None
        )
    model = _build_model(config, atlas)
    fit = fit_axis_model(
        model,
        train,
        validation,
        config.training,
        run_directory / "primary",
        standardizer=standardizer,
        device=device,
    )
    raw = predict_axes(model, evaluation, config.training, standardizer=standardizer, device=device)
    zero = predict_axes(
        model,
        zero_axis_inputs(evaluation),
        config.training,
        standardizer=standardizer,
        device=device,
    )
    rows = _prediction_rows(raw, config=config, condition="raw")
    rows.extend(_prediction_rows(zero, config=config, condition="zero"))
    rows.extend(_atlas_predictions(evaluation, atlas, config))
    rows.extend(
        _prediction_rows(
            raw,
            config=config,
            condition="exhaustive_deranged",
            override_errors=exhaustive_deranged_axis_errors(raw),
        )
    )
    if config.stage == "confirmation" and config.model_kind.startswith("v2_"):
        label_config = replace(
            config.training,
            control="label_shuffle",
        )
        label_model = _build_model(replace(config, training=label_config), atlas)
        fit_axis_model(
            label_model,
            train,
            validation,
            label_config,
            run_directory / "label-shuffle",
            device=device,
        )
        label_predictions = predict_axes(label_model, evaluation, label_config, device=device)
        rows.extend(_prediction_rows(label_predictions, config=config, condition="label_shuffle"))
        assert config.axis_model is not None
        geometry_config = replace(config.training, control="geometry_disabled")
        geometry_model_config = replace(config.axis_model, geometry_required=False)
        geometry_experiment = replace(
            config,
            training=geometry_config,
            axis_model=geometry_model_config,
        )
        geometry_model = _build_model(geometry_experiment, atlas)
        fit_axis_model(
            geometry_model,
            train,
            validation,
            geometry_config,
            run_directory / "geometry-disabled",
            device=device,
        )
        geometry_predictions = predict_axes(
            geometry_model, evaluation, geometry_config, device=device
        )
        rows.extend(
            _prediction_rows(geometry_predictions, config=config, condition="geometry_disabled")
        )
        for channel in ("brightness", "time", "geometry", "period"):
            ablated = predict_axes(
                model,
                ablate_v2_inputs(evaluation_prepared, channel),
                config.training,
                device=device,
            )
            rows.extend(_prediction_rows(ablated, config=config, condition=f"{channel}_ablation"))
    rows_path = run_directory / "evaluation-rows.jsonl"
    rows_path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "schema": AXIAL_EXPERIMENT_SUMMARY_SCHEMA,
        "run_id": config.run_id,
        "config_sha256": config.config_sha256,
        "comparison_spec_sha256": AXIAL_COMPARISON_SPEC_SHA256,
        "git_commit": commit,
        "stage": config.stage,
        "fold": config.fold,
        "model_kind": config.model_kind,
        "seed": config.training.seed,
        "train_count": len(train),
        "validation_count": len(validation),
        "calibration_count": len(calibration_prepared),
        "evaluation_count": len(evaluation),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "fit": asdict(fit),
        "atlas_sha256": atlas.atlas_sha256,
        "artifacts": {
            resolved_path.name: sha256_file(resolved_path),
            generated_atlas_path.name: sha256_file(generated_atlas_path),
            rows_path.name: sha256_file(rows_path),
        },
        "environment": _environment(),
    }
    summary_path = run_directory / "axial-experiment-summary.json"
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "AXIAL_EXPERIMENT_SCHEMA",
    "AXIAL_EXPERIMENT_SUMMARY_SCHEMA",
    "AxialExperimentConfig",
    "AxialExperimentError",
    "fit_train_only_atlas",
    "run_axial_experiment",
]
