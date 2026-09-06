"""Dry-run planning and job-boundary-resumable execution for publication OOF runs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..v2.axial_experiment import (
    AXIAL_EXPERIMENT_SUMMARY_SCHEMA,
    AxialExperimentConfig,
    fit_train_only_atlas,
    run_axial_experiment,
)
from ..v2.axial_training import AxialTrainingConfig
from ..v2.comparison_contract import AXIAL_COMPARISON_SPEC_SHA256
from .artifacts import (
    PHASES,
    ArtifactReference,
    PhaseJob,
    PublicationPhaseManifest,
    PublicationRunManifest,
    atomic_write_json,
    canonical_json,
)
from .contract import (
    PUBLICATION_SPEC_SHA256,
    PublicationContractError,
    PublicationInputs,
    load_publication_spec,
    repository_root,
    sha256_bytes,
    sha256_file,
    validate_publication_inputs,
)

PUBLICATION_PLAN_SCHEMA = "delphi.publication-oof-plan.v1"
OOF_ARMS = ("v1_faithful", "v1_corrected", "amplitude_mlp")
OOF_SEEDS = (17, 42, 137, 777, 2027)


@dataclass(frozen=True)
class OOFJobPlan:
    job_id: str
    fold: int
    seed: int
    model_kind: str
    config: AxialExperimentConfig
    output_directory: Path
    atlas_path: Path

    def as_mapping(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "fold": self.fold,
            "seed": self.seed,
            "model_kind": self.model_kind,
            "config_sha256": self.config.config_sha256,
            "config": self.config.as_mapping(),
            "output_directory": str(self.output_directory),
            "atlas_path": str(self.atlas_path),
        }


@dataclass(frozen=True)
class OOFExecutionPlan:
    output_root: Path
    dump_root: Path
    device: str
    jobs: tuple[OOFJobPlan, ...]
    schema: str = PUBLICATION_PLAN_SCHEMA
    publication_spec_sha256: str = PUBLICATION_SPEC_SHA256

    def __post_init__(self) -> None:
        if self.schema != PUBLICATION_PLAN_SCHEMA:
            raise PublicationContractError("OOF plan schema mismatch")
        if self.publication_spec_sha256 != PUBLICATION_SPEC_SHA256:
            raise PublicationContractError("OOF plan specification hash mismatch")
        expected = len(OOF_ARMS) * 5 * len(OOF_SEEDS)
        if len(self.jobs) != expected or len({job.job_id for job in self.jobs}) != expected:
            raise PublicationContractError("OOF plan must contain exactly 75 unique jobs")

    @property
    def plan_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.as_mapping()).encode("utf-8"))

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "publication_spec_sha256": self.publication_spec_sha256,
            "output_root": str(self.output_root),
            "dump_root": str(self.dump_root),
            "device": self.device,
            "jobs": [job.as_mapping() for job in self.jobs],
        }


def _ids_sha256(identifiers: Sequence[str]) -> str:
    return sha256_bytes(canonical_json(sorted(identifiers)).encode("utf-8"))


def require_clean_git(repository: str | Path | None = None) -> str:
    """Return the full implementation revision or fail before publication work."""
    root = repository_root() if repository is None else Path(repository).resolve()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicationContractError(f"cannot identify publication implementation: {exc}") from exc
    if status.strip():
        raise PublicationContractError("publication execution requires a clean Git worktree")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise PublicationContractError("publication execution requires a full Git revision")
    return commit


def build_oof_plan(
    inputs: PublicationInputs,
    *,
    output_root: str | Path,
    dump_root: str | Path,
    device: str = "cuda",
) -> OOFExecutionPlan:
    """Build, but do not execute, the frozen 3-arm × 5-fold × 5-seed plan."""
    spec = load_publication_spec()
    seeds = tuple(spec["cross_validation"]["training_seeds"])
    if seeds != OOF_SEEDS or tuple(spec["axis_evaluation"]["arms"]) != OOF_ARMS:
        raise PublicationContractError("publication code and frozen OOF arm/seed policy differ")
    destination = Path(output_root).resolve()
    repository = repository_root()
    try:
        destination.relative_to(repository)
    except ValueError as exc:
        raise PublicationContractError(
            "publication output_root must be inside the repository so run configurations remain portable"
        ) from exc
    raw_root = Path(dump_root).resolve()
    jobs: list[OOFJobPlan] = []
    for model_kind in OOF_ARMS:
        for fold in range(5):
            atlas_path = destination / "oof" / "fold-atlases" / f"fold-{fold}.json"
            atlas_reference = atlas_path.relative_to(repository).as_posix()
            for seed in OOF_SEEDS:
                job_id = f"publication-oof-{model_kind.replace('_', '-')}-f{fold}-s{seed}"
                config = AxialExperimentConfig(
                    run_id=job_id,
                    stage="confirmation",
                    fold=fold,
                    model_kind=model_kind,
                    training=AxialTrainingConfig.frozen(model_kind, seed),
                    catalog_sha256=spec["bindings"]["catalog"]["sha256"],
                    splits_sha256=spec["bindings"]["publication_splits"]["sha256"],
                    cache_configuration_sha256=inputs.cache_configuration_sha256,
                    comparison_spec_sha256=AXIAL_COMPARISON_SPEC_SHA256,
                    atlas_path=atlas_reference,
                )
                jobs.append(
                    OOFJobPlan(
                        job_id=job_id,
                        fold=fold,
                        seed=seed,
                        model_kind=model_kind,
                        config=config,
                        output_directory=destination / "oof" / "runs" / job_id,
                        atlas_path=atlas_path,
                    )
                )
    return OOFExecutionPlan(
        output_root=destination,
        dump_root=raw_root,
        device=device,
        jobs=tuple(jobs),
    )


def create_run_manifest(
    inputs: PublicationInputs, *, output_root: str | Path, implementation_commit: str
) -> PublicationRunManifest:
    spec = load_publication_spec()
    run_id = f"v1-publication-{PUBLICATION_SPEC_SHA256[:12]}-{implementation_commit[:12]}"
    manifest = PublicationRunManifest(
        run_id=run_id,
        implementation_commit=implementation_commit,
        catalog_sha256=spec["bindings"]["catalog"]["sha256"],
        splits_sha256=spec["bindings"]["publication_splits"]["sha256"],
        cache_manifest_sha256=inputs.cache_manifest_sha256,
        cache_configuration_sha256=inputs.cache_configuration_sha256,
        eligible_object_ids_sha256=_ids_sha256(inputs.eligible_object_ids),
        phases={phase: "pending" for phase in PHASES},
    )
    path = Path(output_root) / "publication-run.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicationContractError(f"cannot resume publication manifest: {exc}") from exc
        if existing != manifest.as_mapping():
            if existing.get("implementation_commit") != implementation_commit:
                raise PublicationContractError(
                    "existing publication run uses another implementation commit; "
                    "retain it and use a new output root"
                )
            raise PublicationContractError("existing publication run has different frozen inputs")
    else:
        atomic_write_json(path, manifest.as_mapping(), overwrite=False)
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PublicationContractError(f"cannot read OOF result rows: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicationContractError(f"invalid OOF row {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise PublicationContractError("OOF result rows must be JSON objects")
        rows.append(row)
    return rows


def validate_completed_oof_job(
    job: OOFJobPlan, inputs: PublicationInputs, *, artifact_root: str | Path
) -> tuple[ArtifactReference, ...]:
    """Verify one completed job and its exact 34-object held-out membership."""
    run_directory = job.output_directory
    summary_path = run_directory / "axial-experiment-summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationContractError(f"cannot read completed OOF job {job.job_id}: {exc}") from exc
    expected_summary = {
        "schema": AXIAL_EXPERIMENT_SUMMARY_SCHEMA,
        "run_id": job.job_id,
        "config_sha256": job.config.config_sha256,
        "fold": job.fold,
        "model_kind": job.model_kind,
        "seed": job.seed,
        "evaluation_count": 34,
        "catalog_count": 170,
    }
    for key, expected in expected_summary.items():
        if key == "catalog_count":
            if inputs.object_count != expected:
                raise PublicationContractError("publication input count changed")
        elif summary.get(key) != expected:
            raise PublicationContractError(f"OOF summary mismatch for {job.job_id}: {key}")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PublicationContractError("OOF summary has no artifact hash mapping")
    for filename, expected_hash in artifacts.items():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(expected_hash, str)
            or sha256_file(run_directory / filename) != expected_hash
        ):
            raise PublicationContractError(f"OOF artifact hash mismatch for {job.job_id}")
    rows_path = run_directory / "evaluation-rows.jsonl"
    rows = _read_jsonl(rows_path)
    required_conditions = set(load_publication_spec()["axis_evaluation"]["required_conditions"])
    expected_ids = set(inputs.fold_test_ids[job.fold])
    for condition in required_conditions:
        selected = [row for row in rows if row.get("condition") == condition]
        identifiers = [row.get("object_id") for row in selected]
        if len(selected) != 34 or set(identifiers) != expected_ids or len(set(identifiers)) != 34:
            raise PublicationContractError(
                f"OOF job {job.job_id} is incomplete for condition {condition}"
            )
        if any(
            row.get("model_kind") != job.model_kind or row.get("seed") != job.seed
            for row in selected
        ):
            raise PublicationContractError(f"OOF row identity mismatch for {job.job_id}")
    references = [
        ArtifactReference.from_file(
            summary_path,
            root=artifact_root,
            artifact_id=f"{job.job_id}.summary",
            media_type="application/json",
            role="metrics",
        )
    ]
    for filename in sorted(artifacts):
        path = run_directory / filename
        references.append(
            ArtifactReference.from_file(
                path,
                root=artifact_root,
                artifact_id=f"{job.job_id}.{Path(filename).stem}",
                media_type=(
                    "application/x-ndjson" if filename.endswith(".jsonl") else "application/json"
                ),
                role="predictions" if filename.endswith(".jsonl") else "provenance",
            )
        )
    return tuple(references)


def _phase_status(jobs: Sequence[PhaseJob]) -> str:
    states = {job.status for job in jobs}
    if "failed" in states:
        return "failed"
    if states == {"complete"}:
        return "complete"
    if states == {"pending"}:
        return "pending"
    return "running"


def _write_phase(path: Path, run_id: str, jobs: Sequence[PhaseJob]) -> PublicationPhaseManifest:
    manifest = PublicationPhaseManifest(
        run_id=run_id,
        phase="oof",
        status=_phase_status(jobs),
        jobs=tuple(jobs),
    )
    atomic_write_json(path, manifest.as_mapping())
    return manifest


def _existing_job_state(
    job: OOFJobPlan, inputs: PublicationInputs, output_root: Path
) -> PhaseJob | None:
    if not job.output_directory.exists():
        return None
    if not (job.output_directory / "axial-experiment-summary.json").is_file():
        raise PublicationContractError(
            f"incomplete publication job is retained and will not be overwritten: {job.job_id}"
        )
    references = validate_completed_oof_job(job, inputs, artifact_root=output_root)
    return PhaseJob(job.job_id, "complete", job.config.config_sha256, references)


def execute_oof_plan(
    plan: OOFExecutionPlan,
    inputs: PublicationInputs,
    *,
    repository: str | Path | None = None,
) -> PublicationPhaseManifest:
    """Execute or resume the frozen OOF plan at immutable job boundaries.

    This function is intentionally never called by import or validation. It
    opens outer-fold outcomes and therefore requires an explicit CLI command,
    a clean worktree, and all 170 cache payloads to pass their hashes.
    """
    root = repository_root() if repository is None else Path(repository).resolve()
    commit = require_clean_git(root)
    verified = validate_publication_inputs(
        catalog_path=inputs.catalog_path,
        splits_path=inputs.splits_path,
        cache_manifest_path=inputs.cache_manifest_path,
        verify_cache_files=True,
    )
    if verified != inputs:
        raise PublicationContractError("OOF plan inputs changed after planning")
    run_manifest = create_run_manifest(inputs, output_root=plan.output_root, implementation_commit=commit)
    phase_path = plan.output_root / "phase-oof.json"
    configs_root = plan.output_root / "oof" / "configs"
    runs_root = plan.output_root / "oof" / "runs"
    jobs: list[PhaseJob] = []
    for job in plan.jobs:
        existing = _existing_job_state(job, inputs, plan.output_root)
        jobs.append(
            existing
            if existing is not None
            else PhaseJob(job.job_id, "pending", job.config.config_sha256)
        )
    phase = _write_phase(phase_path, run_manifest.run_id, jobs)
    if phase.status == "complete":
        validate_oof_completeness(plan, inputs)
        return phase
    if phase.status == "failed":
        raise PublicationContractError("publication OOF phase contains a retained failed job")

    for index, job in enumerate(plan.jobs):
        if jobs[index].status == "complete":
            continue
        config_path = configs_root / f"{job.job_id}.json"
        if config_path.exists():
            if sha256_file(config_path) != sha256_bytes(
                (canonical_json(job.config.as_mapping()) + "\n").encode("utf-8")
            ):
                raise PublicationContractError(f"existing OOF config differs: {job.job_id}")
        else:
            atomic_write_json(config_path, job.config.as_mapping(), overwrite=False)
        if not job.atlas_path.exists():
            fit_train_only_atlas(
                cache_manifest_path=inputs.cache_manifest_path,
                splits_path=inputs.splits_path,
                catalog_path=inputs.catalog_path,
                stage="confirmation",
                fold=job.fold,
                output_path=job.atlas_path,
            )
        try:
            run_axial_experiment(
                job.config,
                repository_root=root,
                catalog_path=inputs.catalog_path,
                splits_path=inputs.splits_path,
                cache_manifest_path=inputs.cache_manifest_path,
                output_root=runs_root,
                dump_root=plan.dump_root,
                device=plan.device,
            )
            references = validate_completed_oof_job(job, inputs, artifact_root=plan.output_root)
            jobs[index] = PhaseJob(job.job_id, "complete", job.config.config_sha256, references)
        except Exception as exc:
            jobs[index] = PhaseJob(
                job.job_id,
                "failed",
                job.config.config_sha256,
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_phase(phase_path, run_manifest.run_id, jobs)
            raise
        _write_phase(phase_path, run_manifest.run_id, jobs)
    validate_oof_completeness(plan, inputs)
    return _write_phase(phase_path, run_manifest.run_id, jobs)


def validate_oof_completeness(plan: OOFExecutionPlan, inputs: PublicationInputs) -> None:
    """Require 75 complete jobs and exactly 170 OOF raw rows per arm/seed."""
    raw_membership: dict[tuple[str, int], list[str]] = {
        (arm, seed): [] for arm in OOF_ARMS for seed in OOF_SEEDS
    }
    for job in plan.jobs:
        validate_completed_oof_job(job, inputs, artifact_root=plan.output_root)
        for row in _read_jsonl(job.output_directory / "evaluation-rows.jsonl"):
            if row.get("condition") == "raw":
                raw_membership[(job.model_kind, job.seed)].append(row.get("object_id"))
    expected = set(inputs.eligible_object_ids)
    for identity, identifiers in raw_membership.items():
        if len(identifiers) != 170 or set(identifiers) != expected or len(set(identifiers)) != 170:
            raise PublicationContractError(f"all-170 OOF completeness failed for {identity}")


def plan_from_paths(
    *,
    output_root: str | Path,
    dump_root: str | Path,
    catalog_path: str | Path | None = None,
    splits_path: str | Path | None = None,
    cache_manifest_path: str | Path | None = None,
    device: str = "cuda",
    verify_cache_files: bool = False,
) -> tuple[PublicationInputs, OOFExecutionPlan]:
    inputs = validate_publication_inputs(
        catalog_path=catalog_path,
        splits_path=splits_path,
        cache_manifest_path=cache_manifest_path,
        verify_cache_files=verify_cache_files,
    )
    return inputs, build_oof_plan(
        inputs, output_root=output_root, dump_root=dump_root, device=device
    )


__all__ = [
    "OOF_ARMS",
    "OOF_SEEDS",
    "OOFExecutionPlan",
    "OOFJobPlan",
    "build_oof_plan",
    "create_run_manifest",
    "execute_oof_plan",
    "plan_from_paths",
    "require_clean_git",
    "validate_completed_oof_job",
    "validate_oof_completeness",
]
