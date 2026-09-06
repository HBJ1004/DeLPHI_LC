"""Reviewable, hash-verified manifests for publication-run artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from .contract import PUBLICATION_SPEC_SHA256, PublicationContractError, sha256_file

PUBLICATION_RUN_SCHEMA = "delphi.publication-run.v1"
PUBLICATION_PHASE_SCHEMA = "delphi.publication-phase-manifest.v1"
ARTIFACT_REFERENCE_SCHEMA = "delphi.publication-artifact-ref.v1"
PhaseName = Literal["oof", "fixed-period", "cone-sensitivity", "end-to-end", "final-model"]
PhaseStatus = Literal["pending", "running", "complete", "failed"]
JobStatus = Literal["pending", "complete", "failed"]
PHASES: tuple[PhaseName, ...] = (
    "oof",
    "fixed-period",
    "cone-sensitivity",
    "end-to-end",
    "final-model",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def atomic_write_json(path: str | Path, value: object, *, overwrite: bool = True) -> str:
    """Write canonical JSON with an atomic rename and return its SHA-256."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise PublicationContractError(f"refusing to overwrite publication artifact: {destination}")
    payload = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not overwrite:
            raise PublicationContractError(
                f"refusing to overwrite publication artifact: {destination}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(destination)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _logical_path(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise PublicationContractError("artifact logical_path must be safe and relative")
    return path.as_posix()


@dataclass(frozen=True)
class ArtifactReference:
    """Portable content identity for a generated or external file."""

    artifact_id: str
    logical_path: str
    sha256: str
    size_bytes: int
    media_type: str
    role: str
    schema: str = ARTIFACT_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ARTIFACT_REFERENCE_SCHEMA:
            raise PublicationContractError("artifact-reference schema mismatch")
        if not self.artifact_id or any(character.isspace() for character in self.artifact_id):
            raise PublicationContractError("artifact_id must be nonempty and contain no whitespace")
        object.__setattr__(self, "logical_path", _logical_path(self.logical_path))
        _digest(self.sha256, "artifact.sha256")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise PublicationContractError("artifact size_bytes must be a nonnegative integer")
        if not self.media_type or not self.role:
            raise PublicationContractError("artifact media_type and role are required")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        root: str | Path,
        artifact_id: str,
        media_type: str,
        role: str,
    ) -> "ArtifactReference":
        source = Path(path).resolve()
        base = Path(root).resolve()
        try:
            relative = source.relative_to(base)
        except ValueError as exc:
            raise PublicationContractError("publication artifact lies outside its declared root") from exc
        if not source.is_file():
            raise PublicationContractError(f"publication artifact is missing: {source}")
        return cls(
            artifact_id=artifact_id,
            logical_path=relative.as_posix(),
            sha256=sha256_file(source),
            size_bytes=source.stat().st_size,
            media_type=media_type,
            role=role,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ArtifactReference":
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise PublicationContractError(f"artifact-reference fields differ: {exc}") from exc

    def verify(self, root: str | Path) -> None:
        path = Path(root) / self.logical_path
        if not path.is_file() or path.stat().st_size != self.size_bytes or sha256_file(path) != self.sha256:
            raise PublicationContractError(f"artifact verification failed: {self.artifact_id}")


@dataclass(frozen=True)
class PublicationRunManifest:
    run_id: str
    implementation_commit: str
    catalog_sha256: str
    splits_sha256: str
    cache_manifest_sha256: str
    cache_configuration_sha256: str
    eligible_object_ids_sha256: str
    phases: Mapping[str, str]
    publication_spec_sha256: str = PUBLICATION_SPEC_SHA256
    schema: str = PUBLICATION_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLICATION_RUN_SCHEMA or not self.run_id:
            raise PublicationContractError("invalid publication-run schema or run_id")
        if (
            len(self.implementation_commit) != 40
            or any(character not in "0123456789abcdef" for character in self.implementation_commit)
        ):
            raise PublicationContractError("implementation_commit must be a full Git commit")
        for name in (
            "publication_spec_sha256",
            "catalog_sha256",
            "splits_sha256",
            "cache_manifest_sha256",
            "cache_configuration_sha256",
            "eligible_object_ids_sha256",
        ):
            _digest(getattr(self, name), name)
        if self.publication_spec_sha256 != PUBLICATION_SPEC_SHA256:
            raise PublicationContractError("publication-run specification hash mismatch")
        if dict(self.phases) != {phase: "pending" for phase in PHASES}:
            raise PublicationContractError("new publication runs must start with every phase pending")

    def as_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseJob:
    job_id: str
    status: JobStatus
    config_sha256: str
    artifacts: tuple[ArtifactReference, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.job_id or self.status not in {"pending", "complete", "failed"}:
            raise PublicationContractError("invalid publication phase job")
        _digest(self.config_sha256, "job.config_sha256")
        if self.status == "pending" and (self.artifacts or self.error is not None):
            raise PublicationContractError("pending jobs cannot have artifacts or an error")
        if self.status == "complete" and (not self.artifacts or self.error is not None):
            raise PublicationContractError("complete jobs require artifacts and no error")
        if self.status == "failed" and (not self.error or self.artifacts):
            raise PublicationContractError("failed jobs require an error and no artifacts")

    def as_mapping(self) -> dict[str, object]:
        value = asdict(self)
        value["artifacts"] = [asdict(item) for item in self.artifacts]
        return value


@dataclass(frozen=True)
class PublicationPhaseManifest:
    run_id: str
    phase: PhaseName
    status: PhaseStatus
    jobs: tuple[PhaseJob, ...]
    publication_spec_sha256: str = PUBLICATION_SPEC_SHA256
    schema: str = PUBLICATION_PHASE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLICATION_PHASE_SCHEMA or self.phase not in PHASES:
            raise PublicationContractError("invalid publication phase manifest")
        if self.status not in {"pending", "running", "complete", "failed"}:
            raise PublicationContractError("invalid publication phase status")
        if self.publication_spec_sha256 != PUBLICATION_SPEC_SHA256:
            raise PublicationContractError("phase manifest publication spec mismatch")
        identifiers = [job.job_id for job in self.jobs]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise PublicationContractError("phase jobs must be nonempty and unique")
        states = {job.status for job in self.jobs}
        if self.status == "pending" and states != {"pending"}:
            raise PublicationContractError("pending phase requires pending jobs")
        if self.status == "complete" and states != {"complete"}:
            raise PublicationContractError("complete phase requires complete jobs")
        if self.status == "failed" and "failed" not in states:
            raise PublicationContractError("failed phase requires a failed job")

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "phase": self.phase,
            "status": self.status,
            "publication_spec_sha256": self.publication_spec_sha256,
            "jobs": [job.as_mapping() for job in self.jobs],
        }


def read_phase_manifest(path: str | Path) -> PublicationPhaseManifest:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        jobs = tuple(
            PhaseJob(
                **{
                    **job,
                    "artifacts": tuple(
                        ArtifactReference.from_mapping(item) for item in job.get("artifacts", [])
                    ),
                }
            )
            for job in value.pop("jobs")
        )
        return PublicationPhaseManifest(jobs=jobs, **value)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        if isinstance(exc, PublicationContractError):
            raise
        raise PublicationContractError(f"cannot read publication phase manifest: {exc}") from exc


def verify_artifacts(references: Sequence[ArtifactReference], root: str | Path) -> None:
    identifiers: set[str] = set()
    for reference in references:
        if reference.artifact_id in identifiers:
            raise PublicationContractError("artifact manifest contains duplicate IDs")
        identifiers.add(reference.artifact_id)
        reference.verify(root)


__all__ = [
    "ARTIFACT_REFERENCE_SCHEMA",
    "PHASES",
    "PUBLICATION_PHASE_SCHEMA",
    "PUBLICATION_RUN_SCHEMA",
    "ArtifactReference",
    "PhaseJob",
    "PublicationPhaseManifest",
    "PublicationRunManifest",
    "atomic_write_json",
    "canonical_json",
    "read_phase_manifest",
    "verify_artifacts",
]
