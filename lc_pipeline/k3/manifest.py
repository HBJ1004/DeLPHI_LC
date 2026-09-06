"""Immutable, hash-addressed manifests for K3 experiments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .protocol import K3_PROTOCOL_SHA256


class K3ManifestError(ValueError):
    """Raised when an experiment manifest is missing provenance."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_commit(repository_root: str | Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise K3ManifestError(f"cannot determine implementation commit: {exc}") from exc
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise K3ManifestError("git did not return a full commit hash")
    return commit


def require_clean_repository(repository_root: str | Path) -> None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository_root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise K3ManifestError(f"cannot inspect repository state: {exc}") from exc
    if completed.stdout.strip():
        raise K3ManifestError("definitive K3 runs require a clean git worktree")


@dataclass(frozen=True)
class K3ArtifactReference:
    logical_path: str
    sha256: str
    role: str

    @classmethod
    def from_file(cls, path: str | Path, *, root: str | Path, role: str) -> "K3ArtifactReference":
        source = Path(path).resolve()
        base = Path(root).resolve()
        try:
            logical = source.relative_to(base).as_posix()
        except ValueError as exc:
            raise K3ManifestError("artifact must be inside the declared manifest root") from exc
        if not logical or logical.startswith("../") or not role:
            raise K3ManifestError("artifact logical path and role are required")
        return cls(logical, sha256_file(source), role)

    def as_mapping(self) -> dict[str, str]:
        return {"logical_path": self.logical_path, "sha256": self.sha256, "role": self.role}


@dataclass(frozen=True)
class K3RunManifest:
    run_id: str
    stage: str
    status: str
    protocol_sha256: str
    implementation_commit: str
    configuration_sha256: str
    artifacts: tuple[K3ArtifactReference, ...]
    schema: str = "delphi.k3-run-manifest.v1"

    def __post_init__(self) -> None:
        if not self.run_id or not self.stage or self.status not in {"planned", "running", "complete", "failed"}:
            raise K3ManifestError("run_id, stage, and valid status are required")
        for name, digest in (("protocol", self.protocol_sha256), ("configuration", self.configuration_sha256)):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise K3ManifestError(f"{name} digest must be a lowercase SHA-256")
        if len(self.implementation_commit) != 40 or any(character not in "0123456789abcdef" for character in self.implementation_commit):
            raise K3ManifestError("implementation_commit must be a full lowercase git hash")
        if self.protocol_sha256 != K3_PROTOCOL_SHA256:
            raise K3ManifestError("manifest protocol hash differs from the locked K3 protocol")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "stage": self.stage,
            "status": self.status,
            "protocol_sha256": self.protocol_sha256,
            "implementation_commit": self.implementation_commit,
            "configuration_sha256": self.configuration_sha256,
            "artifacts": [artifact.as_mapping() for artifact in self.artifacts],
        }


def configuration_sha256(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(configuration).encode("utf-8")).hexdigest()


def write_manifest(path: str | Path, manifest: K3RunManifest) -> None:
    destination = Path(path)
    if destination.exists():
        raise K3ManifestError("refusing to overwrite an experiment manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(_canonical_json(manifest.as_mapping()) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
