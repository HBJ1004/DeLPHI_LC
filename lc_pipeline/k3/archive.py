"""Portable, complete SHA-256 index for the K3 publication artifact bundle."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping

from .manifest import sha256_file
from .protocol import K3_PROTOCOL_SHA256


class K3ArchiveError(ValueError):
    """Raised when a publication bundle cannot be indexed safely."""


def _load_json(path: Path, schema: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise K3ArchiveError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise K3ArchiveError(f"{path} has the wrong schema")
    return value


def _files(source: Path) -> tuple[Path, ...]:
    if source.is_symlink():
        raise K3ArchiveError(f"archive source may not be a symlink: {source}")
    if source.is_file():
        return (source,)
    if not source.is_dir():
        raise K3ArchiveError(f"archive source does not exist: {source}")
    paths = tuple(sorted(source.rglob("*")))
    symlinks = tuple(path for path in paths if path.is_symlink())
    if symlinks:
        raise K3ArchiveError(f"archive source contains a symlink: {symlinks[0]}")
    return tuple(path for path in paths if path.is_file())


def _entry(path: Path, logical_path: str) -> dict[str, object]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise K3ArchiveError(f"archive source changed while hashing: {path}")
    return {
        "logical_path": logical_path,
        "sha256": digest,
        "size_bytes": after.st_size,
    }


def build_publication_archive_index(
    *,
    artifact_root: str | Path,
    output: str | Path,
    tool_commit: str,
    external_sources: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    """Hash every bundle file plus explicitly named external release sources."""
    root = Path(artifact_root).resolve()
    destination = Path(output).resolve()
    if not root.is_dir():
        raise K3ArchiveError("artifact root must be an existing directory")
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise K3ArchiveError("archive index must be written inside artifact root") from exc
    if destination.exists():
        raise K3ArchiveError("refusing to overwrite publication archive index")
    if len(tool_commit) != 40 or any(character not in "0123456789abcdef" for character in tool_commit):
        raise K3ArchiveError("tool_commit must be a full lowercase Git commit")

    summary_path = root / "release" / "publication-summary.json"
    environment_path = root / "release" / "environment" / "environment.json"
    summary = _load_json(summary_path, "delphi.k3-publication-summary.v1")
    environment = _load_json(
        environment_path, "delphi.k3-publication-environment.v1"
    )
    analysis_commit = summary.get("implementation_commit")
    if (
        summary.get("protocol_sha256") != K3_PROTOCOL_SHA256
        or environment.get("protocol_sha256") != K3_PROTOCOL_SHA256
        or environment.get("analysis_commit") != analysis_commit
        or not isinstance(analysis_commit, str)
        or len(analysis_commit) != 40
        or any(character not in "0123456789abcdef" for character in analysis_commit)
        or not isinstance(summary.get("promotion_passed"), bool)
    ):
        raise K3ArchiveError("summary and environment provenance do not agree")

    entries: list[dict[str, object]] = []
    root_files = _files(root)
    for path in root_files:
        if path.resolve() == destination:
            continue
        entries.append(_entry(path, path.relative_to(root).as_posix()))
    if _files(root) != root_files:
        raise K3ArchiveError("artifact file set changed while building the index")

    labels: set[str] = set()
    for label, raw_source in sorted((external_sources or {}).items()):
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", label) or label in labels:
            raise K3ArchiveError(f"invalid or duplicate external source label: {label!r}")
        labels.add(label)
        source = Path(raw_source).resolve()
        source_files = _files(source)
        if not source_files:
            raise K3ArchiveError(f"external source is empty: {label}")
        for path in source_files:
            suffix = path.name if source.is_file() else path.relative_to(source).as_posix()
            entries.append(_entry(path, f"external/{label}/{suffix}"))
        if _files(source) != source_files:
            raise K3ArchiveError(
                f"external file set changed while building the index: {label}"
            )

    logical_paths = [str(entry["logical_path"]) for entry in entries]
    if not entries or len(logical_paths) != len(set(logical_paths)):
        raise K3ArchiveError("archive index is empty or contains duplicate logical paths")
    entries.sort(key=lambda entry: str(entry["logical_path"]))
    payload: dict[str, object] = {
        "schema": "delphi.k3-publication-archive-index.v1",
        "protocol_sha256": K3_PROTOCOL_SHA256,
        "analysis_commit": analysis_commit,
        "index_tool_commit": tool_commit,
        "promotion_passed": summary.get("promotion_passed"),
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "files": entries,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {**payload, "archive_index_sha256": sha256_file(destination)}


def verify_publication_archive_index(
    *,
    artifact_root: str | Path,
    index: str | Path,
    external_sources: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    """Verify exact file membership, sizes, and hashes against an archive index."""
    root = Path(artifact_root).resolve()
    index_path = Path(index).resolve()
    try:
        index_path.relative_to(root)
    except ValueError as exc:
        raise K3ArchiveError("archive index must be inside artifact root") from exc
    payload = _load_json(index_path, "delphi.k3-publication-archive-index.v1")
    if payload.get("protocol_sha256") != K3_PROTOCOL_SHA256:
        raise K3ArchiveError("archive index protocol hash is invalid")
    raw_entries = payload.get("files")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise K3ArchiveError("archive index has no file entries")
    declared: dict[str, tuple[str, int]] = {}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise K3ArchiveError("archive index contains an invalid file entry")
        logical = raw.get("logical_path")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        if (
            not isinstance(logical, str)
            or not logical
            or logical.startswith("/")
            or ".." in Path(logical).parts
            or logical in declared
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise K3ArchiveError("archive index contains an invalid file entry")
        declared[logical] = (digest, size)

    actual: dict[str, Path] = {}
    for path in _files(root):
        if path.resolve() != index_path:
            actual[path.relative_to(root).as_posix()] = path
    for label, raw_source in sorted((external_sources or {}).items()):
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", label):
            raise K3ArchiveError(f"invalid external source label: {label!r}")
        source = Path(raw_source).resolve()
        source_files = _files(source)
        if not source_files:
            raise K3ArchiveError(f"external source is empty: {label}")
        for path in source_files:
            suffix = path.name if source.is_file() else path.relative_to(source).as_posix()
            logical = f"external/{label}/{suffix}"
            if logical in actual:
                raise K3ArchiveError(f"duplicate archive logical path: {logical}")
            actual[logical] = path

    if set(actual) != set(declared):
        missing = sorted(set(declared) - set(actual))
        extra = sorted(set(actual) - set(declared))
        raise K3ArchiveError(
            f"archive membership mismatch; missing={missing}, extra={extra}"
        )
    total_size = 0
    for logical, path in sorted(actual.items()):
        digest, size = declared[logical]
        stat = path.stat()
        if stat.st_size != size or sha256_file(path) != digest:
            raise K3ArchiveError(f"archive file size/hash mismatch: {logical}")
        total_size += size
    if payload.get("file_count") != len(declared) or payload.get(
        "total_size_bytes"
    ) != total_size:
        raise K3ArchiveError("archive aggregate counts do not match its entries")
    return {
        "schema": "delphi.k3-publication-archive-verification.v1",
        "passed": True,
        "file_count": len(declared),
        "total_size_bytes": total_size,
        "archive_index_sha256": sha256_file(index_path),
    }


__all__ = [
    "K3ArchiveError",
    "build_publication_archive_index",
    "verify_publication_archive_index",
]
