"""Build deterministic outer packages for the frozen DeLPHI publication archive.

The inner publication index remains the scientific source of truth.  This
module only groups its three source trees into upload-friendly files and adds
an outer checksum manifest for repository downloads.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

from lc_pipeline.k3.archive import verify_publication_archive_index
from lc_pipeline.k3.manifest import sha256_file
from lc_pipeline.version import __version__

SCHEMA = "delphi.k3-publication-package.v1"
ARCHIVE_NAMES = {
    "artifact-root": "delphi-k3-artifacts-v1.0.0.tar.gz",
    "synthetic-data": "delphi-k3-synthetic-data-v1.0.0.tar.gz",
    "source": "delphi-k3-source-v1.0.0.tar.gz",
    "v1-comparator": "delphi-k3-v1-comparator-v1.0.0.npz",
}


class PublicationPackageError(ValueError):
    """Raised when a publication package cannot be built without ambiguity."""


def _regular_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise PublicationPackageError(f"source must be a real directory: {root}")
    entries = tuple(sorted(root.rglob("*")))
    linked = next((path for path in entries if path.is_symlink()), None)
    if linked is not None:
        raise PublicationPackageError(f"source contains a symlink: {linked}")
    return tuple(path for path in entries if path.is_file())


def _git_files(repository: Path) -> tuple[Path, ...]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicationPackageError("cannot enumerate tracked source files") from exc
    paths = tuple(repository / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw)
    if not paths or any(not path.is_file() or path.is_symlink() for path in paths):
        raise PublicationPackageError("tracked source contains a missing or linked file")
    return paths


def _require_clean_git(repository: Path) -> None:
    try:
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicationPackageError("cannot inspect source repository") from exc
    if status:
        raise PublicationPackageError("source repository must be clean before packaging")


def _git_commit(repository: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicationPackageError("cannot resolve source commit") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise PublicationPackageError("source commit is not a full lowercase Git hash")
    return commit


def _snapshot(files: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in files
    )


def _write_tar(source: Path, members: Iterable[Path], root_name: str, output: Path) -> None:
    """Write a byte-reproducible gzip-compressed tar of regular files."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for path in members:
                        relative = path.relative_to(source).as_posix()
                        info = archive.gettarinfo(str(path), arcname=f"{root_name}/{relative}")
                        if not info.isfile():
                            raise PublicationPackageError(f"archive member is not a file: {path}")
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        info.mode = 0o755 if info.mode & 0o111 else 0o644
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _record(path: Path, *, role: str, logical_label: str) -> dict[str, object]:
    return {
        "filename": path.name,
        "logical_label": logical_label,
        "media_type": "application/gzip" if path.name.endswith(".tar.gz") else "application/octet-stream",
        "role": role,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_text() -> str:
    return """# Verify the DeLPHI K3 publication package

Check the outer files first with `publication-archive-manifest.json`. Then
extract the artifact and synthetic-data archives and run:

```bash
delphi-k3 verify-publication-archive-index \\
  --artifact-root artifact-root \\
  --index artifact-root/release/archive-index.json \\
  --external synthetic-data=synthetic-data \\
  --external v1-comparator=delphi-k3-v1-comparator-v1.0.0.npz
```

The verifier checks exact membership, sizes, and SHA-256 values for all frozen
scientific files. The source archive is separately bound by the outer manifest.
"""


def build_package(
    *,
    artifact_root: Path,
    synthetic_data: Path,
    v1_comparator: Path,
    source_root: Path,
    output: Path,
) -> dict[str, object]:
    """Build and verify one complete, upload-ready publication package."""
    if __version__ != "1.0.0":
        raise PublicationPackageError("publication packager requires DeLPHI version 1.0.0")
    artifact_root = artifact_root.resolve()
    synthetic_data = synthetic_data.resolve()
    v1_comparator = v1_comparator.resolve()
    source_root = source_root.resolve()
    output = output.resolve()
    if output.exists():
        raise PublicationPackageError("output already exists")
    if not v1_comparator.is_file() or v1_comparator.is_symlink():
        raise PublicationPackageError("V1 comparator must be a real file")
    _require_clean_git(source_root)
    source_commit = _git_commit(source_root)
    artifact_files = _regular_files(artifact_root)
    synthetic_files = _regular_files(synthetic_data)
    source_files = _git_files(source_root)
    artifact_snapshot = _snapshot(artifact_files)
    synthetic_snapshot = _snapshot(synthetic_files)
    source_snapshot = _snapshot(source_files)
    comparator_snapshot = _snapshot((v1_comparator,))
    index = artifact_root / "release" / "archive-index.json"
    verification = verify_publication_archive_index(
        artifact_root=artifact_root,
        index=index,
        external_sources={
            "synthetic-data": synthetic_data,
            "v1-comparator": v1_comparator,
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        artifact_archive = staging / ARCHIVE_NAMES["artifact-root"]
        synthetic_archive = staging / ARCHIVE_NAMES["synthetic-data"]
        source_archive = staging / ARCHIVE_NAMES["source"]
        comparator_copy = staging / ARCHIVE_NAMES["v1-comparator"]
        _write_tar(artifact_root, artifact_files, "artifact-root", artifact_archive)
        _write_tar(synthetic_data, synthetic_files, "synthetic-data", synthetic_archive)
        _write_tar(source_root, source_files, "delphi-k3-source-v1.0.0", source_archive)
        shutil.copyfile(v1_comparator, comparator_copy)
        _require_clean_git(source_root)
        if (
            _git_commit(source_root) != source_commit
            or _git_files(source_root) != source_files
            or _regular_files(artifact_root) != artifact_files
            or _regular_files(synthetic_data) != synthetic_files
            or _snapshot(artifact_files) != artifact_snapshot
            or _snapshot(synthetic_files) != synthetic_snapshot
            or _snapshot(source_files) != source_snapshot
            or _snapshot((v1_comparator,)) != comparator_snapshot
            or sha256_file(comparator_copy) != sha256_file(v1_comparator)
        ):
            raise PublicationPackageError("a publication source changed during packaging")
        (staging / "VERIFY.md").write_text(_verify_text(), encoding="utf-8")

        summary = json.loads(
            (artifact_root / "release" / "publication-summary.json").read_text(encoding="utf-8")
        )
        records = [
            _record(artifact_archive, role="scientific-artifacts", logical_label="artifact-root"),
            _record(synthetic_archive, role="synthetic-data", logical_label="external/synthetic-data"),
            _record(comparator_copy, role="comparator", logical_label="external/v1-comparator"),
            _record(source_archive, role="source", logical_label="source"),
            _record(staging / "VERIFY.md", role="documentation", logical_label="verification-guide"),
        ]
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "software_version": __version__,
            "source_commit": source_commit,
            "analysis_commit": summary["implementation_commit"],
            "protocol_sha256": summary["protocol_sha256"],
            "archive_index_sha256": verification["archive_index_sha256"],
            "indexed_file_count": verification["file_count"],
            "indexed_total_size_bytes": verification["total_size_bytes"],
            "files": records,
        }
        (staging / "publication-archive-manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--synthetic-data", type=Path, required=True)
    parser.add_argument("--v1-comparator", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = build_package(
        artifact_root=arguments.artifact_root,
        synthetic_data=arguments.synthetic_data,
        v1_comparator=arguments.v1_comparator,
        source_root=arguments.source_root,
        output=arguments.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
