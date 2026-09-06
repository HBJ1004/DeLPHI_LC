"""Verify and safely extract a downloaded DeLPHI publication package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from lc_pipeline.k3.archive import verify_publication_archive_index
from lc_pipeline.k3.manifest import sha256_file
from repro.package_publication_archive import ARCHIVE_NAMES, SCHEMA


class PublicationPackageVerificationError(ValueError):
    """Raised when an outer archive file or extracted member is unsafe or changed."""


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as exc:
                raise PublicationPackageVerificationError("archive member escapes destination") from exc
            if member.issym() or member.islnk() or not member.isfile():
                raise PublicationPackageVerificationError(
                    f"archive contains a non-regular member: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise PublicationPackageVerificationError(
                    f"cannot read archive member: {member.name}"
                )
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def verify_package(package_directory: Path, extract_to: Path) -> dict[str, object]:
    """Check outer hashes, safely extract data, and verify the frozen inner index."""
    package_directory = package_directory.resolve()
    extract_to = extract_to.resolve()
    if extract_to.exists():
        raise PublicationPackageVerificationError("extraction destination already exists")
    manifest_path = package_directory / "publication-archive-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationPackageVerificationError("cannot read package manifest") from exc
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), list):
        raise PublicationPackageVerificationError("package manifest schema is invalid")
    declared_names: set[str] = set()
    for record in manifest["files"]:
        if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
            raise PublicationPackageVerificationError("package file record is invalid")
        name = record["filename"]
        if name in declared_names or Path(name).name != name:
            raise PublicationPackageVerificationError("package filename is invalid or duplicated")
        declared_names.add(name)
        path = package_directory / name
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise PublicationPackageVerificationError(f"package file mismatch: {name}")

    extract_to.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{extract_to.name}.", dir=extract_to.parent))
    try:
        _safe_extract(package_directory / ARCHIVE_NAMES["artifact-root"], staging)
        _safe_extract(package_directory / ARCHIVE_NAMES["synthetic-data"], staging)
        shutil.copyfile(
            package_directory / ARCHIVE_NAMES["v1-comparator"],
            staging / "v1-comparators.npz",
        )
        os.replace(staging, extract_to)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verification = verify_publication_archive_index(
        artifact_root=extract_to / "artifact-root",
        index=extract_to / "artifact-root" / "release" / "archive-index.json",
        external_sources={
            "synthetic-data": extract_to / "synthetic-data",
            "v1-comparator": extract_to / "v1-comparators.npz",
        },
    )
    if verification["archive_index_sha256"] != manifest.get("archive_index_sha256"):
        raise PublicationPackageVerificationError("inner archive index differs from outer manifest")
    return {
        "schema": "delphi.k3-publication-package-verification.v1",
        "passed": True,
        "source_commit": manifest.get("source_commit"),
        **verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-directory", type=Path, required=True)
    parser.add_argument("--extract-to", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = verify_package(arguments.package_directory, arguments.extract_to)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        if arguments.output.exists():
            raise PublicationPackageVerificationError("verification output already exists")
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
