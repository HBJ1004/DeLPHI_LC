from __future__ import annotations

import json
import subprocess

import pytest

from lc_pipeline.k3.archive import build_publication_archive_index
from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
from repro.package_publication_archive import build_package
from repro.verify_publication_package import (
    PublicationPackageVerificationError,
    verify_package,
)

ANALYSIS_COMMIT = "a" * 40
TOOL_COMMIT = "b" * 40


def _scientific_sources(tmp_path):
    root = tmp_path / "artifact-source"
    environment = root / "release" / "environment"
    environment.mkdir(parents=True)
    (root / "release" / "publication-summary.json").write_text(
        json.dumps(
            {
                "schema": "delphi.k3-publication-summary.v1",
                "protocol_sha256": K3_PROTOCOL_SHA256,
                "implementation_commit": ANALYSIS_COMMIT,
                "promotion_passed": False,
            }
        ),
        encoding="utf-8",
    )
    (environment / "environment.json").write_text(
        json.dumps(
            {
                "schema": "delphi.k3-publication-environment.v1",
                "protocol_sha256": K3_PROTOCOL_SHA256,
                "analysis_commit": ANALYSIS_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    (root / "result.txt").write_text("frozen\n", encoding="utf-8")
    synthetic = tmp_path / "synthetic-source"
    synthetic.mkdir()
    (synthetic / "shard.npz").write_bytes(b"synthetic")
    comparator = tmp_path / "v1-comparators.npz"
    comparator.write_bytes(b"comparator")
    build_publication_archive_index(
        artifact_root=root,
        output=root / "release" / "archive-index.json",
        tool_commit=TOOL_COMMIT,
        external_sources={"synthetic-data": synthetic, "v1-comparator": comparator},
    )
    return root, synthetic, comparator


def _source_repository(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    return source


def test_package_round_trip_and_deterministic_bytes(tmp_path):
    root, synthetic, comparator = _scientific_sources(tmp_path)
    source = _source_repository(tmp_path)
    first = tmp_path / "package-one"
    second = tmp_path / "package-two"
    for output in (first, second):
        build_package(
            artifact_root=root,
            synthetic_data=synthetic,
            v1_comparator=comparator,
            source_root=source,
            output=output,
        )
    first_manifest = json.loads((first / "publication-archive-manifest.json").read_text())
    second_manifest = json.loads((second / "publication-archive-manifest.json").read_text())
    assert first_manifest == second_manifest
    assert verify_package(first, tmp_path / "extracted")["passed"] is True


def test_package_verifier_detects_outer_tampering(tmp_path):
    root, synthetic, comparator = _scientific_sources(tmp_path)
    source = _source_repository(tmp_path)
    package = tmp_path / "package"
    build_package(
        artifact_root=root,
        synthetic_data=synthetic,
        v1_comparator=comparator,
        source_root=source,
        output=package,
    )
    comparator_copy = package / "delphi-k3-v1-comparator-v1.0.0.npz"
    comparator_copy.write_bytes(b"changed")
    with pytest.raises(PublicationPackageVerificationError, match="mismatch"):
        verify_package(package, tmp_path / "extracted")
