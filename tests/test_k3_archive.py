from __future__ import annotations

import json

import pytest

from lc_pipeline.k3.archive import (
    K3ArchiveError,
    build_publication_archive_index,
    verify_publication_archive_index,
)
from lc_pipeline.k3.manifest import sha256_file
from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256

ANALYSIS_COMMIT = "a" * 40
TOOL_COMMIT = "b" * 40


def _fixture(tmp_path):
    root = tmp_path / "artifacts"
    release = root / "release"
    environment = release / "environment"
    environment.mkdir(parents=True)
    (release / "publication-summary.json").write_text(
        json.dumps(
            {
                "schema": "delphi.k3-publication-summary.v1",
                "protocol_sha256": K3_PROTOCOL_SHA256,
                "implementation_commit": ANALYSIS_COMMIT,
                "promotion_passed": True,
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
    (root / "models").mkdir()
    (root / "models" / "model.pt").write_bytes(b"weights")
    return root


def test_archive_index_hashes_internal_and_external_files(tmp_path):
    root = _fixture(tmp_path)
    external = tmp_path / "comparator.npz"
    external.write_bytes(b"comparison")
    output = root / "release" / "archive-index.json"
    result = build_publication_archive_index(
        artifact_root=root,
        output=output,
        tool_commit=TOOL_COMMIT,
        external_sources={"v1-comparator": external},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    by_path = {entry["logical_path"]: entry for entry in payload["files"]}
    assert "release/archive-index.json" not in by_path
    assert by_path["models/model.pt"]["sha256"] == sha256_file(
        root / "models" / "model.pt"
    )
    assert by_path["external/v1-comparator/comparator.npz"]["sha256"] == sha256_file(
        external
    )
    assert payload["file_count"] == len(payload["files"]) == 4
    assert result["archive_index_sha256"] == sha256_file(output)
    verified = verify_publication_archive_index(
        artifact_root=root,
        index=output,
        external_sources={"v1-comparator": external},
    )
    assert verified["passed"] is True
    assert verified["file_count"] == 4


def test_archive_index_refuses_overwrite_and_provenance_mismatch(tmp_path):
    root = _fixture(tmp_path)
    output = root / "release" / "archive-index.json"
    build_publication_archive_index(
        artifact_root=root, output=output, tool_commit=TOOL_COMMIT
    )
    with pytest.raises(K3ArchiveError, match="overwrite"):
        build_publication_archive_index(
            artifact_root=root, output=output, tool_commit=TOOL_COMMIT
        )

    other = _fixture(tmp_path / "other")
    environment = other / "release" / "environment" / "environment.json"
    value = json.loads(environment.read_text(encoding="utf-8"))
    value["analysis_commit"] = "c" * 40
    environment.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(K3ArchiveError, match="provenance"):
        build_publication_archive_index(
            artifact_root=other,
            output=other / "release" / "archive-index.json",
            tool_commit=TOOL_COMMIT,
        )


def test_archive_index_rejects_symlinks(tmp_path):
    root = _fixture(tmp_path)
    (root / "linked").symlink_to(root / "models" / "model.pt")
    with pytest.raises(K3ArchiveError, match="symlink"):
        build_publication_archive_index(
            artifact_root=root,
            output=root / "release" / "archive-index.json",
            tool_commit=TOOL_COMMIT,
        )


def test_archive_verifier_detects_tampering_and_missing_external_source(tmp_path):
    root = _fixture(tmp_path)
    external = tmp_path / "comparator.npz"
    external.write_bytes(b"comparison")
    output = root / "release" / "archive-index.json"
    build_publication_archive_index(
        artifact_root=root,
        output=output,
        tool_commit=TOOL_COMMIT,
        external_sources={"v1-comparator": external},
    )
    with pytest.raises(K3ArchiveError, match="membership"):
        verify_publication_archive_index(artifact_root=root, index=output)
    (root / "models" / "model.pt").write_bytes(b"tampered")
    with pytest.raises(K3ArchiveError, match="size/hash"):
        verify_publication_archive_index(
            artifact_root=root,
            index=output,
            external_sources={"v1-comparator": external},
        )
