"""Tests for immutable publication-environment capture."""

from __future__ import annotations

import hashlib
import json

import pytest

from lc_pipeline.k3 import environment
from lc_pipeline.k3.environment import K3EnvironmentError
from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256


def command_result(command: list[str], *, required: bool) -> dict[str, object]:
    del required
    if command[1:4] == ["-m", "pip", "freeze"]:
        return {
            "command": command,
            "returncode": 0,
            "stdout": "numpy==2.1.3\ntorch==2.5.1",
            "stderr": "",
            "error": None,
        }
    return {
        "command": command,
        "returncode": 0,
        "stdout": "verified command output",
        "stderr": "",
        "error": None,
    }


def test_environment_capture_is_hash_bound_and_immutable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(environment, "_command", command_result)
    summary = environment.capture_publication_environment(
        output_directory=tmp_path,
        repository_root=tmp_path,
        analysis_commit="a" * 40,
        capture_tool_commit="b" * 40,
    )
    manifest = json.loads((tmp_path / "environment.json").read_text())
    freeze = (tmp_path / "pip-freeze.txt").read_bytes()
    assert manifest["schema"] == "delphi.k3-publication-environment.v1"
    assert manifest["protocol_sha256"] == K3_PROTOCOL_SHA256
    assert manifest["analysis_commit"] == "a" * 40
    assert manifest["capture_tool_commit"] == "b" * 40
    assert manifest["pip_freeze"]["sha256"] == hashlib.sha256(freeze).hexdigest()
    assert summary["environment_manifest_sha256"] == hashlib.sha256(
        (tmp_path / "environment.json").read_bytes()
    ).hexdigest()
    with pytest.raises(K3EnvironmentError, match="overwrite"):
        environment.capture_publication_environment(
            output_directory=tmp_path,
            repository_root=tmp_path,
            analysis_commit="c" * 40,
            capture_tool_commit="b" * 40,
        )


def test_environment_capture_rejects_invalid_analysis_commit(tmp_path) -> None:
    with pytest.raises(K3EnvironmentError, match="analysis_commit"):
        environment.capture_publication_environment(
            output_directory=tmp_path,
            repository_root=tmp_path,
            analysis_commit="short",
            capture_tool_commit="b" * 40,
        )
