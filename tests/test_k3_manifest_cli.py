"""Run provenance and CLI smoke contract tests."""

from __future__ import annotations

import json

import pytest

from lc_pipeline.k3.cli import main
from lc_pipeline.k3.manifest import (
    K3ManifestError,
    K3RunManifest,
    configuration_sha256,
    write_manifest,
)
from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256


def test_protocol_and_renderer_cli(capsys) -> None:
    assert main(["validate-protocol"]) == 0
    assert K3_PROTOCOL_SHA256 in capsys.readouterr().out
    assert main(["renderer-smoke", "--cases", "1", "--tolerance", "1e-10"]) == 0
    assert "passed" in capsys.readouterr().out


def test_manifest_is_immutable_and_protocol_bound(tmp_path) -> None:
    manifest = K3RunManifest(
        run_id="run",
        stage="synthetic",
        status="planned",
        protocol_sha256=K3_PROTOCOL_SHA256,
        implementation_commit="a" * 40,
        configuration_sha256=configuration_sha256({"x": 1}),
        artifacts=(),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    with pytest.raises(K3ManifestError, match="overwrite"):
        write_manifest(path, manifest)


def test_train_smoke_cli_is_reviewable(tmp_path) -> None:
    output = tmp_path / "smoke-summary.json"
    assert main(["train-smoke", "--objects", "4", "--output", str(output), "--allow-dirty"]) == 0
    assert output.is_file() and output.with_suffix(".pt").is_file() and output.with_suffix(".manifest.json").is_file()
    payload = json.loads(output.read_text())
    assert payload["protocol_sha256"] == K3_PROTOCOL_SHA256
