"""Ensure the classical convex-inversion adapter stays independent of Torch."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lc_pipeline.v2.convexinv import canonical_json, sha256_file


def test_convexinv_import_does_not_import_torch(tmp_path: Path) -> None:
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("convexinv imported Torch")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import lc_pipeline.v2.convexinv as convexinv
assert convexinv.ConvexinvParameters(lambda_deg=0, beta_deg=0, period_hours=6).render()
"""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_lightweight_helpers_preserve_serialization_and_digest_semantics(tmp_path: Path) -> None:
    assert canonical_json({"z": 1, "a": [True, None]}) == '{"a":[true,null],"z":1}'
    with pytest.raises(ValueError):
        canonical_json(float("nan"))

    path = tmp_path / "payload.bin"
    path.write_bytes(b"a small payload\x00\xff")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()

    # The helper output remains directly JSON-compatible for provenance files.
    json.dumps({"digest": sha256_file(path)}, allow_nan=False)
