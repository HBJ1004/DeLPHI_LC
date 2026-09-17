from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro.export_k3_grid_timing_cases import _gpu_model


def test_gpu_model_removes_uuid_driver_and_memory() -> None:
    raw = ["NVIDIA GeForce RTX 4070, GPU-01234567-abcd, 591.86, 12282 MiB"]
    assert _gpu_model(raw) == "NVIDIA GeForce RTX 4070"


def test_gpu_model_requires_one_text_record() -> None:
    with pytest.raises(ValueError, match="invalid GPU"):
        _gpu_model([])


def test_source_has_no_hard_coded_private_path() -> None:
    source = Path("repro/export_k3_grid_timing_cases.py").read_text(encoding="utf-8")
    # The regex literal contains these tokens by design; no concrete host path is allowed.
    assert "/mnt/d/Downloads" not in source
    assert "/home/onlyb" not in source
    json.dumps({"source": source})
