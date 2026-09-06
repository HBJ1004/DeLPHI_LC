"""Tests for compatibility across the declared PyTorch 2.x range."""

from __future__ import annotations

import torch

from lc_pipeline.torch_compat import make_grad_scaler


def test_grad_scaler_is_disabled_for_cpu_training() -> None:
    scaler = make_grad_scaler("cpu", enabled=False)
    assert scaler.is_enabled() is False


def test_grad_scaler_falls_back_for_torch_before_2_3(monkeypatch) -> None:
    sentinel = object()

    def legacy_scaler(*, enabled: bool):
        assert enabled is False
        return sentinel

    monkeypatch.delattr(torch.amp, "GradScaler", raising=False)
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", legacy_scaler)
    assert make_grad_scaler("cpu", enabled=False) is sentinel
