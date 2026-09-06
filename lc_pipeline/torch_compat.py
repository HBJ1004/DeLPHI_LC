"""Compatibility helpers for the supported PyTorch 2.x range."""

from __future__ import annotations

from typing import Any

import torch


def make_grad_scaler(device_type: str, *, enabled: bool) -> Any:
    """Create a gradient scaler on PyTorch versions before and after 2.3."""
    scaler_type = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_type is not None:
        return scaler_type(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
