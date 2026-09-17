from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro.build_k3_external_atlas import _fold_roles


def test_fold_roles_selects_one_complete_fold(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold": 0,
                        "train_ids": ["asteroid_1"],
                        "validation_ids": ["asteroid_2"],
                        "calibration_ids": ["asteroid_3"],
                        "test_ids": ["asteroid_4"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _fold_roles(path, 0)["train_ids"] == ["asteroid_1"]
    with pytest.raises(ValueError, match="no unique fold"):
        _fold_roles(path, 1)
