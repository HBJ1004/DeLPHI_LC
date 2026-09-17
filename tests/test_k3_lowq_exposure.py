from __future__ import annotations

import json
from pathlib import Path

from repro.audit_k3_lowq_exposure import audit


def _manifest(path: Path, role: str, donors: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "split": role,
                "geometry_donor_ids": donors,
                "shape_donor_ids": [f"{value}/model_1" for value in donors],
            }
        ),
        encoding="utf-8",
    )


def test_lowq_exposure_separates_donor_identity_from_pole_labels(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "objects": [
                    {"damit_asteroid_id": 1},
                    {"damit_asteroid_id": 2},
                    {"damit_asteroid_id": 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    lineage = tmp_path / "lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": "damit:1",
                        "lineage_status": "structured_reference_overlap",
                    },
                    {
                        "object_id": "damit:2",
                        "lineage_status": "structured_reference_disjoint",
                    },
                    {
                        "object_id": "damit:3",
                        "lineage_status": "missing_structured_reference",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    synthetic = tmp_path / "synthetic"
    _manifest(synthetic / "train" / "train-0000.manifest.json", "train", ["asteroid_1"])
    _manifest(
        synthetic / "validation" / "validation-0000.manifest.json",
        "validation",
        ["asteroid_1", "asteroid_2"],
    )
    _manifest(synthetic / "test" / "test-0000.manifest.json", "test", ["asteroid_3"])
    output = tmp_path / "exposure.json"
    result = audit(index, synthetic, lineage, output)
    assert result["train_or_validation_exposed_count"] == 2
    assert result["roles"]["test"]["lowq_exposed_count"] == 1
    assert result["train_or_validation_lineage_status_counts"] == {
        "structured_reference_disjoint": 1,
        "structured_reference_overlap": 1,
        "missing_structured_reference": 0,
        "not_analyzed": 0,
    }
    assert output.is_file()
