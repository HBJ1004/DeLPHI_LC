from __future__ import annotations

import csv
import json
from pathlib import Path

from repro.analyze_k3_lowq_lineage import analyze


def _write_sources(root: Path, asteroid_id: str, bibcode: str) -> None:
    directory = root / "files" / asteroid_id
    directory.mkdir(parents=True)
    with (directory / "lc.ref.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("bibcode", "display_label"))
        writer.writeheader()
        writer.writerow({"bibcode": bibcode, "display_label": bibcode})


def test_lineage_uses_union_of_all_fold_training_sources(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    (snapshot / "tables").mkdir(parents=True)
    (snapshot / "tables" / "asteroid_models.csv").write_text("id\n1\n", encoding="utf-8")
    _write_sources(snapshot, "asteroid_1", "TRAIN-A")
    _write_sources(snapshot, "asteroid_2", "TRAIN-B")
    _write_sources(snapshot, "asteroid_10", "TRAIN-B")
    _write_sources(snapshot, "asteroid_11", "NEW-C")
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"folds": [{"train_ids": ["asteroid_1"]}, {"train_ids": ["asteroid_2"]}]}),
        encoding="utf-8",
    )
    census = tmp_path / "census.json"
    census.write_text(
        json.dumps(
            {
                "reference_quality_warning": "lower confidence",
                "objects": [
                    {
                        "object_id": "damit:10",
                        "quality_flag": 1,
                        "oracle_at_3_error_deg": 20.0,
                        "atlas_oracle_at_3_error_deg": 25.0,
                        "observation_count": 20,
                        "epoch_count": 2,
                    },
                    {
                        "object_id": "damit:11",
                        "quality_flag": 2,
                        "oracle_at_3_error_deg": 10.0,
                        "atlas_oracle_at_3_error_deg": 30.0,
                        "observation_count": 30,
                        "epoch_count": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = analyze(snapshot, splits, census, output, bootstrap_resamples=20)
    assert result["lineage_counts"] == {
        "source_disjoint": 1,
        "source_overlap": 1,
        "missing_source": 0,
    }
    assert result["source_disjoint"]["k3"]["mean_deg"] == 10.0
    assert result["source_disjoint"]["paired_atlas_minus_k3_mean_deg"] == 20.0
    assert (output / "objects.csv").is_file()
