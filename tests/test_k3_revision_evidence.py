import csv
import json
from pathlib import Path

from repro.export_k3_revision_evidence import _timing_descriptive


def test_timing_descriptive_uses_only_matching_arm_and_mode(tmp_path: Path) -> None:
    path = tmp_path / "timing.csv"
    fieldnames = (
        "object_id",
        "arm",
        "timing_mode",
        "starts_requested",
        "wall_seconds",
        "inference_and_optional_load_seconds",
        "selected_axis_x",
        "selected_axis_y",
        "selected_axis_z",
    )
    rows = []
    for index in range(170):
        for arm in ("classical15", "classical20", "guided15", "guided20", "standard6"):
            for mode in ("cold", "warm"):
                for repeat in range(3):
                    rows.append(
                        {
                            "object_id": f"asteroid_{index}",
                            "arm": arm,
                            "timing_mode": mode,
                            "starts_requested": 12 if arm == "guided20" else 146,
                            "wall_seconds": 40 if arm == "classical20" and mode == "cold" else 50,
                            "inference_and_optional_load_seconds": 4 if arm == "guided20" else 0,
                            "selected_axis_x": 1,
                            "selected_axis_y": 0,
                            "selected_axis_z": 0,
                        }
                    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = _timing_descriptive(path)
    assert result["guided20_cold_mean_starts"] == 12
    assert result["guided20_cold_mean_inference_and_optional_load_seconds"] == 4
    assert result["classical20_cold_mean_seconds"] == 40
    assert result["classical20_warm_mean_seconds"] == 50
    assert result["selected_poles_identical_across_three_repeats"] is True
    assert result["objects_with_guided_classical_selected_axis_disagreement_over_20_deg"] == 0


def test_revision_export_module_has_stable_schema() -> None:
    source = Path("repro/export_k3_revision_evidence.py").read_text(encoding="utf-8")
    assert json.dumps("delphi.k3-publication-revision-export.v1") in source
