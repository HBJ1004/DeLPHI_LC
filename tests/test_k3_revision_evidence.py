import csv
import json
from pathlib import Path

from repro.export_k3_revision_evidence import (
    _fixed_work_descriptive,
    _sanitize_private_paths,
    _timing_descriptive,
)


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
    assert result["median_object_classical_over_guided_time_ratio"] == 0.8
    assert result["objects_with_guided_mean_time_slower_than_classical"] == 170


def test_fixed_work_summary_exposes_capacity_rejections(tmp_path: Path) -> None:
    rows = []
    for index in range(170):
        supported = index < 144
        rows.append(
            {
                "baseline": {
                    "valid_starts": 6 if supported else 0,
                    "iterations": 300 if supported else 0,
                    "success": supported and index < 106,
                    "wall_seconds": 2.0 if supported else 0.1,
                },
                "candidate": {
                    "valid_starts": 6 if supported else 0,
                    "iterations": 300 if supported else 0,
                    "success": supported and index < 119,
                    "wall_seconds": 2.5 if supported else 0.1,
                },
            }
        )
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    result = _fixed_work_descriptive(path)
    assert result["solver_supported_object_count"] == 144
    assert result["solver_capacity_rejection_count"] == 26
    assert result["baseline_supported_recovery_count"] == 106
    assert result["candidate_supported_recovery_count"] == 119
    assert result["supported_wall_time_ratio_baseline_over_candidate"] == 0.8


def test_revision_export_module_has_stable_schema() -> None:
    source = Path("repro/export_k3_revision_evidence.py").read_text(encoding="utf-8")
    assert json.dumps("delphi.k3-publication-revision-export.v1") in source


def test_private_paths_are_sanitized_recursively() -> None:
    value = {
        "linux": "/mnt/d/private/run/convexinv",
        "home": ["/home/person/data/report.json"],
        "windows": r"C:\Users\person\run\result.json",
        "public": "/usr/bin/gcc",
    }
    assert _sanitize_private_paths(value) == {
        "linux": "<local-run-artifact>/convexinv",
        "home": ["<local-run-artifact>/report.json"],
        "windows": "<local-run-artifact>/result.json",
        "public": "/usr/bin/gcc",
    }
