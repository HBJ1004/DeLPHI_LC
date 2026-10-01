"""Explore fixed classical budgets from previously saved per-start fits.

This is a post hoc analysis of the already opened 140-object evaluation set.
It cannot replace the frozen 12-start comparison or provide a new holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from run_direct_benchmark import LADDER, classical_starts

from lc_pipeline import grid_benchmark as grid
from lc_pipeline import workflow_benchmark as prior

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path(json.loads((ROOT / "development-report.json").read_text())["source_archive"])
OUTPUT = ROOT / "classical-ladder-exploratory.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lock = json.loads((ROOT / "evaluation-lock.json").read_text())
    scoring = json.loads((ROOT / "scoring-lock.json").read_text())
    score = json.loads((ROOT / "score-loaded.json").read_text())
    references = {}
    with Path(scoring["reference_catalog"]["path"]).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("eligible"):
                references[item["object_id"]] = [s["vector"] for s in item["solutions"]]

    starts = classical_starts()
    grid20 = grid.pole_grid(20)
    indexes = []
    for start in starts[6:]:
        matching = np.where(np.linalg.norm(grid20 - start, axis=1) < 1e-10)[0]
        if len(matching) != 1:
            raise ValueError("classical start missing from saved 20-degree grid")
        indexes.append(int(matching[0]))

    direct = {item["object_id"]: item for item in score["object_rows"]}
    rows = []
    for number, metadata in enumerate(lock["objects"], start=1):
        object_id = metadata["object_id"]
        standard = prior._load_records(prior._archived_case(ARCHIVE, object_id, 0, "standard6"))
        broad = prior._load_records(prior._archived_case(ARCHIVE, object_id, 0, "classical20"))
        if len(standard) != 6 or len(broad) != 146:
            raise ValueError(f"incomplete archived fits for {object_id}")
        records = standard + [broad[index] for index in indexes]
        candidate = {}
        for count in LADDER:
            fits = prior._successful_fits(records[:count], "initial")
            selected = min(fits, key=prior._fit_key) if fits else None
            if selected is None:
                candidate[str(count)] = None
            else:
                candidate[str(count)] = {
                    "axis": selected["axis"],
                    "reference_error_degrees": prior._reference_error(selected["axis"], references[object_id]),
                    "final_relative_rms": selected["final_relative_rms"],
                }
        observed = direct[object_id]["arms"]["classical"]
        saved12 = candidate["12"]
        rows.append(
            {
                "object_id": object_id,
                "fold": metadata["fold"],
                "classical": candidate,
                "delphi": direct[object_id]["arms"]["delphi"],
                "saved12_vs_direct12_axis_degrees": prior.axial_angle_degrees(saved12["axis"], observed["axis"]),
                "saved12_vs_direct12_rms_difference": saved12["final_relative_rms"] - observed["final_relative_rms"],
            }
        )
        if number % 20 == 0:
            print(f"{number}/{len(lock['objects'])} objects", flush=True)

    results = []
    for count in LADDER:
        valid = [row for row in rows if row["classical"][str(count)] is not None]
        differences = [
            row["classical"][str(count)]["reference_error_degrees"] - row["delphi"]["reference_error_degrees"]
            for row in valid
        ]
        within = [
            int(row["classical"][str(count)]["reference_error_degrees"] <= 20)
            - int(row["delphi"]["reference_error_degrees"] <= 20)
            for row in valid
        ]
        rms = [
            row["classical"][str(count)]["final_relative_rms"] / row["delphi"]["final_relative_rms"]
            for row in valid
        ]
        results.append(
            {
                "classical_starts": count,
                "completed_objects": len(valid),
                "classical_mean_reference_error_degrees": float(np.mean([row["classical"][str(count)]["reference_error_degrees"] for row in valid])),
                "delphi_mean_reference_error_degrees": float(np.mean([row["delphi"]["reference_error_degrees"] for row in valid])),
                "classical_minus_delphi_mean_error_degrees": float(np.mean(differences)),
                "classical_within20": sum(row["classical"][str(count)]["reference_error_degrees"] <= 20 for row in valid),
                "delphi_within20": sum(row["delphi"]["reference_error_degrees"] <= 20 for row in valid),
                "classical_minus_delphi_within20_fraction": float(np.mean(within)),
                "classical_over_delphi_geometric_rms_ratio": float(np.exp(np.mean(np.log(rms)))),
            }
        )

    output = {
        "scope": "post_hoc_exploratory_saved_fits_on_previously_opened_evaluation_objects",
        "archive": str(ARCHIVE),
        "evaluation_lock_sha256": digest(ROOT / "evaluation-lock.json"),
        "loaded_score_sha256": digest(ROOT / "score-loaded.json"),
        "validation": {
            "max_saved12_vs_direct12_axis_degrees": max(row["saved12_vs_direct12_axis_degrees"] for row in rows),
            "max_absolute_saved12_vs_direct12_rms_difference": max(abs(row["saved12_vs_direct12_rms_difference"]) for row in rows),
        },
        "budgets": results,
        "object_rows": rows,
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({"validation": output["validation"], "budgets": results}, indent=2))


if __name__ == "__main__":
    main()
