"""Read-only publication exports for the sealed 2026-09-15 reliability run.

This is intentionally not part of the reliability runner: it consumes sealed
receipts and cannot create predictions, fits, or modify a scientific run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .reliability_scoring import stratified_asteroid_bootstrap


class ReliabilityPublicationError(ValueError):
    """A sealed publication input is incomplete or has changed."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unseal(path: Path) -> Any:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = document["payload"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ReliabilityPublicationError(f"cannot read sealed input: {path}") from exc
    if (
        document.get("seal_schema") != "sha256-canonical-json-v1"
        or document.get("payload_sha256") != hashlib.sha256(canonical(payload).encode()).hexdigest()
    ):
        raise ReliabilityPublicationError(f"seal verification failed: {path}")
    return payload


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ReliabilityPublicationError(f"refusing to overwrite differing export: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _load_archived_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["condition"]: row
            for row in csv.DictReader(stream)
            if row["condition"].startswith("two_d-")
        }


def verify_inputs(
    run_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    """Verify seals and every source binding used by the 25-cell export."""
    root = Path(run_root).resolve()
    study = unseal(root / "study.json")
    rows = unseal(root / "scored-rows.json")
    if not isinstance(study, dict) or not isinstance(rows, list):
        raise ReliabilityPublicationError("unexpected sealed reliability schemas")
    objects = {str(x["object_id"]): x for x in study.get("objects", [])}
    # This export uses only the grid cohort.  Verifying unrelated training and
    # solver bindings would make a read-only figure depend on multi-GB inputs.
    grid_ids = {str(value) for value in study.get("grid_ids", [])}
    if len(grid_ids) != 80 or not grid_ids <= set(objects):
        raise ReliabilityPublicationError("invalid grid-eligible cohort in study seal")
    for entry in [objects[oid]["lightcurve"] for oid in sorted(grid_ids)]:
        path = Path(entry["path"])
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ReliabilityPublicationError(f"source binding changed: {path}")
    report_path, report_manifest_path = root / "report.json", root / "report-manifest.json"
    report = unseal(report_path)
    report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
    # The frozen runner attaches file hashes after write_reports fingerprints
    # the numeric report; those hashes are not part of that earlier fingerprint.
    numeric_report = {key: value for key, value in report.items() if key != "report_files"}
    numeric = {
        "rows": rows,
        "report": numeric_report,
        "objects": study.get("objects", []),
        "conditions": study.get("conditions", []),
    }
    report_bytes = json.dumps(
        numeric, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if (
        report_manifest.get("schema") != "delphi.k3-reliability-report.v1"
        or report_manifest.get("numeric_sha256") != hashlib.sha256(report_bytes).hexdigest()
    ):
        raise ReliabilityPublicationError("report numeric manifest mismatch")
    for entry in report["report_files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != entry["sha256"]:
            raise ReliabilityPublicationError("report output binding changed")
    table_path = root / "sampling-table.csv"
    archived = _load_archived_table(table_path)
    selected = [
        r
        for r in rows
        if str(r.get("condition", "")).startswith("two_d-") and r.get("status") != "ineligible"
    ]
    scored = {(r["object_id"], r["condition"], r["repeat"]): r for r in selected}
    # The scored receipt is derived from the sealed prediction receipt.  Check
    # all actual two-dimensional prediction envelopes and their lightcurve hash.
    seen = set()
    for prediction in root.glob("predictions/*/two_d-*/repeat-*.json"):
        receipt = unseal(prediction)
        key = (receipt.get("object_id"), receipt.get("condition"), receipt.get("repeat"))
        if key in seen:
            raise ReliabilityPublicationError(f"duplicate prediction receipt: {prediction}")
        seen.add(key)
        source = objects.get(str(receipt.get("object_id")), {}).get("lightcurve", {})
        if receipt.get("source_sha256") != source.get("sha256"):
            raise ReliabilityPublicationError(f"prediction source hash mismatch: {prediction}")
        row = scored.get(key)
        if (
            row is None
            or row.get("input_sha256") != receipt.get("input_sha256")
            or row.get("source_sha256") != receipt.get("source_sha256")
            or row.get("axes") != receipt.get("axes")
        ):
            raise ReliabilityPublicationError(f"prediction/scored payload mismatch: {prediction}")
    if len(seen) != len(selected):
        raise ReliabilityPublicationError(
            f"prediction/scored receipt count mismatch: {len(seen)} != {len(selected)}"
        )
    hashes = {
        name: sha256(root / name)
        for name in (
            "study.json",
            "scored-rows.json",
            "report.json",
            "report-manifest.json",
            "sampling-table.csv",
        )
    }
    hashes["exporter_source"] = sha256(Path(__file__))
    return study, rows, archived, hashes


def recompute_two_d(
    rows: list[dict[str, Any]],
    archived: Mapping[str, Mapping[str, str]],
    baseline: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Compute each cell as the mean of per-object means over its three repeats."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not str(row.get("condition", "")).startswith("two_d-"):
            continue
        if row.get("status") not in {"ok", "failed"}:
            continue
        grouped[str(row["condition"])][str(row["object_id"])].append(row)
    output = []
    for condition, objects in sorted(grouped.items()):
        repeat_means = []
        counts = []
        objects = dict(sorted(objects.items()))
        for object_rows in objects.values():
            if sorted(int(r["repeat"]) for r in object_rows) != [0, 1, 2]:
                raise ReliabilityPublicationError(f"missing repeat in {condition}")
            repeat_means.append(float(np.mean([float(r["error_deg"]) for r in object_rows])))
            counts.extend(int(r.get("metadata", {}).get("n_points", 0)) for r in object_rows)
        if len(repeat_means) != 80:
            raise ReliabilityPublicationError(
                f"{condition} has {len(repeat_means)}, not 80 grid-eligible objects"
            )
        expected = archived.get(condition)
        if expected is None:
            raise ReliabilityPublicationError(f"archived sampling table lacks {condition}")
        mean = float(np.mean(repeat_means))
        archived_mean = float(expected["mean_error_deg"])
        low, high = (
            float(expected["mean_error_bootstrap_lower_95"]),
            float(expected["mean_error_bootstrap_upper_95"]),
        )
        boot = stratified_asteroid_bootstrap(
            [
                {"object_id": oid, "fold": int(values[0]["fold"]), "value": value}
                for (oid, values), value in zip(objects.items(), repeat_means, strict=True)
            ],
            value_key="value",
            resamples=10_000,
            seed=20260915,
        )
        if (
            not math.isclose(mean, archived_mean, rel_tol=0, abs_tol=1e-10)
            or not math.isclose(boot["bootstrap_lower_95"], low, abs_tol=1e-10)
            or not math.isclose(boot["bootstrap_upper_95"], high, abs_tol=1e-10)
            or int(expected["n_matched"]) != 80
        ):
            raise ReliabilityPublicationError(f"archived statistic mismatch: {condition}")
        # Names are a stable condition API, not a historical-night proxy.
        tokens = condition.replace("two_d-block_count-", "").replace("-per_block_cap-", " ").split()
        block, cap = tokens
        result = {
            "condition": condition,
            "block_count": "all" if block == "None" else int(block),
            "observation_cap_per_block": "all" if cap == "None" else int(cap),
            "n_objects": 80,
            "mean_error_deg": mean,
            "ci95_low_deg": low,
            "ci95_high_deg": high,
            "retained_observations_min": min(counts),
            "retained_observations_median": float(np.median(counts)),
            "retained_observations_max": max(counts),
            "failed_predictions": sum(
                r["status"] == "failed" for values in objects.values() for r in values
            ),
            "scheduled_predictions": 3 * len(objects),
            "archived_mean_error_deg": archived_mean,
        }
        if baseline is not None:
            if set(objects) != set(baseline):
                raise ReliabilityPublicationError("grid baseline and condition object sets differ")
            differences = [
                {"object_id": oid, "fold": int(values[0]["fold"]), "value": value - baseline[oid]}
                for (oid, values), value in zip(objects.items(), repeat_means, strict=True)
            ]
            paired = stratified_asteroid_bootstrap(
                differences, value_key="value", resamples=10_000, seed=20260915
            )
            result.update(
                {
                    "paired_change_deg": float(np.mean([x["value"] for x in differences])),
                    "paired_ci95_low_deg": paired["bootstrap_lower_95"],
                    "paired_ci95_high_deg": paired["bootstrap_upper_95"],
                }
            )
        output.append(result)
    if len(output) != 25:
        raise ReliabilityPublicationError(f"expected 25 two-dimensional cells, found {len(output)}")
    return output


def _csv(rows: list[dict[str, Any]]) -> str:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _tex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"Blocks & Cap/block & Mean error ($^\circ$) & 95\% CI & Retained obs. \\",
        r"\hline",
    ]

    def order(row):
        return tuple(
            float("inf") if row[key] == "all" else int(row[key])
            for key in ("block_count", "observation_cap_per_block")
        )

    for row in sorted(rows, key=order):
        lines.append(
            f"{row['block_count']} & {row['observation_cap_per_block']} & {row['mean_error_deg']:.2f} & [{row['ci95_low_deg']:.2f}, {row['ci95_high_deg']:.2f}] & {row['retained_observations_min']}--{row['retained_observations_max']} "
            + r"\\"
        )
    return "\n".join(lines + [r"\end{tabular}", ""])


def _requirements(study: Mapping[str, Any]) -> list[dict[str, str]]:
    objects = list(study["objects"])

    def span(key: str, suffix: str = "") -> str:
        values = [float(row[key]) for row in objects]
        return f"{min(values):g}--{max(values):g}{suffix} (locked 170-object cohort)"

    return [
        {
            "Input property": "Observations per native session",
            "Software requirement": "At least 2 valid positive-flux observations in each retained native session",
            "Tested range and measured performance": "Native-session-preserving caps; retained counts vary by object/condition (see heatmap table).",
        },
        {
            "Input property": "Observing blocks",
            "Software requirement": "At least one valid native session; block grouping is a diagnostic convention, not an API requirement",
            "Tested range and measured performance": "1, 3, 5, 10, or all blocks; 80 matched asteroids. A block is not necessarily one night or apparition.",
        },
        {
            "Input property": "Total observations",
            "Software requirement": "At least two per retained native session; no performance minimum established",
            "Tested range and measured performance": f"{span('native_observations')}; total caps 10, 20, 50, 100, 200, 500, or all.",
        },
        {
            "Input property": "Time baseline",
            "Software requirement": "Finite observation epochs",
            "Tested range and measured performance": f"{span('time_span_days', ' d')}; descriptive only, not an acceptance threshold.",
        },
        {
            "Input property": "Supplied period",
            "Software requirement": "Finite, positive period in hours",
            "Tested range and measured performance": f"{span('period_hours', ' h')}; sensitivity factors 0.5--2.0, not a period-estimator limit.",
        },
        {
            "Input property": "Observing geometry",
            "Software requirement": "Finite nonzero Sun and observer vectors for every retained observation",
            "Tested range and measured performance": "Required by the native-input API; no geometry-based performance threshold is established.",
        },
    ]


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def baseline_errors(
    rows: list[dict[str, Any]], condition: str, expected_count: int
) -> dict[str, float]:
    selected = [r for r in rows if r["condition"] == condition and r["status"] != "ineligible"]
    output = {r["object_id"]: float(r["error_deg"]) for r in selected}
    if len(output) != expected_count or len(selected) != expected_count:
        raise ReliabilityPublicationError(f"incomplete or duplicated baseline {condition}")
    if any(
        r["status"] not in {"ok", "failed"} or not 0 <= float(r["error_deg"]) <= 90
        for r in selected
    ):
        raise ReliabilityPublicationError(f"invalid baseline {condition}")
    return output


def render_publication(run_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create idempotent vector-PDF, CSV/TeX, macros and provenance manifest."""
    study, rows, archived, hashes = verify_inputs(run_root)
    grid_baseline = baseline_errors(rows, "grid-full", 80)
    full_baseline = baseline_errors(rows, "full", 170)
    cells = recompute_two_d(rows, archived, grid_baseline)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_new(output / "k3_data_degradation.csv", _csv(cells))
    _write_new(output / "k3_data_degradation.tex", _tex_table(cells))
    requirements = _requirements(study)
    _write_new(output / "k3_requirements.csv", _csv(requirements))
    req_tex = [
        r"\begin{tabular}{p{0.22\linewidth}p{0.31\linewidth}p{0.39\linewidth}}",
        r"Input property & Software requirement & Tested range and measured performance \\",
        r"\hline",
    ]
    req_tex += [
        " & ".join(
            latex_escape(r[k])
            for k in (
                "Input property",
                "Software requirement",
                "Tested range and measured performance",
            )
        )
        + r" \\"
        for r in requirements
    ]
    _write_new(output / "k3_requirements.tex", "\n".join(req_tex + [r"\end{tabular}", ""]))
    # Native width does not exceed manuscript text width; labels remain >=9 pt.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    block_order, cap_order = [1, 3, 5, 10, "all"], [5, 10, 20, 50, "all"]
    grid = np.array(
        [
            [
                next(
                    x["mean_error_deg"]
                    for x in cells
                    if x["block_count"] == b and x["observation_cap_per_block"] == c
                )
                for c in cap_order
            ]
            for b in block_order
        ]
    )
    pdf = output / "k3_data_degradation.pdf"
    if not pdf.exists():
        fig, ax = plt.subplots(figsize=(7.1, 5.1))
        image = ax.imshow(grid, cmap="viridis_r", aspect="auto")
        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if grid[i, j] > 30 else "black",
                )
        ax.set_xticks(range(5), cap_order, fontsize=9)
        ax.set_yticks(range(5), block_order, fontsize=9)
        ax.set_xlabel("Observation cap per merged block (all = no cap)", fontsize=10)
        ax.set_ylabel("Merged block count (all = no cap)", fontsize=10)
        ax.set_title(
            "K3 oracle@3 under data degradation\n80 matched asteroids; three repeats per cell",
            fontsize=11,
        )
        bar = fig.colorbar(image, ax=ax)
        bar.set_label("Mean oracle@3 error (degrees)", fontsize=9)
        bar.ax.tick_params(labelsize=9)
        fig.tight_layout()
        fig.savefig(
            pdf,
            format="pdf",
            metadata={
                "Creator": "DeLPHI read-only publication export",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(fig)
    with (Path(run_root) / "sampling-table.csv").open(newline="", encoding="utf-8") as stream:
        baseline = {r["condition"]: r for r in csv.DictReader(stream)}
    grid_mean, full_mean = (
        float(np.mean(list(grid_baseline.values()))),
        float(np.mean(list(full_baseline.values()))),
    )
    for key, actual in (("grid-full", grid_mean), ("full", full_mean)):
        if not math.isclose(
            actual, float(baseline[key]["mean_error_deg"]), rel_tol=0, abs_tol=1e-10
        ):
            raise ReliabilityPublicationError(f"baseline mean differs from archive: {key}")
    macros = "\n".join(
        [
            f"\\newcommand{{\\KThreeGridFullMean}}{{{grid_mean:.6f}}}",
            f"\\newcommand{{\\KThreeFullMean}}{{{full_mean:.6f}}}",
            r"\newcommand{\KThreeGridCohort}{80}",
            "",
        ]
    )
    _write_new(output / "k3_reliability_macros.tex", macros)
    numbers = {
        "schema": "delphi.k3-reliability-publication-numbers.v1",
        "two_d_cells": cells,
        "baseline_grid_full_mean_deg": grid_mean,
        "baseline_full_mean_deg": full_mean,
        "grid_eligible_objects": 80,
        "repeats_per_two_d_cell": 3,
    }
    _write_new(
        output / "k3_reliability_numbers.json", json.dumps(numbers, sort_keys=True, indent=2) + "\n"
    )
    report = "# K3 reliability publication export\n\nAll 25 two-dimensional cells were independently recomputed from sealed receipts as means of three repeats within the 80 grid-eligible asteroids. They match the archived sampling-table means and their archived 95% CIs; the matched cohort is 80 in every cell.\n\nBlocks are native-session intervals merged when overlapping or separated by less than 30 days, not nights or historical apparitions. Caps preserve native sessions with at least two points, so actual retained observation counts can fall below a nominal cap; ranges are in `k3_data_degradation.csv`. No 1.5-degree highlighting, old 50-point threshold, or period-estimator limit is asserted.\n"
    _write_new(output / "report.md", report)
    assets = (
        "k3_data_degradation.csv",
        "k3_data_degradation.tex",
        "k3_data_degradation.pdf",
        "k3_requirements.csv",
        "k3_requirements.tex",
        "k3_reliability_macros.tex",
        "k3_reliability_numbers.json",
        "report.md",
    )
    manifest = {
        "schema": "delphi.k3-reliability-publication.v1",
        "source_sha256": hashes,
        "files": {name: sha256(output / name) for name in assets},
        "reproduction_command": "python-pinned -m repro.export_k3_reliability_publication --run-root <sealed-run> --output <output>",
        "verification": {
            "two_d_cells": 25,
            "grid_eligible_objects": 80,
            "repeats_per_cell": 3,
            "baseline_grid_full_mean_deg": grid_mean,
            "baseline_full_mean_deg": full_mean,
        },
    }
    _write_new(output / "manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest
