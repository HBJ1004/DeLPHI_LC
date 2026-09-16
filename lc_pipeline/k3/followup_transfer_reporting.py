"""Read-only export of sealed follow-up transfer and grid-benchmark records."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import median

import numpy as np

SCHEMA = "delphi.k3-followup-transfer-publication.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_sha(path: Path) -> str:
    """Hash a directory by relative name and content, independent of its location."""
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha(item)))
    return digest.hexdigest()


def _ztf_sampling_summary(objects_dir: Path) -> dict:
    epoch_counts, point_counts, spans = [], [], []
    for path in sorted(objects_dir.glob("*.json")):
        record = _load(path)
        epochs = record["epochs"]
        observations = [item for epoch in epochs for item in epoch["observations"]]
        times = [float(item["time_jd"]) for item in observations]
        epoch_counts.append(len(epochs))
        point_counts.append(len(observations))
        spans.append(max(times) - min(times))
    if not point_counts:
        raise ValueError(f"no prepared ZTF objects found in {objects_dir}")
    return {
        "n_objects": len(point_counts),
        "epochs_min_median_max": [min(epoch_counts), median(epoch_counts), max(epoch_counts)],
        "points_min_median_max": [min(point_counts), median(point_counts), max(point_counts)],
        "span_days_min_median_max": [min(spans), median(spans), max(spans)],
        "construction": "all retained observations for an object form one supplied epoch",
    }


def _grid_angular_comparison(grid: dict) -> dict:
    selected = grid["results"]["selected_axial_errors_by_object"]
    classical = selected["classical20"]["cold"]
    guided = selected["guided20"]["cold"]
    classical_by_id = {row["object_id"]: row for row in classical}
    guided_by_id = {row["object_id"]: row for row in guided}
    if set(classical_by_id) != set(guided_by_id):
        raise ValueError("classical and guided angular cohorts differ")
    rows = []
    for object_id in sorted(classical_by_id):
        base, candidate = classical_by_id[object_id], guided_by_id[object_id]
        rows.append(
            {
                "object_id": object_id,
                "fold": int(base["fold"]),
                "classical_error_deg": float(base["mean_selected_error_degrees"]),
                "guided_error_deg": float(candidate["mean_selected_error_degrees"]),
            }
        )
    differences = np.asarray(
        [row["guided_error_deg"] - row["classical_error_deg"] for row in rows], dtype=float
    )
    rng = np.random.default_rng(20260915)
    by_fold = {
        fold: np.asarray([index for index, row in enumerate(rows) if row["fold"] == fold])
        for fold in sorted({row["fold"] for row in rows})
    }
    boot = np.empty(10_000, dtype=float)
    for draw in range(boot.size):
        indices = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in by_fold.values()]
        )
        boot[draw] = differences[indices].mean()
    return {
        "n_objects": len(rows),
        "guided_minus_classical_mean_error_deg": float(differences.mean()),
        "guided_minus_classical_median_error_deg": float(np.median(differences)),
        "stratified_bootstrap_interval_95_deg": {
            "lower": float(np.percentile(boot, 2.5)),
            "upper": float(np.percentile(boot, 97.5)),
            "resamples": 10_000,
            "seed": 20260915,
        },
        "guided_lower_equal_higher_error_counts": [
            int(np.sum(differences < 0)),
            int(np.sum(differences == 0)),
            int(np.sum(differences > 0)),
        ],
        "within_20_status_changes": int(
            np.sum((np.asarray([r["classical_error_deg"] for r in rows]) <= 20)
                   != (np.asarray([r["guided_error_deg"] for r in rows]) <= 20))
        ),
        "absolute_difference_over_20_deg": int(np.sum(np.abs(differences) > 20)),
        "absolute_difference_over_40_deg": int(np.sum(np.abs(differences) > 40)),
        "objects": rows,
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to overwrite differing export: {path}")
        return
    path.write_text(text, encoding="utf-8", newline="\n")


def _pdf(path: Path, figure) -> None:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    content = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite differing PDF: {path}")
        return
    path.write_bytes(content)


def _variant_rows(score: dict) -> list[dict]:
    rows = []
    for name, value in score["variants"].items():
        available = value["fixed_available_input_cohort"]
        all_intended = value["all_intended_objects"]
        rows.append(
            {
                "study": "mapped_ztf_development_diagnostic",
                "variant": name,
                "cohort": "available_inputs",
                "n_objects": available["n_objects"],
                "mean_error_deg": available["mean_oracle_at_3_error_deg"],
                "failure_count": sum(
                    bool(row.get("failure_reason"))
                    for row in value.get("objects", [])
                    if row.get("input_available")
                ),
                "scope": "same-identity proxy/development diagnostic; not independent transfer",
            }
        )
        rows.append(
            {
                "study": "mapped_ztf_development_diagnostic",
                "variant": name,
                "cohort": "all_intended_90deg_failure_treatment",
                "n_objects": all_intended["n_objects"],
                "mean_error_deg": all_intended["mean_oracle_at_3_error_deg"],
                "failure_count": value["prediction_failure_count"],
                "scope": "same-identity proxy/development diagnostic; failures retained at 90 deg",
            }
        )
    return rows


def export_followup_transfer(data_root: Path, output: Path, comparator_path: Path) -> dict:
    """Export existing JSON records only; never runs prediction, fitting, or scoring."""
    data_root = Path(data_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ztf_path = (
        data_root
        / "generalization-20260910/mapped-ztf-development-diagnostic-score-current-endpoint.json"
    )
    alcdef_path = (
        data_root
        / "generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json"
    )
    grid_path = data_root / "pole-grid-benchmark/20260912-environment-relocked-v3/full-score.json"
    grid_lock_path = data_root / "pole-grid-benchmark/20260912-environment-relocked-v3/lock.json"
    ztf_objects_dir = (
        data_root
        / "generalization-20260910/mapped-ztf/prepared-current-endpoint/objects"
    )
    temporal_dir = data_root / "temporal-scores-final"
    comparator_path = Path(comparator_path)
    for path in (ztf_path, alcdef_path, grid_path, grid_lock_path, comparator_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (ztf_objects_dir, temporal_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    ztf, alcdef, grid = (_load(path) for path in (ztf_path, alcdef_path, grid_path))
    grid_lock = _load(grid_lock_path)
    rows = _variant_rows(ztf)
    with np.load(comparator_path, allow_pickle=False) as comparator:
        comparator_ids = [str(value) for value in comparator["object_ids"]]
        atlas_by_id = dict(zip(comparator_ids, comparator["atlas_errors_deg"].astype(float)))
    ztf_ids = [
        row["object_id"]
        for row in ztf["variants"]["original"]["objects"]
        if row.get("input_available")
    ]
    missing_ztf = sorted(set(ztf_ids) - set(atlas_by_id))
    if missing_ztf:
        raise ValueError(f"ZTF objects absent from comparator artifact: {missing_ztf}")
    ztf_atlas_values = [float(atlas_by_id[object_id]) for object_id in ztf_ids]
    rows.append(
        {
            "study": "mapped_ztf_development_diagnostic",
            "variant": "fixed_train_only_atlas",
            "cohort": "available_inputs",
            "n_objects": len(ztf_atlas_values),
            "mean_error_deg": float(np.mean(ztf_atlas_values)),
            "failure_count": 0,
            "scope": "same-identity reference; fixed train-only three-axis atlas comparator",
        }
    )
    atlas_values = [row["atlas_oracle_at_3_error_deg"] for row in alcdef["objects"]]
    rows.extend(
        (
            {
                "study": "alcdef_gaia_transfer",
                "variant": "candidate",
                "cohort": "locked_30",
                "n_objects": alcdef["secondary"]["n_objects"],
                "mean_error_deg": alcdef["primary"]["value"],
                "failure_count": alcdef["failures"]["n_objects"],
                "scope": "model-training-unexposed; retrospective; references model-derived",
            },
            {
                "study": "alcdef_gaia_transfer",
                "variant": "fixed_train_only_atlas",
                "cohort": "locked_30",
                "n_objects": len(atlas_values),
                "mean_error_deg": sum(atlas_values) / len(atlas_values),
                "failure_count": 0,
                "scope": "fixed train-only three-axis atlas comparator",
            },
            {
                "study": "alcdef_gaia_transfer",
                "variant": "uniform_random_three",
                "cohort": "locked_30",
                "n_objects": alcdef["secondary"]["n_objects"],
                "mean_error_deg": alcdef["uniform_random_three"][
                    "mean_oracle_at_3_error_deg"
                ],
                "failure_count": 0,
                "scope": "uniform random three-axis reference; 10000 trials",
            },
        )
    )
    csv_path = output / "followup-comparison.csv"
    fields = list(rows[0])
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write(csv_path, handle.getvalue())
    # Cold measurements are primary. Warm measurements remain in the sealed source but
    # are not exported as scientific comparisons because they are harness-sensitive.
    standard = grid["results"]["standard6_descriptive"]
    analyses = grid["results"]["analyses"]
    grid_analyses = {
        name: {
            "scope": value["scope"],
            "baseline_arm": value["baseline_arm"],
            "candidate_arm": value["candidate_arm"],
            "object_count": value["recovery"]["object_count"],
            "baseline_success_count": value["recovery"]["baseline_success_count"],
            "candidate_success_count": value["recovery"]["candidate_success_count"],
            "runtime_point_ratio": value["runtime"].get("point_ratio"),
            "runtime_interval_95": value["runtime"].get("percentile_interval_95"),
            "classical_mean_seconds": value["runtime"]["classical_seconds"]["mean"],
            "guided_mean_seconds": value["runtime"]["guided_seconds"]["mean"],
            "recovery_difference_lower": value["recovery"]["simultaneous_exact_lower"],
            "rms_point_ratio": value["rms"].get("point_ratio"),
            "rms_interval_95": value["rms"].get("percentile_interval_95"),
            "passed_all_four_conditions": value["decision"]["passed_all_four_conditions"],
        }
        for name, value in analyses.items()
        if name.startswith("cold_")
    }
    primary = grid_analyses["cold_step20_primary"]
    environment = grid_lock["environment"]
    public_host = {
        "platform": environment["platform"],
        "cpu_models": environment["cpu_models"],
        "cpu_count": environment["cpu_count"],
        "gpu": environment["gpu"],
        "python": environment["python"],
        "packages": environment["packages"],
        "neural_torch_threads": environment["neural_torch_threads"],
        "workers": grid_lock["settings"]["workers"],
    }
    timing = {
        "study": "pole_grid_benchmark_20260912",
        "scope": "170 objects, three repeats; process-cold timing on one recorded host",
        "runtime_point_ratio": primary["runtime_point_ratio"],
        "rms_point_ratio": primary["rms_point_ratio"],
        "recovery_baseline": primary["baseline_success_count"],
        "recovery_guided": primary["candidate_success_count"],
        "standard6_cold": standard["cold"],
        "analyses": grid_analyses,
        "arm_summaries_cold": {
            name: values["cold"] for name, values in grid["results"]["arm_summaries"].items()
        },
        "angular_comparison_step20": _grid_angular_comparison(grid),
        "host": public_host,
        "timing_interval_scope": "object-resampling interval; does not include host-to-host variation",
    }
    temporal = [_load(path) for path in sorted(temporal_dir.glob("asteroid_*.json"))]
    if not temporal:
        raise ValueError("temporal cohort is empty")
    temporal_summary = {
        "n_objects": len(temporal),
        "object_errors_deg": {
            row["object_id"]: float(row["oracle_at_3_error_deg"]) for row in temporal
        },
        "mean_error_deg": float(np.mean([row["oracle_at_3_error_deg"] for row in temporal])),
        "scope": "prespecified descriptive temporal cases; too small for a performance estimate",
    }
    summary = {
        "schema": SCHEMA,
        "ztf_variants": list(ztf["variants"]),
        "comparison_rows": rows,
        "alcdef_scope": alcdef["scope"],
        "ztf_sampling": _ztf_sampling_summary(ztf_objects_dir),
        "temporal_cases": temporal_summary,
        "broad_grid": timing,
    }
    _write(output / "followup-summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    original = ztf["variants"]["original"]
    tex = "".join(
        (
            f"\\newcommand{{\\ZtfUsableMean}}{{{original['fixed_available_input_cohort']['mean_oracle_at_3_error_deg']:.4f}}}\n",
            f"\\newcommand{{\\ZtfAllMean}}{{{original['all_intended_objects']['mean_oracle_at_3_error_deg']:.4f}}}\n",
            f"\\newcommand{{\\ZtfAtlasMean}}{{{np.mean(ztf_atlas_values):.4f}}}\n",
            f"\\newcommand{{\\AlcdefCandidateMean}}{{{alcdef['primary']['value']:.4f}}}\n",
            f"\\newcommand{{\\AlcdefAtlasMean}}{{{sum(atlas_values) / len(atlas_values):.4f}}}\n",
            f"\\newcommand{{\\AlcdefRandomMean}}{{{alcdef['uniform_random_three']['mean_oracle_at_3_error_deg']:.4f}}}\n",
            f"\\newcommand{{\\GridRuntimeRatio}}{{{primary['runtime_point_ratio']:.4f}}}\n",
            f"\\newcommand{{\\GridRmsRatio}}{{{primary['rms_point_ratio']:.4f}}}\n",
            f"\\newcommand{{\\TemporalMean}}{{{temporal_summary['mean_error_deg']:.4f}}}\n",
        )
    )
    _write(output / "followup-macros.tex", tex)
    grid_csv = [
        "analysis,object_count,baseline_successes,candidate_successes,runtime_ratio,rms_ratio,passed_all_four_conditions,scope"
    ]
    for name, value in grid_analyses.items():
        grid_csv.append(
            ",".join(
                str(value[key])
                for key in (
                    "object_count",
                    "baseline_success_count",
                    "candidate_success_count",
                    "runtime_point_ratio",
                    "rms_point_ratio",
                    "passed_all_four_conditions",
                )
            )
            + f",{value['scope']}"
        )
        grid_csv[-1] = name + "," + grid_csv[-1]
    _write(output / "broad-grid-summary.csv", "\n".join(grid_csv) + "\n")
    # PDFs are intentionally small vector plots, derived solely from the exported rows.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.3))
    subset = [row for row in rows if row["study"] == "alcdef_gaia_transfer"]
    survey = [
        (
            f"ZTF usable\n(n={original['fixed_available_input_cohort']['n_objects']})",
            original["fixed_available_input_cohort"]["mean_oracle_at_3_error_deg"],
        ),
        (
            f"ZTF intended\n(n={original['all_intended_objects']['n_objects']}; {original['prediction_failure_count']} failures)",
            original["all_intended_objects"]["mean_oracle_at_3_error_deg"],
        ),
        (f"ZTF atlas\n(n={len(ztf_atlas_values)})", float(np.mean(ztf_atlas_values))),
        (f"ALCDEF candidate\n(n={subset[0]['n_objects']})", subset[0]["mean_error_deg"]),
        (f"ALCDEF atlas\n(n={subset[1]['n_objects']})", subset[1]["mean_error_deg"]),
        (f"ALCDEF random\n(n={subset[2]['n_objects']})", subset[2]["mean_error_deg"]),
    ]
    ax.bar(
        [value[0] for value in survey],
        [value[1] for value in survey],
        color=("#6b8eb5", "#8ca9c7", "#4878a8", "#d28445"),
    )
    ax.set_ylabel("mean oracle-at-3 error (deg)")
    ax.set_title("Survey comparison; ZTF is same-identity development diagnostic")
    fig.tight_layout()
    _pdf(output / "survey-comparison.pdf", fig)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    runtime = analyses["cold_step20_primary"]["runtime"]
    rms = analyses["cold_step20_primary"]["rms"]
    for ax, record, label, color in (
        (axes[0], runtime, "classical / guided runtime", "#4d8b6f"),
        (axes[1], rms, "guided / classical RMS", "#b45b5b"),
    ):
        interval = record["percentile_interval_95"]
        point = record["point_ratio"]
        ax.errorbar(
            [0],
            [point],
            yerr=[[point - interval["lower"]], [interval["upper"] - point]],
            fmt="o",
            color=color,
            capsize=5,
        )
        ax.axhline(1, color="black", linewidth=0.8)
        ax.set_xlim(-0.8, 0.8)
        ax.set_xticks([])
        ax.set_ylabel(label)
        ax.set_title("95% percentile interval")
    fig.suptitle("Broad-grid cold 20-degree primary (170 objects x 3 repeats)")
    fig.tight_layout()
    _pdf(output / "broad-grid-timing-rms.pdf", fig)
    plt.close(fig)
    files = {
        path.name: _sha(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    sources = {
        "ztf_score": _sha(ztf_path),
        "alcdef_analysis": _sha(alcdef_path),
        "broad_grid_score": _sha(grid_path),
        "broad_grid_lock": _sha(grid_lock_path),
        "ztf_prepared_objects_tree": _tree_sha(ztf_objects_dir),
        "temporal_scores_tree": _tree_sha(temporal_dir),
        "comparator_artifact": _sha(comparator_path),
        "exporter_source": _sha(Path(__file__)),
    }
    manifest = {"schema": SCHEMA, "source_sha256": sources, "files": files}
    _write(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
