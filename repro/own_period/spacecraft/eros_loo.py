"""POST HOC EXPLORATORY Eros diagnostic, PROTOCOL.md Addendum 5.

Run with the pinned Python from source-own-period:
    python -B -m repro.own_period.spacecraft.eros_loo --device cpu
Only complete original epochs are removed, before the unchanged tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.config import K3TokenizerConfig
from lc_pipeline.k3.grid import axial_healpix_grid
from repro.network_root import NetworkPath, check_npz, guard_output, json_metadata, switched
from repro.own_period import predict_oof as prediction

sys.dont_write_bytecode = True

OBJECT_ID = "asteroid_1714"
from repro.own_period import DataPath

OWN = DataPath()
SPACECRAFT = NetworkPath("results", "spacecraft", legacy_path=OWN / "spacecraft")
OOF = NetworkPath("results", "headline", "own-period-oof.npz", legacy_path=OWN / "own-period-oof.npz")
EXPECTED_ERROR_DEG = 71.67375588798046
TOLERANCE = 1e-3  # GPU-to-GPU rounding; attempt 2 differed by 1.52e-06 per axis component, 1.0e-04 deg


def binding(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def calendar_date(jd: float) -> str:
    return (datetime(1970, 1, 1) + timedelta(days=jd - 2440587.5)).date().isoformat()


def time_bounds(epochs) -> tuple[float, float]:
    times = [o.time_jd for epoch in epochs for o in epoch.observations]
    return min(times), max(times)


def partition_epochs(epochs, gap_days: float) -> list[tuple]:
    """Join intact lightcurves until the next uncovered gap exceeds the limit."""
    groups, end = [], None
    for epoch in sorted(epochs, key=lambda value: time_bounds((value,))[0]):
        start, stop = time_bounds((epoch,))
        if end is None or start - end > gap_days:
            groups.append([])
        groups[-1].append(epoch)
        end = stop if end is None else max(end, stop)
    return [tuple(group) for group in groups]


def inventory(epochs) -> dict:
    start, end = time_bounds(epochs)
    return {
        "epoch_ids": [epoch.epoch_id for epoch in epochs],
        "lightcurves": len(epochs),
        "points": sum(len(epoch.observations) for epoch in epochs),
        "start_jd": start,
        "end_jd": end,
        "start_date": calendar_date(start),
        "end_date": calendar_date(end),
        "span_days": end - start,
    }


def group_statistics(epochs) -> dict:
    observations = [o for epoch in epochs for o in epoch.observations]
    sun = np.asarray([o.sun_asteroid_ecliptic_j2000_au for o in observations])
    observer = np.asarray([o.observer_asteroid_ecliptic_j2000_au for o in observations])
    sun = sun / np.linalg.norm(sun, axis=1, keepdims=True)
    observer = observer / np.linalg.norm(observer, axis=1, keepdims=True)
    amplitudes = [
        float(np.ptp(-2.5 * np.log10([o.relative_brightness for o in epoch.observations])))
        for epoch in epochs
    ]
    return {
        **inventory(epochs),
        "mean_phase_angle_deg": float(np.degrees(np.arccos(
            np.clip(np.sum(sun * observer, axis=1), -1, 1)
        )).mean()),
        "mean_observer_abs_latitude_deg": float(np.degrees(np.arcsin(
            np.clip(np.abs(observer[:, 2]), 0, 1)
        )).mean()),
        "mean_sun_abs_latitude_deg": float(np.degrees(np.arcsin(
            np.clip(np.abs(sun[:, 2]), 0, 1)
        )).mean()),
        "lightcurve_peak_to_peak_amplitudes_mag": amplitudes,
        "median_peak_to_peak_amplitude_mag": float(np.median(amplitudes)),
    }


def predict(models, epochs, known_period, targets, device):
    """Use the helpers, raw-map ensemble and adjustment from predict_oof.py."""
    example = prediction.K3TrainingExample(
        object_id=OBJECT_ID,
        tokenized=prediction.tokenize_epochs(epochs, known_period=known_period),
        target_axes=targets,
    )
    inputs = {
        key: value.to(device)
        for key, value in prediction.collate_examples((example,)).items()
        if key in prediction.INPUT_KEYS
    }
    grids = np.stack([
        np.asarray(prediction.score_axial_grid(model, inputs, chunk_size=1024), dtype=np.float64)
        for model in models
    ])
    modes = prediction.modes_from_score_grid(prediction.ensemble_score_grids(grids[:, None, :])[0])
    starts = np.asarray([mode.axis_xyz for mode in modes], dtype=np.float64)
    axes, scores = prediction.refine_ensemble_axes(models, inputs, starts)
    return axes, scores, starts, grids, inputs, example.tokenized.observation_time_origin_jd


def diagnostics(models, inputs, grids, axes, reference) -> dict:
    # Standardize each seed against its own complete map, then average seeds.
    means = grids.mean(axis=1)
    scales = grids.std(axis=1, ddof=0)
    if np.any(scales <= 0):
        raise ValueError("a seed score map has zero variance")
    mean_z = ((grids - means[:, None]) / scales[:, None]).mean(axis=0)
    candidates = torch.as_tensor(
        reference[None, None, :], dtype=inputs["phase_features"].dtype,
        device=inputs["phase_features"].device,
    )
    with torch.no_grad():
        reference_scores = np.asarray([
            float(model(**inputs, candidates=candidates).scores[0, 0].cpu()) for model in models
        ])
    reference_z = float(((reference_scores - means) / scales).mean())
    percentile = float(100 * (np.mean(mean_z < reference_z) + 0.5 * np.mean(mean_z == reference_z)))
    grid = axial_healpix_grid(32).vectors
    latitude = np.degrees(np.arcsin(np.clip(np.abs(grid[:, 2]), 0, 1)))
    unit_axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    return {
        "oracle_error_deg": prediction.oracle_at_k_error_deg(axes, reference[None, :], k=3),
        "candidates_lambda_beta_deg": np.column_stack((
            np.degrees(np.arctan2(unit_axes[:, 1], unit_axes[:, 0])) % 360,
            np.degrees(np.arcsin(np.clip(unit_axes[:, 2], -1, 1))),
        )).tolist(),
        "reference_percentile": percentile,
        "score_abs_latitude_pearson_r": float(np.corrcoef(mean_z, latitude)[0, 1]),
        "reference_mean_z_score": reference_z,
        "seed_reference_scores": reference_scores.tolist(),
        "seed_map_means": means.tolist(),
        "seed_map_standard_deviations": scales.tolist(),
    }


def progress(step: int, total: int, note: str) -> None:
    print(f"[{step}/{total}] {note}", flush=True)
    if os.environ.get("JOBS_ID"):
        subprocess.run(["job", "step", str(step), "--of", str(total), "--note", note], check=True)


def expected_baseline():
    check_npz(OOF)
    with np.load(OOF, allow_pickle=False) as archive:
        index = list(archive["object_ids"]).index(OBJECT_ID)
        axes = archive["refined_axes"][index].copy()
        error = float(archive["errors_deg"][index])
        if (not switched() and error != EXPECTED_ERROR_DEG) or int(archive["folds"][index]) != 1:
            raise ValueError("stored baseline differs from the plan")
    return axes, error


def report(result: dict) -> str:
    check = result["baseline_reproduction"]
    lines = [
        "# Eros leave-one-out — POST HOC EXPLORATORY", "",
        "Diagnostic only; no model or manuscript selection. Period fixed at "
        f"{result['period_hours']:.15f} h; fold 1, seeds {result['seeds']}; device {result['device']}.", "",
        "Groups split at gaps >30 days; apparitions split at gaps >365 days. "
        "Original lightcurves are preserved and removed before tokenization; the phase origin "
        "is recalculated by the unchanged tokenizer. Geometry means weight observations equally. "
        "Amplitudes are median lightcurve peak-to-peak values of -2.5 log10(flux).", "",
        "Percentiles use exact reference-axis scores, each seed z-scored with its full "
        "nside=32 map (population SD), then averaged; midrank percentage among 6,144 axes. "
        "Pearson r uses that averaged map and absolute grid latitude. "
        "Candidate extraction/refinement uses the original raw-map ensemble.", "",
        f"Baseline reproduction: **{'PASS' if check['passed'] else 'FAIL'}**; "
        f"DAMIT oracle error {check['actual_error_deg']:.14f}° "
        f"(stored {check['expected_error_deg']:.14f}°; "
        f"difference {check['error_absolute_difference_deg']:.3g}°); "
        f"maximum refined-axis component difference {check['max_axis_component_difference']:.3g}; "
        f"absolute tolerance {TOLERANCE:g}, relative tolerance zero. "
        "The variant errors below use the spacecraft reference, not the DAMIT catalogue poles.", "",
        "Apparitions: " + "; ".join(
            f"{a['apparition_id']} {a['start_date']}–{a['end_date']} "
            f"({a['points']} points, {a['lightcurves']} lightcurves)"
            for a in result["apparitions"]
        ) + ".", "",
        "| Variant | Points | Lightcurves | Oracle error (°) | Candidate 1 (λ, β)° | "
        "Candidate 2 (λ, β)° | Candidate 3 (λ, β)° | Reference percentile (%) | r(score, abs β) |",
        "|---|---:|---:|---:|---|---|---|---:|---:|",
    ]
    for row in result["variants"]:
        if row["status"] != "ok":
            lines.append(f"| {row['variant_id']} ({row['status']}) | {row['points']} | "
                         f"{row['lightcurves']} | — | — | — | — | — | — |")
            continue
        candidates = " | ".join(f"({lam:.2f}, {beta:.2f})" for lam, beta in row["candidates_lambda_beta_deg"])
        lines.append(f"| {row['variant_id']} | {row['points']} | {row['lightcurves']} | "
                     f"{row['oracle_error_deg']:.3f} | {candidates} | "
                     f"{row['reference_percentile']:.3f} | {row['score_abs_latitude_pearson_r']:.4f} |")
    lines += ["", "| Group | Apparition | Points | Lightcurves | Dates | Span (d) | "
              "Mean phase (°) | Mean observer abs β (°) | Mean Sun abs β (°) | Median amplitude (mag) |",
              "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for group in result["groups"]:
        lines.append(f"| {group['group_id']} | {group['apparition_id']} | {group['points']} | "
                     f"{group['lightcurves']} | {group['start_date']}–{group['end_date']} | "
                     f"{group['span_days']:.3f} | {group['mean_phase_angle_deg']:.3f} | "
                     f"{group['mean_observer_abs_latitude_deg']:.3f} | "
                     f"{group['mean_sun_abs_latitude_deg']:.3f} | "
                     f"{group['median_peak_to_peak_amplitude_mag']:.4f} |")
    skipped = [row["variant_id"] + ": " + row["reason"] for row in result["variants"] if row["status"] == "skipped"]
    lines += ["", "Skipped: " + ("; ".join(skipped) if skipped else "none") + ".",
              f"Analysis wall time: {result['wall_seconds']:.3f} s. "
              f"Script: `{result['script']['path']}`. "
              "Full precision values and SHA-256 input bindings are in results.json.", ""]
    return "\n".join(lines)


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    output = SPACECRAFT / "eros-loo"
    guard_output(output)
    check_npz(OOF)
    if output.exists():
        raise FileExistsError(output)
    catalog = {row["object_id"]: row for row in map(
        json.loads, (prediction.DATA / "catalog.jsonl").read_text().splitlines()
    )}
    periods, provenance, _ = prediction.load_periods(str(OWN / "own-periods-170.json"), catalog)
    period = float(periods[OBJECT_ID])
    if period != 5.269258787836131:
        raise ValueError("Eros own period differs from the plan")
    splits = json.loads((prediction.DATA / "publication-splits-v2.3.json").read_text())["folds"]
    if [int(row["fold"]) for row in splits if OBJECT_ID in row["test_ids"]] != [1]:
        raise ValueError("Eros is not exclusively in fold 1")
    source = prediction.load_damit_object(prediction.DUMP, OBJECT_ID)
    targets = np.asarray([s["vector"] for s in catalog[OBJECT_ID]["solutions"]], dtype=np.float32)
    reference_record = next(row for row in json.loads(
        (OWN / "spacecraft/reference-poles.json").read_text()
    )["objects"] if row["object"] == "Eros")
    reference = np.asarray(reference_record["unit_vector"], dtype=np.float64)
    known_period = prediction.KnownPeriod(period, provenance)
    groups = partition_epochs(source.epochs, 30.0)
    apparitions = partition_epochs(source.epochs, 365.0)
    app_of = {e.epoch_id: f"A{i}" for i, app in enumerate(apparitions, 1) for e in app}
    group_rows = []
    for i, group in enumerate(groups, 1):
        if len({app_of[e.epoch_id] for e in group}) != 1:
            raise ValueError("a 30-day group crosses an apparition boundary")
        group_rows.append({"group_id": f"G{i:02d}", "apparition_id": app_of[group[0].epoch_id],
                           **group_statistics(group)})
    variants = [("all", "baseline", source.epochs)]
    for i, group in enumerate(groups, 1):
        excluded = {e.epoch_id for e in group}
        variants.append((f"drop-G{i:02d}", "leave_one_group_out",
                         tuple(e for e in source.epochs if e.epoch_id not in excluded)))
    for i, app in enumerate(apparitions, 1):
        selected = {e.epoch_id for e in app}
        variants.append((f"drop-A{i}", "leave_one_apparition_out",
                         tuple(e for e in source.epochs if e.epoch_id not in selected)))
    for i, app in enumerate(apparitions, 1):
        selected = {e.epoch_id for e in app}
        variants.append((f"only-A{i}", "keep_one_apparition",
                         tuple(e for e in source.epochs if e.epoch_id in selected)))
    models, names, model_bindings = prediction._load_models(
        prediction.MODELS, prediction.MODEL_MANIFEST, (1,), args.device
    )
    if len(models) != len(prediction.K3_OOF_SEEDS):
        raise ValueError("incorrect seed count")
    paths = [OWN / "own-periods-170.json", OOF,
             OWN / "spacecraft/reference-poles.json", OWN / "spacecraft/PROTOCOL.md",
             prediction.DATA / "catalog.jsonl", prediction.DATA / "publication-splits-v2.3.json",
             prediction.MODEL_MANIFEST, prediction.DUMP / f"files/{OBJECT_ID}/lc.txt",
             prediction.DUMP / f"files/{OBJECT_ID}/model_{source.model_id}/spin.txt",
             prediction.DUMP / f"files/{OBJECT_ID}/model_{source.model_id}/shape.txt",
             Path(prediction.__file__)]
    paths += [prediction.MODELS / name for name in names]
    result = {
        **json_metadata(),
        "schema": "delphi.eros-loo.v1", "label": "POST HOC EXPLORATORY",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "object_id": OBJECT_ID, "period_hours": period, "period_provenance": provenance,
        "fold": 1, "seeds": list(prediction.K3_OOF_SEEDS), "device": args.device,
        "threads": torch.get_num_threads(), "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "script": binding(Path(__file__)),
        "inputs": [binding(path) for path in paths], "models": model_bindings,
        "reference": reference_record, "groups": group_rows,
        "apparitions": [{"apparition_id": f"A{i}", **inventory(app)}
                        for i, app in enumerate(apparitions, 1)],
        "minimum": {"source_lightcurves": 1,
                    "observations_per_lightcurve": K3TokenizerConfig().minimum_epoch_observations},
        "definitions": {
            "group": "intact epochs; new group when next start minus current end >30 days",
            "apparition": "intact epochs; new apparition when next start minus current end >365 days",
            "percentile": "100 * (fraction grid scores < reference + 0.5 * fraction equal); "
                          "exact reference, per-seed map z-score ddof=0, then seed mean",
            "pearson_r": "seed-averaged z-scored map versus absolute grid latitude in degrees",
            "geometry_means": "observation-weighted arithmetic means of angles in degrees",
            "amplitude": "median across original lightcurves of ptp(-2.5*log10(relative_brightness))",
        }, "variants": [],
    }
    expected_axes, expected_error = expected_baseline()
    progress(0, len(variants), "baseline reproduction")
    for step, (variant_id, kind, epochs) in enumerate(variants, 1):
        variant_started = time.perf_counter()
        counts = {"points": sum(len(e.observations) for e in epochs), "lightcurves": len(epochs)}
        row = {"variant_id": variant_id, "kind": kind, **counts}
        if not epochs or any(len(e.observations) < result["minimum"]["observations_per_lightcurve"] for e in epochs):
            row.update(status="skipped", reason="below tokenizer minimum: one epoch, two observations per epoch")
            result["variants"].append(row)
            progress(step, len(variants), f"{variant_id}: skipped")
            continue
        axes, scores, starts, grids, inputs, origin = predict(models, epochs, known_period, targets, args.device)
        row.update(inventory(epochs), refined_axes=axes.tolist(), refined_scores=scores.tolist(),
                   initial_axes=starts.tolist(), observation_time_origin_jd=origin)
        if variant_id == "all":
            actual_error = prediction.oracle_at_k_error_deg(axes, targets, k=3)
            axis_difference = float(np.max(np.abs(axes - expected_axes)))
            error_difference = abs(actual_error - expected_error)
            passed = axis_difference <= TOLERANCE and error_difference <= TOLERANCE
            result["baseline_reproduction"] = {
                "passed": passed, "absolute_tolerance": TOLERANCE, "relative_tolerance": 0,
                "expected_error_deg": expected_error, "actual_error_deg": actual_error,
                "error_absolute_difference_deg": error_difference,
                "max_axis_component_difference": axis_difference,
                "expected_refined_axes": expected_axes.tolist(), "actual_refined_axes": axes.tolist(),
            }
            print(json.dumps({"baseline_reproduction": result["baseline_reproduction"]}), flush=True)
            if not passed:
                row.update(status="baseline_failed", wall_seconds=time.perf_counter() - variant_started)
                result["variants"].append(row)
                break
        row.update(diagnostics(models, inputs, grids, axes, reference), status="ok",
                   wall_seconds=time.perf_counter() - variant_started)
        result["variants"].append(row)
        progress(step, len(variants), f"{variant_id}: oracle {row['oracle_error_deg']:.3f} deg; "
                 f"reference percentile {row['reference_percentile']:.3f}")
    result["wall_seconds"] = time.perf_counter() - started
    result["completed_variants"] = sum(row["status"] == "ok" for row in result["variants"])
    result["planned_variants"] = len(variants)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "results.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (output / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write(report(result))
    if not result["baseline_reproduction"]["passed"]:
        raise RuntimeError("STOP: Eros baseline reproduction failed; no variants evaluated; see results.json")
    print(json.dumps({"output": str(output), "wall_seconds": result["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
