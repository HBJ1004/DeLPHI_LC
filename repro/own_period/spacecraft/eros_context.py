"""POST HOC EXPLORATORY: how unusual is the Eros photometry within the 170-asteroid sample?

    python -m repro.own_period.spacecraft.eros_context

Per lightcurve with at least 15 points: peak-to-peak amplitude (98th minus 2nd
percentile of -2.5 log10 flux) and solar phase angle at the first point. Per
asteroid: medians over lightcurves. For Eros also: how well the aspect angle of
the spacecraft pole and of each DeLPHI candidate explains the amplitudes
(least-squares fit amplitude ~ 1 + sin^2(aspect) + phase, R^2). Writes
data/own-period-20261021/spacecraft/eros-loo/context.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from repro.own_period import predict_oof as prediction

OWN = prediction.FOLLOWUP / "data/own-period-20261021"
SPACECRAFT = OWN / "spacecraft"
OBJECT_ID = "asteroid_1714"
MIN_POINTS = 15
LARGE_AMPLITUDE = 0.4
LOW_LATITUDE = 20.0


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def lightcurves(object_id: str):
    for epoch in prediction.load_damit_object(prediction.DUMP, object_id).epochs:
        obs = epoch.observations
        if len(obs) < MIN_POINTS:
            continue
        mag = -2.5 * np.log10(np.asarray([o.relative_brightness for o in obs]))
        view = unit(np.mean([unit(o.observer_asteroid_ecliptic_j2000_au) for o in obs], axis=0))
        first_v, first_s = unit(obs[0].observer_asteroid_ecliptic_j2000_au), unit(obs[0].sun_asteroid_ecliptic_j2000_au)
        yield {"amplitude_mag": float(np.percentile(mag, 98) - np.percentile(mag, 2)),
               "phase_deg": float(np.degrees(np.arccos(np.clip(first_v @ first_s, -1, 1)))),
               "view": view}


def r2_amplitude_aspect(rows, axis) -> float:
    amp = np.array([r["amplitude_mag"] for r in rows])
    phase = np.array([r["phase_deg"] for r in rows])
    aspect = np.arccos(np.clip(np.abs([r["view"] @ axis for r in rows]), 0, 1))
    x = np.c_[np.ones(len(amp)), np.sin(aspect) ** 2, phase]
    coef, *_ = np.linalg.lstsq(x, amp, rcond=None)
    return float(1 - ((amp - x @ coef) ** 2).sum() / ((amp - amp.mean()) ** 2).sum())


def main() -> int:
    oof_path = OWN / "own-period-oof.npz"
    with np.load(oof_path) as data:
        ids = [str(x) for x in data["object_ids"]]
        errors = dict(zip(ids, map(float, data["errors_deg"])))
        axes = np.asarray(data["refined_axes"][ids.index(OBJECT_ID)])
    per_object = {}
    for object_id in ids:
        rows = list(lightcurves(object_id))
        if rows:
            per_object[object_id] = {
                "median_amplitude_mag": float(np.median([r["amplitude_mag"] for r in rows])),
                "median_phase_deg": float(np.median([r["phase_deg"] for r in rows])),
                "lightcurves": len(rows), "oof_error_deg": errors[object_id]}
    amp = np.array([v["median_amplitude_mag"] for v in per_object.values()])
    phase = np.array([v["median_phase_deg"] for v in per_object.values()])
    eros = per_object[OBJECT_ID]
    reference = next(r for r in json.loads((SPACECRAFT / "reference-poles.json").read_text())["objects"]
                     if r["object"] == "Eros")
    eros_rows = list(lightcurves(OBJECT_ID))
    similar = {k: v for k, v in per_object.items() if k != OBJECT_ID
               and v["median_amplitude_mag"] >= 0.8 * eros["median_amplitude_mag"]
               and v["median_phase_deg"] >= 0.8 * eros["median_phase_deg"]}
    result = {
        "schema": "delphi.eros-context.v1", "label": "POST HOC EXPLORATORY",
        "script": str(Path(__file__).resolve()), "oof": str(oof_path), "min_points": MIN_POINTS,
        "n_objects": len(per_object),
        "population_median_amplitude_mag": float(np.median(amp)),
        "population_median_phase_deg": float(np.median(phase)),
        "eros": eros,
        "eros_fraction_amplitude_at_least": float((amp >= eros["median_amplitude_mag"]).mean()),
        "eros_fraction_phase_at_least": float((phase >= eros["median_phase_deg"]).mean()),
        "similar_objects_rule": "median amplitude and median phase both >= 0.8 x Eros",
        "similar_objects": similar,
        "eros_amplitude_aspect_r2": {
            "spacecraft_pole": r2_amplitude_aspect(eros_rows, np.asarray(reference["unit_vector"])),
            "delphi_candidates": [r2_amplitude_aspect(eros_rows, unit(a)) for a in axes]},
        "per_object": per_object,
    }
    # Pole latitude of large-amplitude asteroids: mean |beta| over DAMIT models of each asteroid.
    beta: dict[str, list[float]] = {}
    with (prediction.DUMP / "tables/asteroid_models.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            beta.setdefault(f"asteroid_{row['asteroid_id']}", []).append(abs(float(row["beta"])))
    for object_id, value in per_object.items():
        value["mean_abs_beta_deg"] = float(np.mean(beta[object_id]))
    large = [v for v in per_object.values() if v["median_amplitude_mag"] >= LARGE_AMPLITUDE]
    low = [v for v in per_object.values() if v["mean_abs_beta_deg"] < LOW_LATITUDE]
    result["latitude_vs_amplitude"] = {
        "large_amplitude_threshold_mag": LARGE_AMPLITUDE, "low_latitude_threshold_deg": LOW_LATITUDE,
        "n_large_amplitude": len(large),
        "large_amplitude_mean_abs_beta_deg": float(np.mean([v["mean_abs_beta_deg"] for v in large])),
        "all_mean_abs_beta_deg": float(np.mean([v["mean_abs_beta_deg"] for v in per_object.values()])),
        "n_low_latitude": len(low),
        "n_low_latitude_large_amplitude": sum(v["median_amplitude_mag"] >= LARGE_AMPLITUDE for v in low),
        "eros_mean_abs_beta_deg": per_object[OBJECT_ID]["mean_abs_beta_deg"]}
    out = SPACECRAFT / "eros-loo/context.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in result if k != "per_object"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
