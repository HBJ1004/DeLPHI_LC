"""Post hoc, label-free preprocessing diagnostics; frozen inference is unchanged."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

LIGHT_DAYS_PER_AU = 499.00478383615643 / 86400.0
VARIANTS = {
    "original": (False, False, False),
    "distance": (True, False, False),
    "light_time": (False, True, False),
    "distance_light_time": (True, True, False),
    "geometry_epochs": (False, False, True),
    "distance_geometry_epochs": (True, False, True),
    "light_time_geometry_epochs": (False, True, True),
    "distance_light_time_geometry_epochs": (True, True, True),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new_json(path: Path, value: object) -> None:
    """Never overwrite an earlier evidence file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _vectors(observations: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray([row[key] for row in observations], dtype=float)
    if vectors.shape != (len(observations), 3) or not np.all(np.isfinite(vectors)):
        raise ValueError("geometry must consist of finite three-vectors")
    distances = np.linalg.norm(vectors, axis=1)
    if np.any(distances <= 0):
        raise ValueError("zero geometry vector")
    return vectors / distances[:, None], distances


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {"count": len(values), "minimum": float(np.min(array)),
            "median": float(np.median(array)), "maximum": float(np.max(array))}


def diagnose_object(value: Mapping) -> dict:
    """Inspect photon/phase/geometry structure, without accessing pole references."""
    period_days = float(value["known_period_hours"]) / 24.0
    if not math.isfinite(period_days) or period_days <= 0:
        raise ValueError("invalid known period")
    all_obs = [row for epoch in value["epochs"] for row in epoch["observations"]]
    if not all_obs:
        raise ValueError("no observations")
    times = np.asarray([row["time_jd"] for row in all_obs], dtype=float)
    flux = np.asarray([row["relative_brightness"] for row in all_obs], dtype=float)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(flux)) or np.any(flux <= 0):
        raise ValueError("invalid observation time or brightness")
    _, r = _vectors(all_obs, "sun_asteroid_ecliptic_j2000_au")
    _, delta = _vectors(all_obs, "observer_asteroid_ecliptic_j2000_au")
    origin = float(times.min())
    separations, concentrations, bin_counts, spans = [], [], [], []
    for epoch in value["epochs"]:
        obs = epoch["observations"]
        if not obs:
            continue
        t = np.asarray([row["time_jd"] for row in obs])
        spans.append(float(np.ptp(t)))
        bins = np.minimum((np.remainder((t - origin) / period_days, 1.0) * 64).astype(int), 63)
        dirs = [_vectors(obs, key)[0] for key in ("sun_asteroid_ecliptic_j2000_au", "observer_asteroid_ecliptic_j2000_au")]
        for bin_id in np.unique(bins):
            subset = bins == bin_id
            bin_counts.append(int(subset.sum()))
            for vectors in dirs:
                selected = vectors[subset]
                concentrations.append(float(np.linalg.norm(np.mean(selected, axis=0))))
                if len(selected) > 1:
                    separations.append(float(np.degrees(np.arccos(np.clip(np.min(selected @ selected.T), -1, 1)))))
    errors = [float(row["measured_error"]) / float(row["relative_brightness"])
              for row in all_obs if row.get("measured_error") is not None]
    return {"object_id": value["object_id"], "epoch_count": len(value["epochs"]),
            "observation_count": len(all_obs), "span_days": float(np.ptp(times)),
            "epoch_span_days": _summary(spans), "occupied_bin_counts": _summary(bin_counts),
            "within_bin_directional_diameters_deg": _summary(separations),
            "mixed_bin_direction_count": len(separations),
            "mixed_bin_directions_above_30_deg": sum(s > 30 for s in separations),
            "mean_direction_concentrations": _summary(concentrations),
            "relative_errors": _summary(errors),
            "log_flux_log_distance_product_correlation": _correlation(np.log(flux), np.log(r * delta)),
            "retarded_time_range_days": float(np.ptp(delta * LIGHT_DAYS_PER_AU)),
            "retarded_phase_range_cycles": float(np.ptp(delta * LIGHT_DAYS_PER_AU) / period_days)}


def _geometry_groups(observations: list[dict], span_days: float, angle_deg: float) -> tuple[list[list[dict]], list[dict]]:
    """Chronological greedy groups with an all-pairs angular bound, not mean geometry."""
    if span_days <= 0 or not 0 < angle_deg < 180:
        raise ValueError("invalid geometry-group limits")
    obs = sorted(observations, key=lambda row: row["time_jd"])
    if not obs:
        return [], []
    sun, _ = _vectors(obs, "sun_asteroid_ecliptic_j2000_au")
    observer, _ = _vectors(obs, "observer_asteroid_ecliptic_j2000_au")
    cosine = math.cos(math.radians(angle_deg))
    groups: list[list[int]] = []
    for index, row in enumerate(obs):
        if groups:
            group = groups[-1]
            within_time = row["time_jd"] - obs[group[0]]["time_jd"] <= span_days
            within_angle = all(float(np.min(directions[group] @ directions[index])) >= cosine - 1e-12
                               for directions in (sun, observer))
            if within_time and within_angle:
                group.append(index)
                continue
        groups.append([index])
    retained = [[obs[index] for index in group] for group in groups if len(group) >= 2]
    singletons = [obs[group[0]] for group in groups if len(group) == 1]
    return retained, singletons


def transform_object(value: Mapping, variant: str, *, span_days: float = 30.0, angle_deg: float = 10.0) -> dict:
    """Explicit post hoc factorial interventions, not a silent input-contract change."""
    if variant not in VARIANTS:
        raise ValueError("unknown diagnostic variant")
    if "generalization_transform" in value:
        raise ValueError("input already transformed; refusing possible double correction")
    distance, light_time, regroup = VARIANTS[variant]
    result = copy.deepcopy(dict(value))
    input_count = sum(len(epoch["observations"]) for epoch in result["epochs"])
    original_times = []
    for epoch in result["epochs"]:
        for observation in epoch["observations"]:
            r = float(np.linalg.norm(observation["sun_asteroid_ecliptic_j2000_au"]))
            delta = float(np.linalg.norm(observation["observer_asteroid_ecliptic_j2000_au"]))
            if min(r, delta) <= 0 or not math.isfinite(r * delta):
                raise ValueError("invalid distances")
            original_times.append(float(observation["time_jd"]))
            if distance:
                factor = (r * delta) ** 2
                observation["relative_brightness"] *= factor
                if observation.get("measured_error") is not None:
                    observation["measured_error"] *= factor
            if light_time:
                observation["time_jd"] -= delta * LIGHT_DAYS_PER_AU
    dropped = []
    if regroup:
        epochs = []
        for source in result["epochs"]:
            groups, singletons = _geometry_groups(source["observations"], span_days, angle_deg)
            dropped.extend({"source_epoch": source["epoch_id"], "time_jd": row["time_jd"], "reason": "geometry_group_has_one_observation"}
                           for row in singletons)
            epochs.extend({"epoch_id": f'{source["epoch_id"]}-geometry-{index:04d}', "observations": group}
                          for index, group in enumerate(groups))
        result["epochs"] = epochs
    result["generalization_transform"] = {
        "variant": variant, "role": "post_hoc_development_only", "distance_reduced": distance,
        "light_time_subtracted": light_time, "light_time_days_per_au": LIGHT_DAYS_PER_AU,
        "time_scale_conversion": "none", "retarded_geometry_recomputed": False,
        "phase_function_corrected": False, "geometry_regrouped": regroup,
        "maximum_span_days": span_days if regroup else None,
        "maximum_pairwise_direction_angle_deg": angle_deg if regroup else None,
        "input_observations": input_count,
        "retained_observations": sum(len(epoch["observations"]) for epoch in result["epochs"]),
        "dropped_observations": dropped,
        "usable": bool(result["epochs"]),
    }
    return result


def prepare_diagnostics(manifest_path: Path, spec_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("diagnostics output directory must be new")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["objects"]
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list) or not rows:
        raise ValueError("input manifest requires object rows")
    identifiers = [row["object_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate diagnostic object IDs")
    if manifest.get("expected_object_count", len(rows)) != len(rows):
        raise ValueError("manifest omits intended objects; retain unavailable inputs as failure rows")
    spec = yaml.safe_load(spec_path.read_text())
    if spec.get("schema") != "delphi.k3-generalization-study-spec.v1":
        raise ValueError("invalid generalization study specification")
    grouping = spec["development"]["geometry_epochs"]
    if grouping["minimum_observations"] != 2 or grouping["cross_native_epoch_merge"] is not False:
        raise ValueError("unsupported epoch-grouping policy")
    span_days = float(grouping["maximum_span_days"])
    angle_deg = float(grouping["maximum_pairwise_sun_or_observer_angle_deg"])
    output.mkdir(parents=True)
    # The frozen directory includes object records with authoritative prepared paths/hashes.
    records = []
    for row in rows:
        object_id = row["object_id"]
        if row.get("status", "ready") != "ready":
            records.extend({"object_id": object_id, "variant": variant, "status": "input_unavailable",
                            "reason": row["status"], "path": None, "sha256": None}
                           for variant in VARIANTS)
            continue
        path = manifest_path.parent / row.get("path", f"objects/{object_id}.json")
        if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("prepared path escapes manifest directory")
        expected = row.get("prepared_sha256", row.get("sha256"))
        if expected is None or sha256(path) != expected:
            raise ValueError(f"prepared input hash mismatch: {object_id}")
        value = json.loads(path.read_text())
        if value["object_id"] != object_id:
            raise ValueError("prepared identity mismatch")
        from .survey_identity import validate_survey_binding

        validate_survey_binding(value)
        diagnostic = diagnose_object(value)
        for variant in VARIANTS:
            transformed = transform_object(value, variant, span_days=span_days, angle_deg=angle_deg)
            transformed["generalization_transform"]["source_prepared_sha256"] = sha256(path)
            transformed["generalization_transform"]["study_spec_sha256"] = sha256(spec_path)
            target = output / variant / "objects" / path.name
            write_new_json(target, transformed)
            records.append({"object_id": object_id, "variant": variant, "status": "ready",
                            "source_sha256": sha256(path), "path": target.relative_to(output).as_posix(),
                            "sha256": sha256(target), "transform": transformed["generalization_transform"],
                            "diagnostics": diagnose_object(transformed) if transformed["epochs"] else None})
        write_new_json(output / "original-diagnostics" / path.name, diagnostic)
    receipt = {"schema": "delphi.k3-generalization-diagnostics.v1", "role": "post_hoc_development_only",
               "source_manifest_sha256": sha256(manifest_path), "study_spec_sha256": sha256(spec_path),
               "implementation_sha256": sha256(Path(__file__)), "object_count": len(rows),
               "available_input_count": sum(row.get("status", "ready") == "ready" for row in rows),
               "missing_input_policy": "retain_each_object_in_each_variant_as_failed_prediction",
               "identity_map_sha256": manifest.get("identity_map_sha256"),
               "variants": list(VARIANTS), "records": records}
    write_new_json(output / "manifest.json", receipt)
    return receipt


def invariant_variant_matrix() -> list[tuple[bool, bool, bool]]:
    """Return the exact factorial design, useful for configuration regression tests."""
    return list(itertools.product((False, True), repeat=3))
