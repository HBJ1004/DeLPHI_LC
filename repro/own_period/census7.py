"""Fixed census-seven protocol: CPU preparation/scoring and separate GPU steps.

Run prepare, validate, search, predict own (fold0 then all25), score --stage
own, predict damit (fold0), score --stage final. Only score reads reference
poles; only the last two steps read current reference periods.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lc_pipeline.v2.data import parse_damit_lightcurve
from lc_pipeline.v2.preprocessing import KnownPeriod, Observation, ObservationEpoch

from . import DataPath
from .search_external import FOLLOWUP, curves_of, sha256

SOURCE = Path(__file__).resolve().parents[2]
DATA = DataPath("census7")
PROTOCOL_SHA256 = "51fc1dc2a025db67bf55c06643c48b5b46e04102f46cbf6e8bfee53227426cd3"
# Protocol Amendment 2: new-PC outputs are not bit-identical; axes must agree within this.
AXIS_TOLERANCE_DEG = 0.01
MODEL_SHA256 = "6d041bcbedd9e7ecacc9c4768d0eeb81827065f016dd4db9e982373db3ebfefa"
INDEX_SHA256 = "392d2dda3336bee1c4639b93f90fc1db4225aff57bb30c78d5435bbb0625a4be"
PUBLISHED_SHA256 = "b3adff514191f4f72f325587c64342859224a75333aa9b7ade18bf2e6ae8f861"
MODELS = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/models"
MODEL_MANIFEST = FOLLOWUP / "data/external-models-final.json"
ARCHIVE = MODELS.parent / "release/archive-index.json"
OLD_SNAPSHOT = FOLLOWUP.parent / "damit-20250610T000301Z"
OLD_CENSUS = FOLLOWUP / "source/experiments/lowq_damit_census_20260917"
REFERENCE = FOLLOWUP / "data/damit-postsnapshot-check/raw/asteroid_models.csv"
PROVENANCE = "own-fourier-gpu-top1"
OBJECTS = (
    ("asteroid_10867", 49, "Pales", "new"),
    ("asteroid_10868", 279, "Thule", "new"),
    ("asteroid_10869", 366, "Vincentina", "new"),
    ("asteroid_2772", 215, "Oenone", "upgraded"),
    ("asteroid_2858", 269, "Justitia", "upgraded"),
    ("asteroid_10104", 818, "Kapteynia", "upgraded"),
    ("asteroid_1092", 896, "Sphinx", "upgraded"),
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def binding(path):
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path)}


def checked(record):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"input hash mismatch: {path}")
    return path


def checked_jsonl(record):
    rows = [json.loads(line) for line in checked(record).read_text().splitlines() if line]
    result = {row["object_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate prediction identity")
    return result


def epochs_of(path):
    return tuple(
        ObservationEpoch(
            row["epoch_id"],
            tuple(Observation(**observation) for observation in row["observations"]),
        )
        for row in read(path)["epochs"]
    )


def prepare_lightcurve(source, path):
    epochs = parse_damit_lightcurve(source)
    write(path, {"object_id": path.stem, "epochs": [asdict(epoch) for epoch in epochs]})
    if epochs_of(path) != epochs:
        raise ValueError(f"prepared epoch round trip differs: {source}")
    return epochs


def identity_matches(value, aliases):
    """Donor shape IDs carry a /model suffix; asteroid and MPC IDs differ."""
    return isinstance(value, str) and value.split("/", 1)[0] in aliases


def exposure_audit():
    alias_path = (
        FOLLOWUP / "data/generalization-20260910/synthetic-exposure/full-identity-aliases.json"
    )
    identities = read(alias_path)["objects"]
    split_paths = sorted(
        {
            p.resolve()
            for root in (SOURCE / "repro/data", FOLLOWUP / "inputs/training-run/repro/data")
            for p in root.rglob("*split*.json")
        }
    )
    publication = (
        FOLLOWUP
        / "inputs/training-run/repro/data/damit-20250610T000301Z/publication-splits-v2.3.json"
    )
    if publication.resolve() not in split_paths:
        raise ValueError("missing publication splits")
    # This exact hash is also recorded by the frozen real-OOF evaluations.
    if sha256(publication) != "1c5b15b6577805326a0c36605c174277a1d6653a60c2ac26b717373fe8d61df4":
        raise ValueError("publication split differs from the frozen networks")
    metadata = FOLLOWUP / "data/generalization-20260910/synthetic-exposure/metadata"
    archive = read(ARCHIVE)
    if sha256(ARCHIVE) != read(MODEL_MANIFEST)["archive_index_sha256"]:
        raise ValueError("frozen archive index mismatch")
    members = {r["logical_path"]: r for r in archive["files"]}
    donor_paths = sorted(metadata.rglob("*.manifest.json"))
    expected = {
        k
        for k in members
        if k.startswith("external/synthetic-data/") and k.endswith(".manifest.json")
    }
    if {"external/" + p.relative_to(metadata).as_posix() for p in donor_paths} != expected:
        raise ValueError("synthetic donor inventory incomplete")
    donors = []
    for path in donor_paths:
        if sha256(path) != members["external/" + path.relative_to(metadata).as_posix()]["sha256"]:
            raise ValueError(f"synthetic donor manifest mismatch: {path}")
        donors.append((path, read(path)))

    def hits(value, aliases, location=""):
        if isinstance(value, dict):
            return [hit for k, v in value.items() for hit in hits(v, aliases, f"{location}/{k}")]
        if isinstance(value, list):
            return [hit for i, v in enumerate(value) for hit in hits(v, aliases, f"{location}/{i}")]
        return [location] if identity_matches(value, aliases) else []

    rows = []
    for oid, mpc, name, cohort in OBJECTS:
        aliases = {oid, f"damit:{oid.removeprefix('asteroid_')}", f"mpc:{mpc}"}
        for identity in identities:
            if identity.get("mpc_number") == mpc:
                aliases.update((identity["object_id"], f"damit:{identity['damit_id']}"))
        split_hits = [
            {"path": str(path), "locations": found}
            for path in split_paths
            if (found := hits(read(path), aliases))
        ]
        donor_hits = [
            {"path": str(path), "split": d["split"], "role": role}
            for path, d in donors
            for role in ("geometry_donor_ids", "shape_donor_ids")
            if any(identity_matches(v, aliases) for v in d[role])
        ]
        rows.append(
            {
                "object_id": oid,
                "mpc_number": mpc,
                "name": name,
                "cohort": cohort,
                "aliases_checked": sorted(aliases),
                "split_hits": split_hits,
                "synthetic_donor_hits": donor_hits,
                "exposed": bool(split_hits or donor_hits),
            }
        )
    return {
        "scope": "frozen networks: real splits and all archived synthetic donor manifests",
        "split_files": [binding(p) for p in split_paths],
        "identity_aliases": binding(alias_path),
        "archive_index": binding(ARCHIVE),
        "synthetic_manifests": [binding(p) for p in donor_paths],
        "synthetic_split_counts": dict(Counter(d["split"] for _, d in donors)),
        "objects": rows,
    }


def prepare(out):
    if sha256(DATA / "PROTOCOL.md") != PROTOCOL_SHA256:
        raise ValueError("fixed protocol hash mismatch")
    if sha256(MODEL_MANIFEST) != MODEL_SHA256:
        raise ValueError("frozen model manifest mismatch")
    expected = {
        name: digest
        for line in (DATA / "inputs-sha256.txt").read_text().splitlines()
        for digest, name in [line.split(maxsplit=1)]
    }
    inputs = {name: binding(FOLLOWUP / name) for name in expected}
    if any(inputs[name]["sha256"] != digest for name, digest in expected.items()):
        raise ValueError("protocol input hash mismatch")
    # Hashing reference bytes seals provenance, without parsing periods or poles.
    exposure = exposure_audit()
    rows = []
    for oid, mpc, name, cohort in OBJECTS:
        source = FOLLOWUP / f"data/damit-postsnapshot-check/raw/lc/{oid}.lc.txt"
        path = out / "prepared" / f"{oid}.json"
        epochs = prepare_lightcurve(source, path)
        rows.append(
            {
                "object_id": oid,
                "mpc_number": mpc,
                "name": name,
                "cohort": cohort,
                "raw": binding(source),
                "prepared": binding(path),
                "epoch_count": len(epochs),
                "observation_count": sum(len(e.observations) for e in epochs),
                "epoch_observation_counts": [len(e.observations) for e in epochs],
            }
        )
    old_inputs = []
    for oid, _, _, cohort in OBJECTS:
        if cohort == "upgraded":
            source = OLD_SNAPSHOT / f"files/{oid}/lc.txt"
            path = out / "prepared-validation" / f"{oid}.json"
            prepare_lightcurve(source, path)
            old_inputs.append({"object_id": oid, "raw": binding(source), "prepared": binding(path)})
    write(out / "exposure.json", exposure)
    source_files = [
        Path(__file__),
        SOURCE / "repro/run_k3_lowq_census.py",
        SOURCE / "repro/own_period/predict_external.py",
        SOURCE / "repro/own_period/search_external.py",
        SOURCE / "repro/own_period/score.py",
        SOURCE / "repro/compute_reference_baselines.py",
        SOURCE / "repro/period_search/search.py",
        *(
            SOURCE / f"lc_pipeline/{p}.py"
            for p in (
                "v2/data",
                "k3/bundle",
                "k3/tokenizer",
                "k3/evaluation",
                "k3/inference",
                "k3/ztf_prediction",
            )
        ),
    ]
    models = read(MODEL_MANIFEST)["models"]
    for model in models:
        if sha256(MODELS / model["filename"]) != model["sha256"]:
            raise ValueError(f"frozen checkpoint hash mismatch: {model['filename']}")
    manifest = {
        "schema": "delphi.census7.prepared.v1",
        "protocol": binding(DATA / "PROTOCOL.md"),
        "input_hash_list": binding(DATA / "inputs-sha256.txt"),
        "sealed_inputs": inputs,
        "model_manifest": binding(MODEL_MANIFEST),
        "exposure": binding(out / "exposure.json"),
        "sources": [binding(p) for p in source_files],
        "validation_inputs": old_inputs,
        "validation_index": binding(OLD_CENSUS / "input-index.json"),
        "validation_predictions": binding(
            OLD_CENSUS / "predictions-fold0-five-seed/predictions.jsonl"
        ),
        "objects": rows,
        "reference_values_read": False,
    }
    if (
        manifest["validation_index"]["sha256"] != INDEX_SHA256
        or manifest["validation_predictions"]["sha256"] != PUBLISHED_SHA256
    ):
        raise ValueError("stored lower-quality census differs from the published inputs")
    write(out / "prepared-manifest.json", manifest)
    print(
        json.dumps(
            {
                "prepared_objects": len(rows),
                "exposed": [r["object_id"] for r in exposure["objects"] if r["exposed"]],
            }
        )
    )


def context(out):
    manifest = read(out / "prepared-manifest.json")
    if manifest["protocol"]["sha256"] != PROTOCOL_SHA256:
        raise ValueError("prepared protocol mismatch")
    for record in (
        manifest["protocol"],
        manifest["model_manifest"],
        manifest["exposure"],
        *manifest["sources"],
        *manifest["sealed_inputs"].values(),
    ):
        checked(record)
    for row in manifest["objects"]:
        checked(row["raw"])
        checked(row["prepared"])
    return manifest


def validated(out, manifest):
    validation = read(out / "validation.json")
    if (
        validation.get("passed") is not True
        or validation["prepared_manifest_sha256"] != sha256(out / "prepared-manifest.json")
        or len(validation["objects"]) != len(manifest["validation_inputs"])
    ):
        raise ValueError("GPU validation has not passed for these prepared inputs")
    for row in manifest["validation_inputs"]:
        checked(row["raw"])
        checked(row["prepared"])
    for record in (manifest["validation_index"], manifest["validation_predictions"]):
        checked(record)
    return validation


def load_models(networks, device):
    import torch

    from lc_pipeline.k3.ztf_prediction import _load_models

    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "this subcommand requires CUDA and must be run by Claude under the shared lock"
        )
    models, names, bindings = _load_models(
        MODELS, MODEL_MANIFEST, (0,) if networks == "fold0" else tuple(range(5)), device
    )
    if len(models) != (5 if networks == "fold0" else 25):
        raise ValueError("wrong number of frozen networks")
    return models, bindings


def prediction(models, epochs, period, device):
    from .predict_external import predict

    # Its alcdef branch is exactly the low-Q census sequence: inference-mode
    # grid scoring/averaging/modes, then gradient-enabled ensemble refinement.
    axes, scores = predict(models, epochs, period, device, "alcdef")
    return {
        "status": "ok",
        "period_hours": period.hours,
        "period_provenance": period.provenance,
        "axes": np.asarray(axes).tolist(),
        "scores": np.asarray(scores).tolist(),
        "epoch_count": len(epochs),
        "observation_count": sum(len(e.observations) for e in epochs),
    }


def validate(out, device):
    manifest = context(out)
    models, bindings = load_models("fold0", device)
    # Only the old published periods are used for validation; current reference
    # periods and all reference poles remain sealed.
    index = read(checked(manifest["validation_index"]))
    published = checked_jsonl(manifest["validation_predictions"])
    rows = []
    for row in manifest["validation_inputs"]:
        path = checked(row["raw"])
        oid = row["object_id"]
        old_id = f"damit:{oid.removeprefix('asteroid_')}"
        old_input = next(r for r in index["objects"] if r["object_id"] == old_id)
        old = published[old_id]
        if (OLD_SNAPSHOT / old_input["lightcurve_path"]).resolve() != path:
            raise ValueError(f"old lightcurve path differs: {oid}")
        if old["lightcurve_sha256"] != row["raw"]["sha256"]:
            raise ValueError(f"old lightcurve differs: {oid}")
        new = prediction(
            models,
            epochs_of(checked(row["prepared"])),
            KnownPeriod(float(old_input["known_period_hours"]), old_input["period_provenance"]),
            device,
        )
        old_axes = np.asarray([r["axis_xyz"] for r in old["axes"]], dtype=float)
        new_axes = np.asarray(new["axes"], dtype=float)
        axes_match = np.array_equal(new_axes, old_axes)
        scores_match = np.array_equal(new["scores"], [r["score"] for r in old["axes"]])
        cross = np.linalg.norm(np.cross(new_axes, old_axes), axis=1)
        max_angle = float(np.degrees(np.arcsin(np.clip(cross, 0.0, 1.0))).max())
        max_score_diff = float(
            np.abs(np.asarray(new["scores"]) - [r["score"] for r in old["axes"]]).max()
        )
        passed = (
            old["status"] == new["status"] == "ok"
            and max_angle <= AXIS_TOLERANCE_DEG
            and all(new[k] == old[k] for k in ("epoch_count", "observation_count"))
        )
        rows.append(
            {
                "object_id": oid,
                "axes_exact": axes_match,
                "scores_exact": scores_match,
                "max_axis_angle_deg": max_angle,
                "max_score_difference": max_score_diff,
                "passed": passed,
                "prediction": new,
                "lightcurve": row["raw"],
            }
        )
        print(oid, "exact validation", passed, flush=True)
    result = {
        "passed": all(r["passed"] for r in rows),
        "axis_tolerance_deg": AXIS_TOLERANCE_DEG,
        "objects": rows,
        "checkpoint_bindings": bindings,
        "prepared_manifest_sha256": sha256(out / "prepared-manifest.json"),
    }
    write(out / "validation.json", result)
    if not result["passed"]:
        raise ValueError(
            "validation differs from the stored census; stopping before search or scoring"
        )


def search(out):
    import torch

    from repro.period_search.search import estimate

    manifest = context(out)
    validated(out, manifest)
    if not torch.cuda.is_available():
        raise ValueError("GPU Fourier search must be run by Claude under the shared lock")
    details, periods = {}, {}
    for row in manifest["objects"]:
        oid = row["object_id"]
        started = time.perf_counter()
        if min(row["epoch_observation_counts"]) < 2:
            result = {"status": "input_rejected_published", "ranked_periods": []}
        else:
            try:
                result = estimate(
                    curves_of(checked(row["prepared"])),
                    "fourier_gpu",
                    resolution=1,
                    timeout=180,
                    sparse_fallback=True,
                )
            except TimeoutError as error:
                result = {"status": "timeout", "reason": str(error), "ranked_periods": []}
        ranked = [float(p) for p in (result.get("ranked_periods") or [])]
        periods[oid] = ranked[0] if result.get("status") == "ok" and ranked else None
        details[oid] = {
            k: result.get(k) for k in ("status", "reason", "input_regime", "observing_intervals")
        }
        details[oid].update(ranked_periods=ranked[:5], seconds=time.perf_counter() - started)
        print(oid, details[oid]["status"], periods[oid], flush=True)
    write(
        out / "periods.json",
        {
            "provenance": PROVENANCE,
            "periods_hours": periods,
            "details": details,
            "prepared_manifest_sha256": sha256(out / "prepared-manifest.json"),
            "search_call": 'estimate(curves, "fourier_gpu", resolution=1, timeout=180, sparse_fallback=True)',
        },
    )


def prediction_path(out, periods, networks):
    return out / (
        "predictions-damit.jsonl"
        if periods == "damit"
        else "predictions.jsonl"
        if networks == "fold0"
        else "predictions-all25.jsonl"
    )


def load_predictions(out, periods, networks):
    path = prediction_path(out, periods, networks)
    manifest = read(path.with_suffix(".manifest.json"))
    if (
        manifest["periods"] != periods
        or manifest["networks"] != networks
        or manifest["prepared_manifest_sha256"] != sha256(out / "prepared-manifest.json")
    ):
        raise ValueError("prediction policy/prepared input mismatch")
    checked(manifest["period_input"])
    rows = checked_jsonl(manifest["predictions"])
    if set(rows) != {r[0] for r in OBJECTS}:
        raise ValueError("prediction cohort mismatch")
    return rows


def own_score_ready(out):
    value = read(out / "scores-own.json")
    if value["stage"] != "own" or value["prepared_manifest_sha256"] != sha256(
        out / "prepared-manifest.json"
    ):
        raise ValueError("own-period scoring has not been written for these inputs")
    for record in value["prediction_inputs"]:
        checked(record)
    for networks in ("fold0", "all25"):
        load_predictions(out, "own", networks)
    return value


def reference_table(manifest, *, periods):
    import pandas as pd

    # usecols deliberately excludes the period field during own-period scoring.
    columns = ["id", "asteroid_id", "quality_flag"] + (
        ["period"] if periods else ["lambda", "beta"]
    )
    key = str(REFERENCE.relative_to(FOLLOWUP))
    path = checked(manifest["sealed_inputs"][key])
    frame = pd.read_csv(path, usecols=columns)
    return frame.loc[frame.quality_flag >= 3].sort_values("id")


def reference_periods(manifest):
    frame = reference_table(manifest, periods=True)
    result = {}
    for oid, _, _, _ in OBJECTS:
        selected = frame.loc[frame.asteroid_id == int(oid.removeprefix("asteroid_"))]
        period = float(selected.iloc[0]["period"])
        if not np.isfinite(period) or period <= 0:
            raise ValueError(f"invalid current DAMIT period: {oid}")
        result[oid] = period
    return result


def predict(out, periods, networks, device):
    manifest = context(out)
    validated(out, manifest)
    if periods == "damit":
        if networks != "fold0":
            raise ValueError("DAMIT-period secondary uses fold0 only")
        own_score_ready(out)
        chosen = reference_periods(manifest)
        provenance = "current-DAMIT-quality3plus-lowest-model-id; secondary-reference-input"
        period_input = binding(out / "scores-own.json")
    else:
        found = read(out / "periods.json")
        if found["prepared_manifest_sha256"] != sha256(out / "prepared-manifest.json"):
            raise ValueError("period input mismatch")
        chosen, provenance = found["periods_hours"], found["provenance"]
        period_input = binding(out / "periods.json")
    models, bindings = load_models(networks, device)
    path = prediction_path(out, periods, networks)
    with path.open("x", buffering=1) as handle:
        for row in manifest["objects"]:
            oid = row["object_id"]
            if chosen[oid] is None:
                result = {"status": "no_period", "period_hours": None}
            else:
                try:
                    result = prediction(
                        models,
                        epochs_of(checked(row["prepared"])),
                        KnownPeriod(chosen[oid], provenance),
                        device,
                    )
                except (ValueError, RuntimeError) as error:
                    result = {
                        "status": "input_rejected",
                        "period_hours": chosen[oid],
                        "reason": str(error),
                    }
            handle.write(json.dumps({"object_id": oid, **result}, allow_nan=False) + "\n")
            print(oid, periods, networks, result["status"], flush=True)
    write(
        path.with_suffix(".manifest.json"),
        {
            "periods": periods,
            "networks": networks,
            "checkpoint_bindings": bindings,
            "device": device,
            "predictions": binding(path),
            "period_input": period_input,
            "prepared_manifest_sha256": sha256(out / "prepared-manifest.json"),
        },
    )


def metric_summary(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return {
        "n": len(rows),
        "n_scored": len(values),
        "n_failed": len(rows) - len(values),
        "median_deg": float(np.median(values)) if values else None,
        "number_within_20_deg": sum(v <= 20 + 1e-8 for v in values),
    }


def score(out, stage):
    from repro.compute_reference_baselines import standard_axes, unit_from_degrees

    from .score import axial_error, period_class

    manifest = context(out)
    validated(out, manifest)
    own = load_predictions(out, "own", "fold0")
    all25 = load_predictions(out, "own", "all25")
    if stage == "final":
        previous = own_score_ready(out)
        damit = load_predictions(out, "damit", "fold0")
        periods = reference_periods(manifest)
    else:
        previous, damit, periods = None, None, None
    frame = reference_table(manifest, periods=False)
    exposure = {row["object_id"]: row for row in read(checked(manifest["exposure"]))["objects"]}
    rows = []
    for source in manifest["objects"]:
        oid = source["object_id"]
        refs = frame.loc[frame.asteroid_id == int(oid.removeprefix("asteroid_"))]
        if refs.empty:
            raise ValueError(f"no quality-3+ reference poles: {oid}")
        poles = np.asarray(
            [unit_from_degrees(r["lambda"], r["beta"]) for r in refs.to_dict("records")]
        )
        row = {k: source[k] for k in ("object_id", "mpc_number", "name", "cohort")}
        row.update(
            own_period_hours=own[oid].get("period_hours"),
            number_reference_poles=len(poles),
            reference_model_ids=[int(v) for v in refs.id],
            exposed=exposure[oid]["exposed"],
            standard_pole_error_deg=axial_error(standard_axes(), poles),
        )
        for name, predictions in (("own", own), ("all25_own", all25), ("damit", damit)):
            if predictions is not None:
                item = predictions[oid]
                row[f"delphi_{name}_status"] = item["status"]
                row[f"delphi_{name}_error_deg"] = (
                    axial_error(item["axes"], poles) if item["status"] == "ok" else None
                )
        if periods is not None:
            row.update(
                damit_period_hours=periods[oid],
                period_class=period_class(row["own_period_hours"], periods[oid]),
            )
        rows.append(row)
    if previous:
        for before, after in zip(previous["objects"], rows, strict=True):
            if any(after[k] != value for k, value in before.items()):
                raise ValueError("own-period scores changed before final scoring")
    metrics = ["delphi_own_error_deg", "standard_pole_error_deg", "delphi_all25_own_error_deg"]
    if stage == "final":
        metrics.append("delphi_damit_error_deg")
    result = {
        "stage": stage,
        "objects": rows,
        "prepared_manifest_sha256": sha256(out / "prepared-manifest.json"),
        "prediction_inputs": [binding(prediction_path(out, "own", n)) for n in ("fold0", "all25")],
        "summary_policy": "descriptive; median over scored objects; failures listed, no imputation; count <=20 deg",
        "summaries": {
            label: {field: metric_summary(selected, field) for field in metrics}
            for label, selected in (
                ("all7", rows),
                ("new3", [r for r in rows if r["cohort"] == "new"]),
            )
        },
    }
    if stage == "final":
        result["own_score"] = binding(out / "scores-own.json")
        result["prediction_inputs"].append(binding(prediction_path(out, "damit", "fold0")))
    write(out / ("scores-own.json" if stage == "own" else "scores.json"), result)
    write_results(out / ("RESULTS-own.md" if stage == "own" else "RESULTS.md"), result)
    print(json.dumps(result["summaries"], indent=2))


def write_results(path, result):
    final = result["stage"] == "final"
    columns = [
        "Asteroid",
        "Cohort",
        "Own P (h)",
        "DeLPHI own (deg)",
        "Standard (deg)",
        "Reference poles",
        "All25 own (deg)",
    ]
    if final:
        columns += ["DAMIT P (h)", "Period class", "DeLPHI DAMIT (deg)"]

    def number(value):
        return "FAILED" if value is None else f"{value:.6f}"

    lines = [
        "Small exploratory census check. Primary: fold0 five seeds. All25 and DAMIT-period results are secondary.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in result["objects"]:
        cells = [
            f"({row['mpc_number']}) {row['name']}" + (" [EXPOSED]" if row["exposed"] else ""),
            row["cohort"],
            number(row["own_period_hours"]),
            number(row["delphi_own_error_deg"]),
            number(row["standard_pole_error_deg"]),
            str(row["number_reference_poles"]),
            number(row["delphi_all25_own_error_deg"]),
        ]
        if final:
            cells += [
                number(row["damit_period_hours"]),
                row["period_class"],
                number(row["delphi_damit_error_deg"]),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Group | Metric | Median (deg) | Within 20 deg | Scored / total |",
        "| --- | --- | --- | --- | --- |",
    ]
    for group, metrics in result["summaries"].items():
        for metric, values in metrics.items():
            lines.append(
                f"| {group} | {metric} | {number(values['median_deg'])} | {values['number_within_20_deg']} | {values['n_scored']} / {values['n']} |"
            )
    lines += [
        "",
        result["summary_policy"],
        "No bootstrap intervals or statistical tests were computed.",
    ]
    if not final:
        lines.append(
            "Current DAMIT periods were not read. Period classes and DAMIT-period predictions await final scoring."
        )
    with path.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "search", "predict", "score"):
        child = commands.add_parser(name)
        child.add_argument("--out-dir", type=Path, required=True)
        if name in ("validate", "predict"):
            child.add_argument("--device", choices=("cuda",), default="cuda")
        if name == "predict":
            child.add_argument("--periods", choices=("own", "damit"), required=True)
            child.add_argument("--networks", choices=("fold0", "all25"), required=True)
        if name == "score":
            child.add_argument("--stage", choices=("own", "final"), required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.out_dir)
    elif args.command == "validate":
        validate(args.out_dir, args.device)
    elif args.command == "search":
        search(args.out_dir)
    elif args.command == "predict":
        predict(args.out_dir, args.periods, args.networks, args.device)
    else:
        score(args.out_dir, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
