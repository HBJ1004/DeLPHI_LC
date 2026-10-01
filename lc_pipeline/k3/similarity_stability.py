"""CPU-only, retrospective similarity and unordered-set stability checks.

No neural weights are loaded. Neighbour predictions are sealed before any
held-out reference is scored. This is not a new independent validation cohort.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .reliability_study import binding, digest, seal, unseal, verify, write_json

SCHEMA = "delphi.k3-similarity-stability.v1"
SEED = 20260916
DRAWS = 10_000
HIGHLIGHTS = (
    "point-fraction-0.5",
    "point-fraction-0.25",
    "whole_session-fraction-0.5",
    "whole_session-fraction-0.25",
    "full-noise_sigma-0.03",
    "full-noise_sigma-0.1",
)
PHOTO_NAMES = (
    "flux_std",
    "flux_q95_minus_q05",
    "harmonic_1",
    "harmonic_2",
    "harmonic_3",
    "harmonic_4",
)
GEOMETRY_NAMES = tuple(
    [f"mean_{kind}_{axis}" for kind in ("sun", "observer") for axis in "xyz"]
    + [
        f"moment_{kind}_{i}{j}"
        for kind in ("sun", "observer")
        for i, j in (("x", "x"), ("x", "y"), ("x", "z"), ("y", "y"), ("y", "z"), ("z", "z"))
    ]
    + ["session_phase_angle_median_rad", "session_phase_angle_iqr_rad"]
)
SAMPLING_NAMES = (
    "log_period_hours",
    "log_points",
    "log_sessions",
    "log1p_span_days",
    "log_median_session_points",
    "median_session_phase_occupancy",
)
FEATURE_NAMES = PHOTO_NAMES + GEOMETRY_NAMES + SAMPLING_NAMES
GROUPS = (tuple(range(6)), tuple(range(6, 26)), tuple(range(26, 32)))
PERMUTATIONS = np.asarray(list(itertools.permutations(range(3))))


def unit_axes(value, *, three=False):
    axes = np.asarray(value, dtype=float)
    if (
        axes.ndim != 2
        or axes.shape[1] != 3
        or not 1 <= len(axes) <= 3
        or (three and len(axes) != 3)
        or not np.isfinite(axes).all()
    ):
        raise ValueError("expected finite one-to-three axes, or exactly three candidates")
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero axis")
    return axes / norms


def axial_angles(a, b):
    dot = np.clip(np.abs(unit_axes(a) @ unit_axes(b).T), 0, 1)
    # Avoid tiny nonzero self-distances caused by floating-point normalization.
    dot[dot > 1 - 1e-14] = 1
    return np.degrees(np.arccos(dot))


def set_displacement(a, b):
    """Return minimum assignment mean and minimum assignment maximum in degrees."""
    cost = axial_angles(unit_axes(a, three=True), unit_axes(b, three=True))
    choices = cost[np.arange(3)[None, :], PERMUTATIONS]
    return {
        "assignment_mean_deg": float(choices.mean(axis=1).min()),
        "bottleneck_deg": float(choices.max(axis=1).min()),
    }


def features_from_sessions(sessions, period_hours):
    """Label-free features from native N-by-8 arrays (time, flux, sun, observer)."""
    if not np.isfinite(period_hours) or period_hours <= 0:
        raise ValueError("period must be finite and positive")
    photo, geometry, phase_angles, occupancies, times, counts = [], [], [], [], [], []
    for raw in sessions:
        x = np.asarray(raw, dtype=float)
        if x.ndim != 2 or x.shape[1] != 8 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("features require valid native sessions")
        if np.any(x[:, 1] <= 0):
            raise ValueError("non-positive flux")
        flux = x[:, 1] / x[:, 1].mean()
        phase = np.remainder((x[:, 0] - x[:, 0].min()) * 24 / period_hours, 1)
        harmonic = [
            float(2 * abs(np.mean((flux - 1) * np.exp(2j * np.pi * h * phase))))
            for h in range(1, 5)
        ]
        photo.append(
            [float(flux.std()), float(np.diff(np.quantile(flux, [0.05, 0.95]))[0]), *harmonic]
        )
        vectors = []
        for start in (2, 5):
            vector = x[:, start : start + 3]
            norm = np.linalg.norm(vector, axis=1, keepdims=True)
            if np.any(norm <= 1e-12):
                raise ValueError("zero geometry vector")
            vectors.append(vector / norm)
        means = np.concatenate([v.mean(axis=0) for v in vectors])
        moments = np.concatenate([(v.T @ v / len(v))[np.triu_indices(3)] for v in vectors])
        geometry.append(np.concatenate([means, moments]))
        phase_angles.append(
            float(np.arccos(np.clip(np.sum(vectors[0] * vectors[1], axis=1), -1, 1)).mean())
        )
        occupancies.append(len(np.unique(np.floor(64 * phase).astype(int))) / 64)
        times.extend([float(x[:, 0].min()), float(x[:, 0].max())])
        counts.append(len(x))
    if not counts:
        raise ValueError("no valid sessions")
    result = np.concatenate(
        [
            np.median(photo, axis=0),
            np.mean(geometry, axis=0),
            [np.median(phase_angles), np.diff(np.quantile(phase_angles, [0.25, 0.75]))[0]],
            [
                np.log(period_hours),
                np.log(sum(counts)),
                np.log(len(counts)),
                np.log1p(max(times) - min(times)),
                np.log(np.median(counts)),
                np.median(occupancies),
            ],
        ]
    )
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("invalid feature vector")
    return result


def nearest_training(training, queries):
    """Fit scaling on training alone. No labels, evaluation errors, or tuning enter."""
    ids, qids = sorted(training), sorted(queries)
    if len(ids) < 3 or not qids or set(ids) & set(qids):
        raise ValueError("need three training neighbours and disjoint nonempty queries")
    x, q = np.array([training[i] for i in ids]), np.array([queries[i] for i in qids])
    if (
        x.shape != (len(ids), 32)
        or q.shape != (len(qids), 32)
        or not np.isfinite(x).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("invalid feature matrix")
    mean, std = x.mean(axis=0), x.std(axis=0)
    active = std > 1e-12
    scale = np.where(active, std, 1)
    delta = (q[:, None, :] - x[None, :, :]) / scale
    distance = np.zeros((len(q), len(x)))
    for group in GROUPS:
        indices = [i for i in group if active[i]]
        if indices:
            distance += np.square(delta[:, :, indices]).mean(axis=2) / len(GROUPS)
    result = []
    for oid, values in zip(qids, distance, strict=True):
        order = np.argsort(values, kind="stable")[:3]
        result.append(
            {
                "object_id": oid,
                "neighbour_ids": [ids[i] for i in order],
                "squared_distances": [float(values[i]) for i in order],
            }
        )
    return result, {
        "train_ids": ids,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "active": active.tolist(),
    }


def neighbour_axes(neighbour_ids, training_references):
    """Only the training-reference dictionary is accepted by the prediction stage."""
    if len(neighbour_ids) != 3 or len(set(neighbour_ids)) != 3:
        raise ValueError("expected three distinct neighbours")
    refs = [unit_axes(training_references[i]) for i in neighbour_ids]
    primary = np.array([r[0] for r in refs])
    nearest = refs[0]
    sensitivity = np.concatenate([nearest, np.repeat(nearest[:1], 3 - len(nearest), axis=0)])
    return {
        "three_neighbour_primary_axes": primary.tolist(),
        "one_neighbour_all_axes": sensitivity.tolist(),
    }


def validate_roles(manifests, objects):
    expected = {o["object_id"]: o["fold"] for o in objects}
    if len(expected) != len(objects):
        raise ValueError("duplicate study object")
    seen, folds = set(), set()
    for manifest in manifests:
        fold = manifest["fold"]
        if fold in folds:
            raise ValueError("duplicate fold")
        folds.add(fold)
        roles = manifest["object_roles"]
        values = [
            oid
            for role in ("train_ids", "validation_ids", "calibration_ids", "test_ids")
            for oid in roles[role]
        ]
        if len(values) != len(set(values)) or set(values) != set(expected):
            raise ValueError("roles overlap, duplicate, or fail to partition study objects")
        for oid in roles["test_ids"]:
            if expected[oid] != fold or oid in seen:
                raise ValueError("test object/fold mismatch")
            seen.add(oid)
    if seen != set(expected):
        raise ValueError("test coverage incomplete")


def validate_saved_rows(study, schedule, rows):
    objects = {o["object_id"]: o for o in study["objects"]}
    configs = {c["condition"]: c for c in study["conditions"]}
    configs["grid-full"] = {"family": "two_d", "repeatable": False}
    expected = {
        (oid, name, repeat)
        for oid in objects
        for name, cfg in configs.items()
        for repeat in range(schedule["repeats"] if cfg["repeatable"] else 1)
    }
    actual = {(r["object_id"], r["condition"], r["repeat"]): r for r in rows}
    if len(actual) != len(rows) or set(actual) != expected:
        raise ValueError("saved prediction inventory missing, extra, or duplicated")
    for (oid, name, _), row in actual.items():
        entry = objects[oid]
        eligible = configs[name]["family"] != "two_d" or oid in study["grid_ids"]
        if row["fold"] != entry["fold"]:
            raise ValueError("prediction fold mismatch")
        if not eligible:
            if (
                row["status"] != "ineligible"
                or row.get("axes") is not None
                or row.get("error_deg") is not None
            ):
                raise ValueError("incorrect ineligibility")
            continue
        if row["status"] not in ("ok", "failed"):
            raise ValueError("eligible prediction has invalid status")
        if row.get("source_sha256") != entry["lightcurve"]["sha256"]:
            raise ValueError("original lightcurve hash mismatch")
        if not np.isfinite(row.get("error_deg", np.nan)) or not 0 <= row["error_deg"] <= 90:
            raise ValueError("invalid oracle error")
        if row["status"] == "ok":
            unit_axes(row["axes"], three=True)
        elif row.get("axes") is not None or row["error_deg"] != 90:
            raise ValueError("failed eligible prediction must have no axes and a 90-degree error")
    if any(actual[(oid, "full", 0)]["status"] != "ok" for oid in objects):
        raise ValueError("full-input baseline requires all original successful predictions")
    return actual


def distribution(rows, key, *, draws=DRAWS):
    """Conditional fold-stratified object bootstrap, not a retraining interval."""
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    valid = sorted((r for r in rows if r.get(key) is not None), key=lambda r: r["object_id"])
    if not valid:
        return {"n": 0, "mean": None, "mean_ci95": None, "median": None, "p90": None, "max": None}
    if len({r["object_id"] for r in valid}) != len(valid):
        raise ValueError("bootstrap unit must be unique asteroid")
    values = np.array([r[key] for r in valid], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite bootstrap value")
    rng = np.random.default_rng(SEED)
    sums = np.zeros(draws)
    for fold in sorted({r["fold"] for r in valid}):
        v = np.array([r[key] for r in valid if r["fold"] == fold])
        sums += v[rng.integers(len(v), size=(draws, len(v)))].sum(axis=1)
    ci = np.quantile(sums / len(values), [0.025, 0.975])
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "mean_ci95": ci.tolist(),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def stability_tables(rows, *, draws=DRAWS):
    baseline = {r["object_id"]: r for r in rows if r["condition"] == "full"}
    groups = defaultdict(list)
    for row in rows:
        if row["condition"] not in ("full", "grid-full"):
            groups[(row["condition"], row["object_id"])].append(row)
    object_rows, repeat_rows = [], []
    for (condition, oid), values in sorted(groups.items()):
        values = sorted(values, key=lambda r: r["repeat"])
        eligible = [r for r in values if r["status"] != "ineligible"]
        if not eligible:
            continue
        good = [r for r in eligible if r["status"] == "ok"]
        base = baseline[oid]
        distances = []
        for row in eligible:
            distance = (
                set_displacement(base["axes"], row["axes"])
                if row["status"] == "ok"
                else {"assignment_mean_deg": None, "bottleneck_deg": None}
            )
            repeat_rows.append(
                {
                    "object_id": oid,
                    "fold": row["fold"],
                    "condition": condition,
                    "repeat": row["repeat"],
                    "status": row["status"],
                    **distance,
                    "oracle_change_deg": row["error_deg"] - base["error_deg"],
                }
            )
            if row["status"] == "ok":
                distances.append(distance)
        repeat_pairs = [
            set_displacement(a["axes"], b["axes"])["assignment_mean_deg"]
            for a, b in itertools.combinations(good, 2)
        ]
        object_rows.append(
            {
                "object_id": oid,
                "fold": base["fold"],
                "condition": condition,
                "scheduled_repeats": len(eligible),
                "successful_repeats": len(good),
                "failed_repeats": len(eligible) - len(good),
                "assignment_mean_deg": float(np.mean([d["assignment_mean_deg"] for d in distances]))
                if distances
                else None,
                "bottleneck_deg": float(np.mean([d["bottleneck_deg"] for d in distances]))
                if distances
                else None,
                "repeat_pair_mean_deg": float(np.mean(repeat_pairs)) if repeat_pairs else None,
                "repeat_pairs": len(repeat_pairs),
                "oracle_change_deg": float(
                    np.mean([r["error_deg"] for r in eligible]) - base["error_deg"]
                ),
            }
        )
    summaries = []
    for condition in sorted({r["condition"] for r in object_rows}):
        selected = [r for r in object_rows if r["condition"] == condition]
        scheduled = [r for r in rows if r["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "highlight": condition in HIGHLIGHTS,
                "scheduled_rows": len(scheduled),
                "eligible_objects": len(selected),
                "successful_rows": sum(r["status"] == "ok" for r in scheduled),
                "failed_rows": sum(r["status"] == "failed" for r in scheduled),
                "ineligible_rows": sum(r["status"] == "ineligible" for r in scheduled),
                **{
                    key: distribution(selected, key, draws=draws)
                    for key in (
                        "assignment_mean_deg",
                        "bottleneck_deg",
                        "repeat_pair_mean_deg",
                        "oracle_change_deg",
                    )
                },
                "largest_displacements": sorted(
                    (r for r in selected if r["assignment_mean_deg"] is not None),
                    key=lambda r: (-r["assignment_mean_deg"], r["object_id"]),
                )[:10],
            }
        )
    return object_rows, repeat_rows, summaries


def prepare(*, original_study: Path, output: Path):
    original_study, output = Path(original_study).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("analysis output must be a new directory")
    study = unseal(original_study / "study.json")
    base = Path(__file__).resolve().parents[2]
    inputs = {
        name: binding(original_study / name)
        for name in ("study.json", "schedule.json", "scored-rows.json", "inference-complete.json")
    }
    inputs["catalog"] = binding(verify(study["catalog"]))
    inputs["protocol"] = binding(base / "repro/protocols/similarity-stability-design-20260916.md")
    sources = [
        Path(__file__),
        base / "repro/run_k3_similarity_stability.py",
        base / "lc_pipeline/k3/reliability_study.py",
        base / "lc_pipeline/k3/reliability_sampling.py",
        base / "lc_pipeline/v2/preprocessing.py",
    ]
    inputs.update({f"source:{p.relative_to(base)}": binding(p) for p in sources})
    manifests = []
    for item in study["bundles"]:
        bound = next(b for b in item["files"] if Path(b["path"]).name == "bundle.json")
        p = verify(bound)
        manifest = json.loads(p.read_text())
        inputs[f"bundle:{manifest['fold']}"] = binding(p)
        manifests.append(manifest)
    validate_roles(manifests, study["objects"])
    if (
        len(study["objects"]) != 170
        or len(manifests) != 5
        or any(
            [
                len(m["object_roles"][r])
                for r in ("train_ids", "validation_ids", "calibration_ids", "test_ids")
            ]
            != [108, 14, 14, 34]
            for m in manifests
        )
    ):
        raise ValueError("unexpected publication cohort/split dimensions")
    split_binding = next(
        b for b in study["bindings"] if Path(b["path"]).name == "publication-splits-v2.3.json"
    )
    splits = json.loads(verify(split_binding).read_text())
    inputs["splits"] = binding(Path(split_binding["path"]))
    for manifest in manifests:
        source_fold = next(f for f in splits["folds"] if f["fold"] == manifest["fold"])
        if manifest["splits_sha256"] != split_binding["sha256"] or any(
            set(manifest["object_roles"][role]) != set(source_fold[role])
            for role in manifest["object_roles"]
        ):
            raise ValueError("bundle roles disagree with original split")
    for obj in study["objects"]:
        inputs[f"lightcurve:{obj['object_id']}"] = binding(verify(obj["lightcurve"]))
    lock = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "retrospective_diagnostic_not_new_independent_validation",
        "inputs": inputs,
        "feature_names": FEATURE_NAMES,
        "feature_groups": GROUPS,
        "bootstrap_draws": DRAWS,
        "bootstrap_seed": SEED,
        "gpu_used": False,
        "original_study": str(original_study),
    }
    seal(output / "analysis-lock.json", lock)
    return {
        "status": "prepared",
        "output": str(output),
        "lock_sha256": digest(output / "analysis-lock.json"),
        "bound_inputs": len(inputs),
    }


def _references(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = {}
    for row in rows:
        if row.get("eligible") is not True:
            continue
        oid = row["object_id"]
        if oid in result:
            raise ValueError("duplicate catalog identity")
        result[oid] = unit_axes([s["vector"] for s in row["solutions"]]).tolist()
    return result


def predict_baselines(study, manifests, feature_rows, references):
    """Fold predictor; only train labels are passed into neighbour_axes."""
    predictions, scalers = [], []
    for manifest in sorted(manifests, key=lambda m: m["fold"]):
        roles = manifest["object_roles"]
        train = {i: feature_rows[i] for i in roles["train_ids"]}
        query = {i: feature_rows[i] for i in roles["test_ids"]}
        neighbours, scaler = nearest_training(train, query)
        train_refs = {i: references[i] for i in train}
        scalers.append({"fold": manifest["fold"], **scaler})
        for n in neighbours:
            predictions.append(
                {**n, "fold": manifest["fold"], **neighbour_axes(n["neighbour_ids"], train_refs)}
            )
    return {
        "schema": SCHEMA,
        "objects": predictions,
        "scalers": scalers,
        "feature_names": list(FEATURE_NAMES),
        "features": [
            {"object_id": i, "values": np.asarray(feature_rows[i]).tolist()}
            for i in sorted(feature_rows)
        ],
        "held_out_labels_used_to_predict": False,
    }


def _csv(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _text_once(path, buffer.getvalue())


def _text_once(path, text):
    # New reports never replace prior evidence.
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def render_report(summary):
    lines = [
        "# Similarity baseline and unordered candidate-set stability",
        "",
        "Retrospective analysis of the existing DAMIT folds. No retraining, GPU inference, "
        "or inversion was performed. The feature rule and analysis specification were "
        "fixed before these new scores were computed; earlier cohort results were already known.",
        "",
        "## Similar-asteroid comparisons",
        "",
        "All methods use the same 170 held-out objects and antipodal oracle@3 reference endpoint. "
        "Neighbour selection and feature scaling use only each fold's 108 training objects.",
        "",
        "| Method | Mean (deg) | Median (deg) | Within 20 deg | Mean difference from K3 (95% interval) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary["similarity"]:
        d = r["difference_from_k3_deg"]
        lines.append(
            f"| {r['method']} | {r['error_deg']['mean']:.2f} | "
            f"{r['error_deg']['median']:.2f} | {100 * r['within20_fraction']:.1f}% | "
            f"{d['mean']:.2f} [{d['mean_ci95'][0]:.2f}, {d['mean_ci95'][1]:.2f}] |"
        )
    lines += [
        "",
        "Positive differences mean greater reference disagreement than K3. "
        "These fixed lookup baselines are not an exhaustive search over retrieval methods. "
        "The primary baseline uses one first-listed reference axis from each of three "
        "neighbours; the sensitivity keeps all catalog axes of the closest neighbour. "
        "Neither baseline uses previous-version model predictions.",
        "",
        "## Candidate-set stability",
        "",
        "Each set is matched one-to-one to the same object's full-input set, allowing "
        "axis sign changes and reordering. Repeats are averaged within asteroid first. "
        "Movement describes the entire proposal set, not just its closest reference axis.",
        "",
        "| Perturbation | Objects | Mean movement (deg; 95% interval) | Median | 90th percentile | Mean oracle change (deg) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_condition = {r["condition"]: r for r in summary["stability"]}
    for c in HIGHLIGHTS:
        r = by_condition[c]
        d = r["assignment_mean_deg"]
        if not d["n"]:
            lines.append(f"| {c} | 0 | unavailable | — | — | — |")
            continue
        lines.append(
            f"| {c} | {d['n']} | {d['mean']:.2f} "
            f"[{d['mean_ci95'][0]:.2f}, {d['mean_ci95'][1]:.2f}] | "
            f"{d['median']:.2f} | {d['p90']:.2f} | {r['oracle_change_deg']['mean']:+.2f} |"
        )
    lines += [
        "",
        "The complete JSON includes every condition, bottleneck distances, "
        "repeat-to-repeat movement, failures/ineligibility, and the ten largest movements "
        "per condition. CSVs retain every eligible repeat and object. Movement summaries "
        "condition on successful predictions; oracle summaries retain eligible failures "
        "at 90 degrees. Stability is not correctness or a calibrated confidence measure.",
        "",
        "## Interpretation and limits",
        "",
        "All intervals use 10,000 object resamples within the existing folds, seed 20260916. "
        "They condition on fixed fitted models and this feature rule, and do not account "
        "for training or model-selection uncertainty. These analyses can support or limit "
        "specific claims about lookup behaviour and repeatability, but cannot establish "
        "the absence of every form of overfitting, cross-survey reliability, or "
        "accuracy-preserving acceleration. Existing external-transfer and timing/fit "
        "limitations remain unchanged.",
        "",
    ]
    return "\n".join(lines)


def run(*, output: Path):
    import platform

    from .reliability_sampling import read_lc

    out = Path(output)
    lock = unseal(out / "analysis-lock.json")
    paths = {name: verify(value) for name, value in lock["inputs"].items()}
    if (out / "manifest.json").exists():
        receipt = unseal(out / "manifest.json")
        for name, checksum in receipt["files"].items():
            if digest(out / name) != checksum:
                raise ValueError("completed analysis output changed")
        return {"status": "already_complete", "output": str(out)}
    study = unseal(paths["study.json"])
    schedule = unseal(paths["schedule.json"])
    rows = unseal(paths["scored-rows.json"])
    validate_saved_rows(study, schedule, rows)
    manifests = [json.loads(p.read_text()) for key, p in paths.items() if key.startswith("bundle:")]
    validate_roles(manifests, study["objects"])
    refs = _references(paths["catalog"])
    if set(refs) != {r["object_id"] for r in study["objects"]}:
        raise ValueError("catalog/study identity mismatch")
    prediction_file = out / "neighbour-predictions.json"
    if prediction_file.exists():
        prediction = unseal(prediction_file)
        if prediction.get("analysis_lock_sha256") != digest(out / "analysis-lock.json"):
            raise ValueError("prediction belongs to a different analysis lock")
    else:
        features = {}
        for obj in sorted(study["objects"], key=lambda o: o["object_id"]):
            curve = read_lc(
                paths[f"lightcurve:{obj['object_id']}"], period_hours=obj["period_hours"]
            )
            if (
                curve.total_observations != obj["native_observations"]
                or curve.native_sessions != obj["native_sessions"]
            ):
                raise ValueError("raw lightcurve feature count differs from original study")
            features[obj["object_id"]] = features_from_sessions(
                [s.rows for s in curve.valid_sessions], obj["period_hours"]
            )
        prediction = predict_baselines(study, manifests, features, refs)
        prediction["analysis_lock_sha256"] = digest(out / "analysis-lock.json")
        seal(prediction_file, prediction)
    # Held-out scoring starts only after the full baseline-prediction file is sealed.
    max_oracle_delta = 0.0
    for row in rows:
        if row["status"] == "ok":
            recomputed = float(axial_angles(row["axes"], refs[row["object_id"]]).min())
            max_oracle_delta = max(max_oracle_delta, abs(recomputed - row["error_deg"]))
    if max_oracle_delta > 1e-5:
        raise ValueError("independent oracle recomputation disagrees with saved rows")
    full = {r["object_id"]: r for r in rows if r["condition"] == "full"}
    errors, summaries = [], []
    names = (
        "K3 ensemble",
        "Three-neighbour first-axis",
        "One-neighbour all-axes",
        "Train-only atlas",
    )
    for r in prediction["objects"]:
        oid = r["object_id"]
        k3 = float(axial_angles(full[oid]["axes"], refs[oid]).min())
        if abs(k3 - full[oid]["error_deg"]) > 1e-5:
            raise ValueError("independent oracle recomputation disagrees with saved result")
        atlas = study["atlases"][r["fold"]]["axes"]
        for name, axes in zip(
            names,
            [
                full[oid]["axes"],
                r["three_neighbour_primary_axes"],
                r["one_neighbour_all_axes"],
                atlas,
            ],
            strict=True,
        ):
            error = float(axial_angles(axes, refs[oid]).min())
            errors.append(
                {
                    "object_id": oid,
                    "fold": r["fold"],
                    "method": name,
                    "error_deg": error,
                    "difference_from_k3_deg": error - k3,
                    "neighbour_1": r["neighbour_ids"][0],
                    "neighbour_2": r["neighbour_ids"][1],
                    "neighbour_3": r["neighbour_ids"][2],
                }
            )
    if len(errors) != 170 * len(names):
        raise ValueError("baseline output inventory is incomplete")
    for name in names:
        selected = [r for r in errors if r["method"] == name]
        summaries.append(
            {
                "method": name,
                "error_deg": distribution(selected, "error_deg"),
                "difference_from_k3_deg": distribution(selected, "difference_from_k3_deg"),
                "within20_fraction": float(np.mean([r["error_deg"] <= 20 for r in selected])),
            }
        )
    objects, repeats, stability = stability_tables(rows)
    summary = {
        "schema": SCHEMA,
        "analysis_lock_sha256": digest(out / "analysis-lock.json"),
        "predictions_sha256": digest(prediction_file),
        "similarity": summaries,
        "stability": stability,
        "saved_rows_validated": len(rows),
        "max_saved_oracle_recomputation_difference_deg": max_oracle_delta,
        "bootstrap": {
            "draws": DRAWS,
            "seed": SEED,
            "unit": "asteroid_within_fold",
            "conditional_on_fixed_models_and_rule": True,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "gpu_used": False,
        "new_model_inference": False,
        "new_inversion": False,
    }
    # Stage reports together in a new subdirectory; interrupted exports can be
    # inspected and recovered without overwriting a previous scientific report.
    report_dir = out / "report"
    report_dir.mkdir(exist_ok=False)
    write_json(report_dir / "summary.json", summary)
    _csv(report_dir / "similarity-objects.csv", errors)
    _csv(report_dir / "stability-objects.csv", objects)
    _csv(report_dir / "stability-repeats.csv", repeats)
    _text_once(report_dir / "report.md", render_report(summary))
    files = [prediction_file, out / "analysis-lock.json", *sorted(report_dir.iterdir())]
    seal(
        out / "manifest.json",
        {"schema": SCHEMA, "files": {str(p.relative_to(out)): digest(p) for p in files}},
    )
    return {
        "status": "complete",
        "output": str(out),
        "objects": 170,
        "stability_conditions": len(stability),
        "saved_rows_validated": len(rows),
        "gpu_used": False,
    }
