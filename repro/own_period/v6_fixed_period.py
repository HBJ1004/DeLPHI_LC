"""V6: the pre-specified v1.0.0 downstream test (six DeLPHI vs six standard starts) at our period.

    PYTHONPATH=<worktree> python repro/own_period/v6_fixed_period.py run   --periods {damit|FILE} --axes NPZ --output DIR [--ids ...] [--max-seconds S]
    PYTHONPATH=<worktree> python repro/own_period/v6_fixed_period.py validate --output DIR
    PYTHONPATH=<worktree> python repro/own_period/v6_fixed_period.py score --output DIR --result JSON

``run`` repeats ``lc_pipeline.k3.downstream.run_fixed_period_benchmark`` of the
published run (implementation commit 7874092, downstream code 214e762) start for
start: the same solver binary (DAMIT convexinv 0.2.1, unpatched, sha 7965542f),
the same parameters, 3600 s timeout, serial child processes, the same object
order (fold order of the split) and the same arm order (protocol-hash rank,
balanced alternation).  It calls the published ``_run_record`` for every start.
The only change is where the period comes from: ``damit`` takes the first frozen
catalogue solution (the published input, used only for validation); a file
takes ``periods_hours`` of that file (our rounded own period).  With a file the
run never reads the catalogue solutions (no reference period or pole); the
DeLPHI starts come from ``--axes`` (``refined_axes`` of an OOF npz).

``score`` is run only after ``run`` has written ``run-complete.json``.  It
rebuilds the published rows and verdict with the published functions
(``_valid_result``, ``evaluate_downstream_gate``, ``axial_angular_error_deg``),
adding the neural inference time stored in the published ensemble artifact to
the DeLPHI arm, as published.  It also reports the variants asked for in the
own-period protocol (failures excluded, period-search time added to both arms).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from lc_pipeline.k3.downstream import (
    DAMIT_STANDARD_STARTS_DEG,
    DownstreamStart,
    _run_record,
    _valid_result,
    evaluate_downstream_gate,
    signed_starts_from_axes,
)
from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
from lc_pipeline.physics.axial import axial_angular_error_deg

from repro.own_period import DataPath

ROOT = DataPath("../../..")
FOLLOWUP = DataPath("../..")
DATA = FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z"
CATALOG = DATA / "catalog.jsonl"
SPLITS = DATA / "publication-splits-v2.3.json"
DUMP = ROOT / "damit-20250610T000301Z"
FROZEN = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092"
PUBLISHED_ENSEMBLE = FROZEN / "evaluations/real-oof-ensemble.npz"
PUBLISHED_RUN = FROZEN / "downstream/fixed-period"
SOLVER_ROOT = FOLLOWUP / "inputs/solver/version_0.2.1/convexinv"
EXECUTABLE = SOLVER_ROOT / "convexinv"
EXECUTABLE_SHA256 = "7965542f9ac721a7935a2bfc5e5984bda6c70502164d0435b8141967ffadfb91"
SOURCE_TREE_SHA256 = "ec71a8c99d5205bcdcdd9e6d924b397200cc2029e8ed4feec9471719a53ae991"
CATALOG_SHA256 = "ac1abdab62453fca6e3d7473af0ad3b7de1bcb04cee9b947fb7079d340613679"
SPLITS_SHA256 = "1c5b15b6577805326a0c36605c174277a1d6653a60c2ac26b717373fe8d61df4"
PERIOD_SEARCH = FOLLOWUP / "experiments/period-refinement-20260921/warm170-first.jsonl"
TIMEOUT = 3600.0


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, sort_keys=True, allow_nan=True)
        stream.write("\n")


def object_order() -> list[str]:
    if digest(SPLITS) != SPLITS_SHA256:
        raise RuntimeError("split file changed")
    folds = sorted(json.loads(SPLITS.read_text())["folds"], key=lambda r: int(r["fold"]))
    return [oid for row in folds for oid in row["test_ids"]]


def arm_order(object_ids: list[str]) -> dict[str, tuple[str, str]]:
    rank = {
        oid: r
        for r, oid in enumerate(
            sorted(object_ids, key=lambda v: hashlib.sha256(f"{K3_PROTOCOL_SHA256}:{v}".encode()).hexdigest())
        )
    }
    return {
        oid: ("baseline", "candidate") if rank[oid] % 2 == 0 else ("candidate", "baseline")
        for oid in object_ids
    }


def lightcurve_paths() -> dict[str, Path]:
    """Lightcurve paths from the catalogue (only the lightcurve fields are used)."""
    if digest(CATALOG) != CATALOG_SHA256:
        raise RuntimeError("catalogue changed")
    out = {}
    for line in CATALOG.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["object_id"]] = (DUMP / row["lightcurve"]["source_path"], row["lightcurve"]["source_sha256"])
    return out


def damit_periods() -> dict[str, float]:
    """Published input: first frozen catalogue solution (validation runs only)."""
    out = {}
    for line in CATALOG.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("solutions"):
                out[row["object_id"]] = float(row["solutions"][0]["period_hours"])
    return out


def load_axes(path: Path, object_ids: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        ids = [str(v) for v in z["object_ids"]]
        axes = np.asarray(z["refined_axes"], dtype=np.float64)
    if ids != object_ids:
        raise RuntimeError("axes file is not in split order")
    return {oid: axes[i] for i, oid in enumerate(ids)}


def run(args: argparse.Namespace) -> int:
    out = args.output.resolve()  # absolute: the solver runs with cwd in each start directory
    if not out.is_absolute():
        raise RuntimeError("output must be absolute")
    if digest(EXECUTABLE) != EXECUTABLE_SHA256:
        raise RuntimeError("solver binary differs from the published one")
    order = object_order()
    if args.periods == "damit":
        periods, provenance, period_sha = damit_periods(), "first frozen catalog solution", digest(CATALOG)
    else:
        source = json.loads(Path(args.periods).read_text())
        periods, provenance, period_sha = source["periods_hours"], source["provenance"], digest(Path(args.periods))
    axes = load_axes(args.axes, order)
    lightcurves = lightcurve_paths()
    ids = order if not args.ids else [oid for oid in order if oid in set(args.ids)]
    arms = arm_order(order)
    header = {
        "schema": "delphi.v67.fixed-period-run.v1",
        "period_policy": provenance,
        "periods": {"source": str(args.periods), "sha256": period_sha},
        "axes": {"path": str(args.axes.resolve()), "sha256": digest(args.axes)},
        "executable": {"path": str(EXECUTABLE), "sha256": EXECUTABLE_SHA256},
        "timeout_seconds": TIMEOUT,
        "object_ids": ids,
    }
    header_path = out / "run-header.json"
    if header_path.exists():
        if json.loads(header_path.read_text()) != header:
            raise RuntimeError("existing run header differs; refusing to mix runs")
    else:
        write_once(header_path, header)
    started = time.perf_counter()
    for number, oid in enumerate(ids, start=1):
        if (out / "runs" / oid / "object-done.json").exists():
            continue
        if args.max_seconds and time.perf_counter() - started > args.max_seconds:
            print(f"chunk time limit reached before {oid}", flush=True)
            return 3
        lightcurve, lc_sha = lightcurves[oid]
        if digest(lightcurve) != lc_sha:
            raise RuntimeError(f"lightcurve changed for {oid}")
        period = periods.get(oid)
        if period is None:
            raise RuntimeError(f"no period for {oid}")
        period = float(period)
        starts = {
            "baseline": tuple(DownstreamStart(lon, lat, period) for lon, lat in DAMIT_STANDARD_STARTS_DEG),
            "candidate": signed_starts_from_axes(axes[oid], period_hours=period),
        }
        valid = {}
        for arm in arms[oid]:
            records = [
                _run_record(
                    executable=EXECUTABLE,
                    source_root=SOLVER_ROOT,
                    lightcurve=lightcurve,
                    output_root=out,
                    object_id=oid,
                    arm=arm,
                    start_index=i,
                    start=start,
                    timeout_seconds=TIMEOUT,
                    compiler_command=("cc", "--version"),
                )
                for i, start in enumerate(starts[arm])
            ]
            valid[arm] = sum(_valid_result(r) for r in records)
            for r in records:
                prov = r["result"]["provenance"]
                # The binary is the published one (sha 7965542f).  The hash of the
                # unpacked solver folder differs from the published folder hash
                # (ec71a8c9; build products), so only the binary is required.
                if prov["executable_sha256"] != EXECUTABLE_SHA256:
                    raise RuntimeError("solver binary differs from the published run")
        write_once(out / "runs" / oid / "object-done.json", {"object_id": oid, "period_hours": period, "arm_order": list(arms[oid]), "valid_starts": valid})
        print(number, oid, f"{period:.6f}", arms[oid][0], valid, flush=True)
        # Guard against the silent-failure trap: the first objects must give valid fits.
        if number == 3 and not any(
            json.loads((out / "runs" / o / "object-done.json").read_text())["valid_starts"][a]
            for o in ids[:3]
            for a in ("baseline", "candidate")
        ):
            raise RuntimeError("no valid fit in the first three objects; stopping")
    if all((out / "runs" / oid / "object-done.json").exists() for oid in ids):
        if not (out / "run-complete.json").exists():
            write_once(out / "run-complete.json", {"objects": len(ids), "header_sha256": digest(header_path)})
    return 0


def _record(base: Path, oid: str, arm: str, i: int) -> dict:
    return json.loads((base / "runs" / oid / arm / f"start-{i}" / "result.json").read_text())


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b)))))


def _xyz(lon: float, lat: float) -> np.ndarray:
    lo, la = math.radians(lon), math.radians(lat)
    return np.asarray([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def validate(args: argparse.Namespace) -> int:
    """Compare every start of a DAMIT-period run with the published records."""
    out = args.output.resolve()
    ids = json.loads((out / "run-header.json").read_text())["object_ids"]
    worst_angle, worst_rms, n, identical, mismatches = 0.0, 0.0, 0, 0, []
    for oid in ids:
        for arm in ("baseline", "candidate"):
            for i in range(6):
                mine, pub = _record(out, oid, arm, i), _record(PUBLISHED_RUN, oid, arm, i)
                n += 1
                if mine["identity"] != pub["identity"]:
                    mismatches.append([oid, arm, i, "identity"])
                    continue
                a, b = mine["result"], pub["result"]
                same_status = (_valid_result(mine), a["return_code"], a["timed_out"], a["iterations"]) == (
                    _valid_result(pub), b["return_code"], b["timed_out"], b["iterations"])
                if not same_status:
                    mismatches.append([oid, arm, i, "status"])
                    continue
                if a["final_lambda_deg"] is not None and b["final_lambda_deg"] is not None:
                    ang = angle_deg(_xyz(a["final_lambda_deg"], a["final_beta_deg"]), _xyz(b["final_lambda_deg"], b["final_beta_deg"]))
                    worst_angle = max(worst_angle, ang)
                    if ang > 1e-6:
                        mismatches.append([oid, arm, i, f"angle {ang:.3g}"])
                if a["relative_rms_from_output"] is not None and b["relative_rms_from_output"] is not None:
                    d = abs(a["relative_rms_from_output"] - b["relative_rms_from_output"])
                    worst_rms = max(worst_rms, d)
                    if d > 0:
                        mismatches.append([oid, arm, i, f"rms {d:.3g}"])
                keys = ("output_lightcurve_sha256", "output_parameter_sha256", "output_area_sha256", "stdout_sha256")
                identical += all(a[k] == b[k] for k in keys)
    result = {
        "objects": len(ids),
        "starts_compared": n,
        "starts_bit_identical_outputs": identical,
        "max_angle_deg_atan2": worst_angle,
        "max_abs_rms_difference": worst_rms,
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    write_once(out / "validation-vs-published.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}), len(mismatches), "mismatches")
    return 0 if result["passed"] else 1


def period_search_seconds() -> dict[str, float]:
    out = {}
    for line in PERIOD_SEARCH.read_text().splitlines():
        row = json.loads(line)
        if row["method"] == "fourier_gpu" and row["mode"] == "warm" and row["repeat"] == 0:
            out[row["object_id"]] = float(row["seconds"])
    return out


def score(args: argparse.Namespace) -> int:
    out = args.output.resolve()
    published = out == PUBLISHED_RUN.resolve()  # scorer check on the published records
    if published:
        header = {"object_ids": object_order(), "period_policy": "first frozen catalog solution (published run)"}
    else:
        if not (out / "run-complete.json").exists():
            raise RuntimeError("run is not complete; scoring refused")
        header = json.loads((out / "run-header.json").read_text())
    ids = header["object_ids"]
    orders = arm_order(object_order())
    # Reference poles are opened only now, after every fit is written.
    catalog = {json.loads(l)["object_id"]: json.loads(l) for l in CATALOG.read_text().splitlines() if l.strip()}
    with np.load(PUBLISHED_ENSEMBLE, allow_pickle=False) as z:
        published_inference = dict(zip([str(v) for v in z["object_ids"]], np.asarray(z["inference_wall_seconds"], dtype=float)))
    search = period_search_seconds()
    rows = []
    for oid in ids:
        if published:
            done = {"period_hours": _record(out, oid, "baseline", 0)["identity"]["period_hours"], "arm_order": list(orders[oid])}
        else:
            done = json.loads((out / "runs" / oid / "object-done.json").read_text())
        targets = np.asarray([s["vector"] for s in catalog[oid]["solutions"]], dtype=np.float64)
        row = {"object_id": oid, "period_hours": done["period_hours"], "arm_order": done["arm_order"],
               "neural_inference_wall_seconds": published_inference[oid],
               "period_search_seconds": search.get(oid)}
        for arm in ("baseline", "candidate"):
            records = [_record(out, oid, arm, i) for i in range(6)]
            valid = [r for r in records if _valid_result(r)]
            best = min(valid, key=lambda r: (float(r["result"]["relative_rms_from_output"]), int(r["identity"]["start_index"])), default=None)
            wall = float(sum(float(r["result"]["wall_time_seconds"]) for r in records))
            if arm == "candidate":
                wall += published_inference[oid]
            if best is None:
                rms, err, success, best_start = float("nan"), float("nan"), False, None
            else:
                res = best["result"]
                rms = float(res["relative_rms_from_output"])
                pole = _xyz(float(res["final_lambda_deg"]), float(res["final_beta_deg"]))
                err = float(np.min(axial_angular_error_deg(pole[None, :], targets)))
                success, best_start = err <= 20.0, int(best["identity"]["start_index"])
            row[arm] = {"wall_seconds": wall, "valid_starts": len(valid), "best_start_index": best_start,
                        "final_rms": rms, "pole_error_deg": err, "success": success,
                        "iterations": int(sum(int(r["result"]["iterations"] or 0) for r in records))}
        rows.append(row)

    def gate(selected: list[dict], add_search: bool = False) -> dict:
        extra = [r["period_search_seconds"] if add_search else 0.0 for r in selected]
        verdict = evaluate_downstream_gate(
            [r["baseline"]["wall_seconds"] + e for r, e in zip(selected, extra)],
            [r["candidate"]["wall_seconds"] + e for r, e in zip(selected, extra)],
            [r["baseline"]["final_rms"] for r in selected],
            [r["candidate"]["final_rms"] for r in selected],
            baseline_success=[r["baseline"]["success"] for r in selected],
            candidate_success=[r["candidate"]["success"] for r in selected],
        ).as_mapping()
        return {"objects": len(selected), **verdict}

    def arm_summary(selected: list[dict]) -> dict:
        res = {}
        for arm in ("baseline", "candidate"):
            errs = np.asarray([r[arm]["pole_error_deg"] for r in selected], dtype=float)
            fin = np.isfinite(errs)
            res[arm] = {
                "wall_seconds_sum": float(sum(r[arm]["wall_seconds"] for r in selected)),
                "wall_seconds_mean": float(np.mean([r[arm]["wall_seconds"] for r in selected])),
                "valid_starts": int(sum(r[arm]["valid_starts"] for r in selected)),
                "objects_without_valid_fit": int(np.sum(~fin)),
                "recovered_within_20": int(sum(r[arm]["success"] for r in selected)),
                "selected_error_mean_valid_deg": float(np.mean(errs[fin])) if fin.any() else None,
                "selected_error_mean_failures_at_90_deg": float(np.mean(np.where(fin, errs, 90.0))),
            }
        return res

    have_search = [r for r in rows if r["period_search_seconds"] is not None]
    both_valid = [r for r in rows if r["baseline"]["valid_starts"] and r["candidate"]["valid_starts"]]
    result = {
        "schema": "delphi.v67.fixed-period-score.v1",
        "run_header": header,
        "run_header_sha256": None if published else digest(out / "run-header.json"),
        "convention": "published: recovery = selected fit within 20 deg of any catalogue solution; an object without a valid fit counts as not recovered; DeLPHI arm wall time includes the neural inference time stored in the published ensemble artifact",
        "verdict_published_convention": gate(rows),
        "verdict_objects_with_valid_fits_in_both_arms": gate(both_valid),
        "verdict_with_period_search_time_added_to_both_arms": gate(have_search, add_search=True),
        "period_search": {"file": str(PERIOD_SEARCH), "sha256": digest(PERIOD_SEARCH), "field": "seconds", "method": "fourier_gpu", "mode": "warm",
                          "objects_with_time": len(have_search),
                          "mean_seconds": float(np.mean([r["period_search_seconds"] for r in have_search])) if have_search else None},
        "arms": arm_summary(rows),
        "rows": rows,
    }
    write_once(args.result, result)
    print(json.dumps({k: result[k] for k in ("verdict_published_convention", "verdict_objects_with_valid_fits_in_both_arms", "verdict_with_period_search_time_added_to_both_arms")}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--periods", required=True)
    r.add_argument("--axes", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--ids", nargs="*")
    r.add_argument("--max-seconds", type=float, default=0.0)
    v = sub.add_parser("validate")
    v.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    return {"run": run, "validate": validate, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
