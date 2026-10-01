"""Compare six DeLPHI starts with a fixed classical pole search.

The development command uses the 30 previously designated development objects.
The evaluation command starts fresh processes on the other 140 objects and
cannot read DAMIT reference axes. Scoring is a separate final command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

from lc_pipeline import grid_benchmark as grid
from lc_pipeline import workflow_benchmark as prior

SCHEMA = "delphi.direct-pole-workflow.v1"
LADDER = (6, 12, 18, 24, 36, 48, 72, 96, 120, 146)
REPEATS = 3
WORKERS = 6
BOOTSTRAPS = 10_000
SEED = 20260923


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def classical_starts() -> np.ndarray:
    """Start with the standard six poles; add widely spaced 20-degree-grid poles."""
    standard = grid.starts_for_arm("standard6")
    candidates = grid.pole_grid(20)
    order = []
    chosen = [row for row in standard]
    remaining = list(range(len(candidates)))
    while remaining:
        distances = np.arccos(
            np.clip((candidates[remaining] @ np.asarray(chosen).T).max(axis=1), -1, 1)
        )
        winner = remaining[int(np.argmax(distances))]
        # Four grid poles coincide with standard starts. Remove all duplicates
        # without changing the deterministic farthest-first ordering.
        if not any(np.linalg.norm(candidates[winner] - row) < 1e-10 for row in chosen):
            order.append(winner)
            chosen.append(candidates[winner])
        remaining.remove(winner)
    if len(chosen) < max(LADDER):
        raise ValueError("classical grid does not contain enough unique starts")
    return np.asarray(chosen[: max(LADDER)])


def development(lock_path: Path, scoring_lock_path: Path, archive: Path, output: Path) -> None:
    lock = read(lock_path)
    scoring = read(scoring_lock_path)
    references = {}
    with Path(scoring["reference_catalog"]["path"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("eligible"):
                references[row["object_id"]] = [item["vector"] for item in row["solutions"]]
    grid20 = grid.pole_grid(20)
    starts = classical_starts()
    grid_indexes = []
    for start in starts[6:]:
        matching = np.where(np.linalg.norm(grid20 - start, axis=1) < 1e-10)[0]
        if len(matching) != 1:
            raise ValueError("classical start is missing from archive grid")
        grid_indexes.append(int(matching[0]))
    rows = []
    for number, object_id in enumerate(lock["development_object_ids"], start=1):
        classical_case = prior._archived_case(archive, object_id, 0, "classical20")
        standard_case = prior._archived_case(archive, object_id, 0, "standard6")
        delphi_case = prior._archived_case(archive, object_id, 0, "guided20")
        classical_records = prior._load_records(classical_case)
        standard_records = prior._load_records(standard_case)
        delphi_records = prior._exact_delphi_records(delphi_case)
        if len(classical_records) != 146 or len(standard_records) != 6:
            raise ValueError(f"incomplete development record for {object_id}")
        d_fits = prior._successful_fits(delphi_records, "initial")
        d_fit = min(d_fits, key=prior._fit_key) if d_fits else None
        if d_fit is None:
            raise ValueError(f"no guided development fit for {object_id}")
        candidates = standard_records + [classical_records[index] for index in grid_indexes]
        values = {}
        for count in LADDER:
            fits = prior._successful_fits(candidates[:count], "initial")
            fit = min(fits, key=prior._fit_key) if fits else None
            values[str(count)] = fit
        rows.append(
            {
                "object_id": object_id,
                "delphi_fit": d_fit,
                "classical_fits": values,
                "reference_axes": references[object_id],
            }
        )
        print(f"development {number}/30 {object_id}", flush=True)

    report = []
    for count in LADDER:
        classical_errors = []
        delphi_errors = []
        ratios = []
        complete = True
        for row in rows:
            c = row["classical_fits"][str(count)]
            d = row["delphi_fit"]
            if c is None:
                complete = False
                continue
            classical_errors.append(prior._reference_error(c["axis"], row["reference_axes"]))
            delphi_errors.append(prior._reference_error(d["axis"], row["reference_axes"]))
            ratios.append(c["final_relative_rms"] / d["final_relative_rms"])
        if not complete:
            entry = {"classical_starts": count, "qualifies": False, "reason": "incomplete"}
        else:
            error_difference = float(np.mean(classical_errors) - np.mean(delphi_errors))
            within_difference = (sum(x <= 20 for x in classical_errors) - sum(x <= 20 for x in delphi_errors)) / 30
            rms_ratio = prior._geometric_mean(ratios)
            entry = {
                "classical_starts": count,
                "classical_mean_error_degrees": float(np.mean(classical_errors)),
                "delphi_mean_error_degrees": float(np.mean(delphi_errors)),
                "classical_within20": sum(x <= 20 for x in classical_errors),
                "delphi_within20": sum(x <= 20 for x in delphi_errors),
                "classical_minus_delphi_mean_error_degrees": error_difference,
                "classical_minus_delphi_within20": within_difference,
                "classical_over_delphi_rms_ratio": rms_ratio,
                "qualifies": error_difference <= 3 and within_difference >= -0.05 and rms_ratio <= 1.01,
            }
        report.append(entry)
        print(entry, flush=True)
    qualifying = [item for item in report if item["qualifies"]]
    result = {
        "schema": SCHEMA,
        "phase": "development_selection",
        "selection": "smallest_classical_budget_meeting_point_quality_margins_on_30_development_objects",
        "margins": {"mean_error_degrees": 3, "within20_fraction": -0.05, "rms_ratio": 1.01},
        "ladder": list(LADDER),
        "classical_starts": starts.tolist(),
        "candidate_results": report,
        "selected_classical_starts": qualifying[0]["classical_starts"] if qualifying else None,
        "development_object_ids": lock["development_object_ids"],
        "source_lock": {"path": str(lock_path.resolve()), "sha256": digest(lock_path)},
        "source_scoring_lock": {"path": str(scoring_lock_path.resolve()), "sha256": digest(scoring_lock_path)},
        "source_archive": str(archive.resolve()),
    }
    write_once(output, result)


def freeze(source_lock_path: Path, development_path: Path, output: Path, scoring_output: Path) -> None:
    source = read(source_lock_path)
    development_report = read(development_path)
    count = development_report["selected_classical_starts"]
    if count not in LADDER or development_report["source_lock"]["sha256"] != digest(source_lock_path):
        raise ValueError("invalid development selection")
    source_scoring_path = Path(development_report["source_scoring_lock"]["path"])
    if digest(source_scoring_path) != development_report["source_scoring_lock"]["sha256"]:
        raise ValueError("source scoring lock changed")
    from lc_pipeline.grid_benchmark import _environment

    evaluation = {
        "schema": SCHEMA,
        "phase": "frozen_reference_blind_evaluation",
        "source_lock": {"path": str(source_lock_path.resolve()), "sha256": digest(source_lock_path)},
        "development_report": {"path": str(development_path.resolve()), "sha256": digest(development_path)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).resolve())},
        "objects": source["objects"],
        "classical_starts": development_report["classical_starts"][:count],
        "selected_classical_starts": count,
        "bundles": source["bundles"],
        "solver": source["solver"],
        "solver_settings": source["solver_settings"],
        "device": source["device"],
        "environment": _environment(source["device"]),
        "code_root": source["code_root"],
        "runtime_sources": source["runtime_sources"],
        "settings": {"repeats": REPEATS, "workers": WORKERS, "bootstrap_resamples": BOOTSTRAPS, "seed": SEED},
    }
    write_once(output, evaluation)
    score_lock = {
        "schema": SCHEMA,
        "phase": "reference_binding_for_later_scoring",
        "evaluation_lock": {"path": str(output.resolve()), "sha256": digest(output)},
        "reference_catalog": read(source_scoring_path)["reference_catalog"],
    }
    write_once(scoring_output, score_lock)


def validate_lock(path: Path, *, environment: bool = True) -> dict:
    lock = read(path)
    if lock.get("schema") != SCHEMA or lock.get("phase") != "frozen_reference_blind_evaluation":
        raise ValueError("invalid evaluation lock")
    if digest(Path(lock["script"]["path"])) != lock["script"]["sha256"]:
        raise ValueError("benchmark script changed after freeze")
    if digest(Path(lock["source_lock"]["path"])) != lock["source_lock"]["sha256"]:
        raise ValueError("source lock changed")
    if digest(Path(lock["development_report"]["path"])) != lock["development_report"]["sha256"]:
        raise ValueError("development selection changed")
    if len(lock["objects"]) != 140 or len(lock["classical_starts"]) != lock["selected_classical_starts"]:
        raise ValueError("evaluation cohort or start count changed")
    if "reference_catalog" in lock:
        raise ValueError("reference leaked into evaluation lock")
    from lc_pipeline.grid_benchmark import _environment

    if environment and lock["environment"] != _environment(lock["device"]):
        raise ValueError("benchmark hardware or environment changed")
    if prior._runtime_sources(Path(lock["code_root"])) != lock["runtime_sources"]:
        raise ValueError("runtime source changed")
    if digest(Path(lock["solver"]["executable"]["path"])) != lock["solver"]["executable"]["sha256"]:
        raise ValueError("solver changed")
    for row in lock["objects"]:
        if digest(Path(row["lightcurve"]["path"])) != row["lightcurve"]["sha256"]:
            raise ValueError(f"photometry changed for {row['object_id']}")
    return lock


def worker(lock_path: Path, job_path: Path) -> None:
    from lc_pipeline.v2.convexinv import _parse_lightcurve_brightness
    from lc_pipeline.v2.preprocessing import KnownPeriod

    lock = read(lock_path)
    job = read(job_path)
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    observed = _parse_lightcurve_brightness(Path(job["lightcurve"]["path"]))
    inference_seconds = 0.0
    if job["arm"] == "delphi":
        from lc_pipeline.k3.damit import _read_lc

        neural_start = time.perf_counter()
        predictor = grid._predictor(lock, job["fold"])
        prediction = predictor.predict(
            _read_lc(Path(job["lightcurve"]["path"])),
            known_period=KnownPeriod(hours=job["period_hours"], provenance="fixed_published_period"),
            object_id=job["object_id"],
        )
        if predictor.device.type == "cuda":
            import torch

            torch.cuda.synchronize(predictor.device)
        inference_seconds = time.perf_counter() - neural_start
        write_once(directory / "prediction.json", prediction)
        starts = prior.delphi_starts([item["axis_xyz"] for item in prediction["axes"]])
    else:
        starts = np.asarray(lock["classical_starts"], dtype=np.float64)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    records, solver_wall = prior._parallel_starts(job, starts, observed, lock, directory / "starts")
    fits = prior._successful_fits(records, "direct")
    selected = min(fits, key=prior._fit_key) if fits else None
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "schema": SCHEMA,
        "phase": "reference_blind_case",
        "object_id": job["object_id"],
        "fold": job["fold"],
        "repeat_index": job["repeat_index"],
        "arm": job["arm"],
        "completed": selected is not None,
        "selected_fit": selected,
        "starts_requested": len(starts),
        "valid_starts": len(fits),
        "inference_and_load_seconds": inference_seconds,
        "solver_wall_seconds": solver_wall,
        "solver_cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "worker_wall_seconds": time.perf_counter() - started,
    }
    write_once(directory / "selected-result.json", result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


def execute(lock_path: Path, output: Path) -> None:
    lock = validate_lock(lock_path)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        {**row, "repeat_index": repeat, "arm": arm}
        for row in lock["objects"]
        for repeat in range(REPEATS)
        for arm in ("classical", "delphi")
    ]
    random.Random(SEED).shuffle(jobs)
    start = output / "execution-start.json"
    if not start.exists():
        write_once(start, {"schema": SCHEMA, "phase": "reference_blind_execution_started", "lock_sha256": digest(lock_path), "jobs": len(jobs)})
    elif read(start)["lock_sha256"] != digest(lock_path):
        raise ValueError("execution directory belongs to a different lock")
    python = lock["environment"]["interpreter"]["path"]
    bindings = []
    for number, job in enumerate(jobs, start=1):
        name = f"{job['object_id']}-r{job['repeat_index']}-{job['arm']}"
        case = output / "cases" / name
        timed = case / "timed-result.json"
        if timed.exists():
            bindings.append({"path": str(timed.relative_to(output)), "sha256": digest(timed)})
            continue
        job_path = output / "jobs" / f"{name}.json"
        if not job_path.exists():
            write_once(job_path, {**job, "directory": str(case)})
        environment = os.environ.copy()
        environment.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        environment["PYTHONPATH"] = lock["code_root"]
        started = time.perf_counter()
        result = subprocess.run(
            [python, str(Path(__file__).resolve()), "_worker", "--lock", str(lock_path.resolve()), "--job", str(job_path.resolve())],
            cwd=lock["code_root"], env=environment, capture_output=True, text=True, timeout=9000,
        )
        elapsed = time.perf_counter() - started
        log = output / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8")
        if result.returncode or len(result.stdout.splitlines()) != 1:
            raise RuntimeError(f"case failed: {name}; see {log}")
        row = json.loads(result.stdout.strip())
        row["wall_seconds"] = elapsed
        write_once(timed, row)
        bindings.append({"path": str(timed.relative_to(output)), "sha256": digest(timed)})
        print(f"{number}/{len(jobs)} {name} {elapsed:.2f}s", flush=True)
    completion = {"schema": SCHEMA, "phase": "reference_blind_execution_complete", "lock_sha256": digest(lock_path), "rows": sorted(bindings, key=lambda row: row["path"]), "references_opened": False}
    write_once(output / "execution.json", completion)


def score(lock_path: Path, scoring_lock_path: Path, execution_path: Path, output: Path) -> None:
    lock = validate_lock(lock_path, environment=False)
    scoring = read(scoring_lock_path)
    execution = read(execution_path)
    if execution["phase"] != "reference_blind_execution_complete" or execution["references_opened"] is not False:
        raise ValueError("execution was not completed reference blind")
    if execution["lock_sha256"] != digest(lock_path) or scoring["evaluation_lock"]["sha256"] != digest(lock_path):
        raise ValueError("scoring and execution locks differ")
    rows = []
    for binding in execution["rows"]:
        path = execution_path.parent / binding["path"]
        if digest(path) != binding["sha256"]:
            raise ValueError("execution record changed")
        rows.append(read(path))
    if len(rows) != 840:
        raise ValueError("execution has wrong number of cases")
    references = {}
    with Path(scoring["reference_catalog"]["path"]).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("eligible"):
                references[record["object_id"]] = [item["vector"] for item in record["solutions"]]
    by_key = {(row["object_id"], row["arm"], row["repeat_index"]): row for row in rows}
    object_rows = []
    for metadata in lock["objects"]:
        object_id = metadata["object_id"]
        entry = {"object_id": object_id, "fold": metadata["fold"], "arms": {}}
        for arm in ("classical", "delphi"):
            repeats = [by_key[(object_id, arm, index)] for index in range(REPEATS)]
            if not all(row["completed"] for row in repeats):
                raise ValueError(f"incomplete {object_id} {arm}")
            fit = repeats[0]["selected_fit"]
            if any(prior.axial_angle_degrees(fit["axis"], row["selected_fit"]["axis"]) > 1e-6 for row in repeats[1:]):
                raise ValueError(f"unstable selected pole for {object_id} {arm}")
            entry["arms"][arm] = {
                "axis": fit["axis"],
                "reference_error_degrees": prior._reference_error(fit["axis"], references[object_id]),
                "final_relative_rms": fit["final_relative_rms"],
                "mean_wall_seconds": sum(row["wall_seconds"] for row in repeats) / REPEATS,
            }
        object_rows.append(entry)

    def endpoints(sample: list[dict]) -> dict:
        c = [row["arms"]["classical"] for row in sample]
        d = [row["arms"]["delphi"] for row in sample]
        n = len(sample)
        return {
            "speed_ratio_classical_over_delphi": sum(row["mean_wall_seconds"] for row in c) / sum(row["mean_wall_seconds"] for row in d),
            "classical_mean_seconds": sum(row["mean_wall_seconds"] for row in c) / n,
            "delphi_mean_seconds": sum(row["mean_wall_seconds"] for row in d) / n,
            "classical_mean_error_degrees": sum(row["reference_error_degrees"] for row in c) / n,
            "delphi_mean_error_degrees": sum(row["reference_error_degrees"] for row in d) / n,
            "classical_minus_delphi_mean_error_degrees": sum(a["reference_error_degrees"] - b["reference_error_degrees"] for a, b in zip(c, d)) / n,
            "classical_minus_delphi_within20": (sum(row["reference_error_degrees"] <= 20 for row in c) - sum(row["reference_error_degrees"] <= 20 for row in d)) / n,
            "classical_within20": sum(row["reference_error_degrees"] <= 20 for row in c),
            "delphi_within20": sum(row["reference_error_degrees"] <= 20 for row in d),
            "rms_ratio_classical_over_delphi": prior._geometric_mean([a["final_relative_rms"] / b["final_relative_rms"] for a, b in zip(c, d)]),
        }

    point = endpoints(object_rows)
    folds = {fold: [row for row in object_rows if row["fold"] == fold] for fold in range(5)}
    rng = random.Random(SEED)
    draws = [endpoints([rng.choice(folds[fold]) for fold in range(5) for _ in folds[fold]]) for _ in range(BOOTSTRAPS)]
    intervals = {key: [float(np.percentile([row[key] for row in draws], 2.5)), float(np.percentile([row[key] for row in draws], 97.5))] for key in ("speed_ratio_classical_over_delphi", "classical_minus_delphi_mean_error_degrees", "classical_minus_delphi_within20", "rms_ratio_classical_over_delphi")}
    quality_match = intervals["classical_minus_delphi_mean_error_degrees"][1] <= 3 and intervals["classical_minus_delphi_within20"][0] >= -0.05 and intervals["rms_ratio_classical_over_delphi"][1] <= 1.01
    report = {"schema": SCHEMA, "phase": "post_execution_reference_scoring", "classical_starts": lock["selected_classical_starts"], "point_estimates": point, "bootstrap_95_intervals": intervals, "comparable_quality_established": quality_match, "speedup_at_least_1_25_established": quality_match and point["speed_ratio_classical_over_delphi"] >= 1.25 and intervals["speed_ratio_classical_over_delphi"][0] > 1, "object_rows": object_rows}
    write_once(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "object_rows"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dev = commands.add_parser("development")
    dev.add_argument("--source-lock", type=Path, required=True)
    dev.add_argument("--source-scoring-lock", type=Path, required=True)
    dev.add_argument("--archive", type=Path, required=True)
    dev.add_argument("--output", type=Path, required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--source-lock", type=Path, required=True)
    frozen.add_argument("--development", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    frozen.add_argument("--scoring-output", type=Path, required=True)
    evaluation = commands.add_parser("execute")
    evaluation.add_argument("--lock", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    scored = commands.add_parser("score")
    scored.add_argument("--lock", type=Path, required=True)
    scored.add_argument("--scoring-lock", type=Path, required=True)
    scored.add_argument("--execution", type=Path, required=True)
    scored.add_argument("--output", type=Path, required=True)
    working = commands.add_parser("_worker")
    working.add_argument("--lock", type=Path, required=True)
    working.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "development":
        development(args.source_lock, args.source_scoring_lock, args.archive, args.output)
    elif args.command == "freeze":
        freeze(args.source_lock, args.development, args.output, args.scoring_output)
    elif args.command == "execute":
        execute(args.lock, args.output)
    elif args.command == "score":
        score(args.lock, args.scoring_lock, args.execution, args.output)
    else:
        worker(args.lock, args.job)


if __name__ == "__main__":
    main()
