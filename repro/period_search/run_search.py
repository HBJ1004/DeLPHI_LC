"""Isolated, resumable timing harness; result files are never overwritten."""

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "DeLPHI-followup/inputs/training-run/repro/data/damit-20250610T000301Z"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_path(row):
    return ROOT / "damit-20250610T000301Z" / row["lightcurve"]["source_path"]


def prediction_job(path, method, resolution, timeout):
    # Catalog labels are deliberately excluded from the worker interface.
    return dict(path=str(path), method=method, resolution=resolution, timeout=timeout)


def run_metadata(args, rows, methods):
    sources = [HERE / "search.py", HERE / "run_search.py"]
    if "legacy4" in methods:
        sources += [
            HERE.parent / "e2e-period-20260921" / name
            for name in ("period_pilot.py", "fourier4.py")
        ]
    return dict(
        schema_version=2,
        python=sys.version,
        platform=platform.platform(),
        command=sys.argv,
        configuration=dict(
            methods=methods,
            resolution=args.resolution,
            timeout=args.timeout,
            repeats=args.repeats,
            limit=args.limit,
            mode=args.mode,
        ),
        source_sha256={str(p.relative_to(HERE.parent)): digest(p) for p in sources},
        catalog_sha256=digest(DATA / "catalog.jsonl"),
        input_sha256={r["object_id"]: digest(input_path(r)) for r in rows},
        object_order=[r["object_id"] for r in rows],
        thread_environment={
            k: "1"
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        scope="development comparison on previously inspected objects",
        mode=args.mode,
    )


def resume_keys(path, metadata):
    previous = json.loads(path.with_suffix(".metadata.json").read_text())
    for key in (
        "schema_version",
        "python",
        "platform",
        "configuration",
        "source_sha256",
        "catalog_sha256",
        "input_sha256",
        "object_order",
        "thread_environment",
    ):
        if key not in previous or previous[key] != metadata[key]:
            raise ValueError(f"Incompatible resume metadata: {key}")
    config = metadata["configuration"]
    allowed = {
        (obj, method, repeat, config["mode"])
        for obj in metadata["object_order"]
        for method in config["methods"]
        for repeat in range(config["repeats"])
    }
    done = {tuple(key) for key in previous.get("resumed_keys", [])}
    for row in map(json.loads, path.read_text().splitlines()):
        key = (row["object_id"], row["method"], row["repeat"], row["mode"])
        if key in done:
            raise ValueError(f"Duplicate resume row: {key}")
        done.add(key)
    if not done <= allowed:
        raise ValueError("Resume contains rows outside the declared experiment")
    return done


def check_warmup(line, method):
    if not line:
        raise RuntimeError("warm-up worker exited")
    result = json.loads(line)
    status = result.get("status")
    if status not in ("ok", "unresolved"):
        raise RuntimeError(f"{method} warm-up failed without retry: {result.get('reason', status)}")
    if status == "unresolved":
        print(
            f"WARNING: {method} warm-up unresolved; backend initialization is not guaranteed",
            file=sys.stderr,
            flush=True,
        )
    return status


def objects():
    rows = [
        r
        for r in map(json.loads, (DATA / "catalog.jsonl").read_text().splitlines())
        if r["eligible"]
    ]
    return sorted(
        rows,
        key=lambda r: hashlib.sha256(
            ("period-pilot-20260921:" + r["object_id"]).encode()
        ).hexdigest(),
    )


def worker():
    from search import estimate, read_curves

    for line in sys.stdin:
        job = json.loads(line)
        start = time.perf_counter()
        try:
            curves = read_curves(Path(job["path"]))
            result = estimate(curves, job["method"], job["resolution"], job["timeout"])
            result.update(
                n_observations=sum(len(c.time) for c in curves),
                baseline_days=float(
                    max(c.time[-1] for c in curves) - min(c.time[0] for c in curves)
                ),
            )
        except Exception as exc:
            result = dict(
                status="timeout" if isinstance(exc, TimeoutError) else "error",
                reason=f"{type(exc).__name__}: {exc}",
                ranked_periods=[],
                candidates=[],
            )
        result["seconds"] = time.perf_counter() - start
        print(json.dumps(result, allow_nan=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--limit", type=int, default=170)
    ap.add_argument(
        "--methods", default="legacy4,lomb_scargle,fourier_cpu,fourier_gpu,fourier_cv,pdm,entropy"
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mode", choices=["warm", "cold"], default="warm")
    ap.add_argument("--resolution", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--resume", type=Path, help="Previous rows to exclude, written to a NEW output")
    args = ap.parse_args()
    if args.worker:
        worker()
        return

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    if args.output is None:
        ap.error("--output required")
    if args.limit < 1 or args.repeats < 1 or args.resolution < 1 or args.timeout <= 0:
        ap.error("limit, repeats, resolution and timeout must be positive")
    import selectors

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    rows = objects()[: args.limit]
    methods = args.methods.split(",")
    from search import METHODS

    if len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        ap.error("methods must be distinct supported method names")
    metadata = run_metadata(args, rows, methods)
    done = resume_keys(args.resume, metadata) if args.resume else set()
    metadata["resumed_keys"] = sorted(done)
    with args.output.with_suffix(".metadata.json").open("x") as f:
        json.dump(metadata, f, indent=2)
    process = None
    warmed = {}
    with args.output.open("x", buffering=1) as output:
        try:
            for repeat in range(args.repeats):
                for index, row in enumerate(rows):
                    shift = (index + repeat) % len(methods)
                    for method in methods[shift:] + methods[:shift]:
                        key = (row["object_id"], method, repeat, args.mode)
                        if key in done:
                            continue
                        job = prediction_job(input_path(row), method, args.resolution, args.timeout)
                        start = time.perf_counter()
                        if process is None:
                            process = subprocess.Popen(
                                command,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                text=True,
                                env=env,
                                start_new_session=True,
                            )
                            warmed.clear()
                        if args.mode == "warm" and method not in warmed:
                            # Warm EACH backend, including CUDA, before its first timed use.
                            process.stdin.write(json.dumps(job) + "\n")
                            process.stdin.flush()
                            sel = selectors.DefaultSelector()
                            sel.register(process.stdout, selectors.EVENT_READ)
                            if not sel.select(args.timeout + 30):
                                os.killpg(process.pid, 9)
                                process.wait()
                                process = None
                                raise TimeoutError("warm-up worker timeout")
                            discarded = process.stdout.readline()
                            sel.close()
                            warmed[method] = check_warmup(discarded, method)
                            start = time.perf_counter()
                        process.stdin.write(json.dumps(job) + "\n")
                        process.stdin.flush()
                        sel = selectors.DefaultSelector()
                        sel.register(process.stdout, selectors.EVENT_READ)
                        if not sel.select(args.timeout + 30):
                            os.killpg(process.pid, 9)
                            process.wait()
                            process = None
                            result = dict(
                                status="timeout",
                                reason="hard worker timeout",
                                ranked_periods=[],
                                candidates=[],
                            )
                        else:
                            line = process.stdout.readline()
                            if not line:
                                raise RuntimeError("worker exited; see stderr")
                            result = json.loads(line)
                        sel.close()
                        result["worker_seconds"] = result.get("seconds")
                        result["seconds"] = time.perf_counter() - start
                        result["warmup_status"] = (
                            warmed.get(method) if args.mode == "warm" else None
                        )
                        result.update(
                            object_id=row["object_id"], method=method, repeat=repeat, mode=args.mode
                        )
                        output.write(json.dumps(result, allow_nan=False) + "\n")
                        print(
                            index + 1,
                            repeat,
                            method,
                            row["object_id"],
                            result["status"],
                            round(result["seconds"], 3),
                            flush=True,
                        )
                        if args.mode == "cold" and process:
                            process.stdin.close()
                            process.wait(timeout=10)
                            process = None
        finally:
            if process:
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    main()
