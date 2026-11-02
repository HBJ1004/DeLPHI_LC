"""Own rotation periods for the lower-quality DAMIT census (9,783 asteroids).

    python -m repro.own_period.lowq_search --output OP/lowq/search.jsonl --max-seconds 3300
    python -m repro.own_period.lowq_search --output OP/lowq/search.jsonl --finalize OP/lowq/own-periods-lowq.json

The GPU Fourier search of ``repro/period_search/search.py`` is applied, with the
settings of the 170-asteroid search (``fourier_gpu``, resolution 1, 180 s budget,
sparse-survey fallback), to exactly the observations that the published census
(``repro/run_k3_lowq_census.py predict``) gives the networks: every source epoch
of ``parse_damit_lightcurve(lc.txt)`` is one curve, magnitude -2.5 log10 of the
relative brightness, times as supplied.  Only the object identity and the
lightcurve path are read from the census index; its period column is not read.

The run is resumable: each finished asteroid is appended to the JSONL file at
once, and a restart skips the asteroids already there.  ``--max-seconds`` stops
before starting a new asteroid when the budget would be exceeded, so that each
chunk holds the GPU lock for a bounded time.  ``--finalize`` writes the period
file (``periods_hours`` + ``provenance``) once, after every asteroid is done.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from lc_pipeline.v2.data import parse_damit_lightcurve
from repro.own_period import DataPath
from repro.period_search.search import Curve, estimate

ROOT = DataPath("../../..")
FOLLOWUP = DataPath("../..")
SNAPSHOT = ROOT / "damit-20250610T000301Z"
CENSUS = FOLLOWUP / "source/experiments/lowq_damit_census_20260917"
INDEX = CENSUS / "input-index.json"
INDEX_SHA256 = "392d2dda3336bee1c4639b93f90fc1db4225aff57bb30c78d5435bbb0625a4be"
PROVENANCE = "own-fourier-gpu-top1"
SEARCH_SOURCE = Path(__file__).resolve().parents[1] / "period_search/search.py"
# Wall time reserved for one more asteroid when deciding whether to start it.
RESERVE_SECONDS = 240.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def census_objects() -> list[dict]:
    if sha256(INDEX) != INDEX_SHA256:
        raise ValueError("census input index differs from the published one")
    rows = json.loads(INDEX.read_text())["objects"]
    # Identity and lightcurve path only; the DAMIT period column stays unread.
    return [{"object_id": r["object_id"], "lightcurve_path": r["lightcurve_path"]} for r in rows]


def curves_of(epochs) -> list[Curve]:
    """One curve per source epoch, exactly the observations of the tokenizer."""
    curves = []
    for epoch in epochs:
        a = np.asarray(
            [[o.time_jd, o.relative_brightness] for o in epoch.observations], dtype=float
        )
        a = a[np.isfinite(a).all(axis=1) & (a[:, 1] > 0)]
        a = a[np.argsort(a[:, 0])]
        if len(a) >= 2:
            curves.append(Curve(a[:, 0], -2.5 * np.log10(a[:, 1])))
    return curves


def search_one(row: dict) -> dict:
    started = time.perf_counter()
    epochs = parse_damit_lightcurve(SNAPSHOT / row["lightcurve_path"])
    base = {
        "object_id": row["object_id"],
        "lightcurve_sha256": sha256(SNAPSHOT / row["lightcurve_path"]),
        "epoch_count": len(epochs),
        "observation_count": sum(len(e.observations) for e in epochs),
    }
    if any(len(e.observations) < 2 for e in epochs):
        # The published census rejected this input before the networks (one
        # source epoch with fewer than two observations); it keeps that status.
        return {**base, "status": "input_rejected_published", "period_hours": None,
                "ranked_periods": [], "input_regime": None,
                "seconds": time.perf_counter() - started}
    try:
        result = estimate(
            curves_of(epochs), "fourier_gpu", resolution=1, timeout=180, sparse_fallback=True
        )
    except TimeoutError as error:
        result = {"status": "timeout", "reason": str(error), "ranked_periods": []}
    ranked = [float(v) for v in (result.get("ranked_periods") or [])]
    ok = result.get("status") == "ok" and bool(ranked)
    return {
        **base,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "input_regime": result.get("input_regime"),
        "observing_intervals": result.get("observing_intervals"),
        "ranked_periods": ranked[:5],
        "period_hours": ranked[0] if ok else None,
        "seconds": time.perf_counter() - started,
    }


def done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open() as handle:
        return {json.loads(line)["object_id"] for line in handle if line.strip()}


def run(output: Path, max_seconds: float, limit: int | None, ids: list[str] | None) -> int:
    rows = census_objects()
    if ids:
        wanted = set(ids)
        rows = [r for r in rows if r["object_id"] in wanted]
    if limit is not None:
        rows = rows[:limit]
    finished = done_ids(output)
    todo = [r for r in rows if r["object_id"] not in finished]
    print(f"{len(finished)} finished, {len(todo)} to do", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = 0
    with output.open("a", buffering=1) as stream:
        for row in todo:
            if time.perf_counter() - started > max_seconds - RESERVE_SECONDS:
                print(f"budget reached after {count} asteroids", flush=True)
                break
            item = search_one(row)
            stream.write(json.dumps(item, sort_keys=True) + "\n")
            count += 1
            if count % 25 == 0:
                print(
                    f"{count} done this chunk, {time.perf_counter() - started:.0f} s; last "
                    f"{item['object_id']} {item['status']} {item['seconds']:.1f} s",
                    flush=True,
                )
    remaining = len(todo) - count
    print(f"chunk finished: {count} new, {remaining} remaining", flush=True)
    return remaining


def finalize(search: Path, destination: Path) -> None:
    rows = census_objects()
    items = {}
    with search.open() as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item["object_id"] in items:
                    raise ValueError(f"duplicate search row {item['object_id']}")
                items[item["object_id"]] = item
    missing = [r["object_id"] for r in rows if r["object_id"] not in items]
    if missing:
        raise ValueError(f"{len(missing)} asteroids not searched yet")
    payload = {
        "schema": "delphi.own-periods.v1",
        "provenance": PROVENANCE,
        "cohort": "lowq",
        "search_call": 'estimate(curves, "fourier_gpu", resolution=1, timeout=180, sparse_fallback=True)',
        "search_source_sha256": sha256(SEARCH_SOURCE),
        "census_index_sha256": INDEX_SHA256,
        "search_jsonl_sha256": sha256(search),
        "periods_hours": {r["object_id"]: items[r["object_id"]]["period_hours"] for r in rows},
        "details": {
            r["object_id"]: {
                k: items[r["object_id"]].get(k)
                for k in ("status", "input_regime", "ranked_periods", "seconds", "reason")
            }
            for r in rows
        },
    }
    with destination.open("x") as stream:
        stream.write(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    found = sum(v is not None for v in payload["periods_hours"].values())
    print(destination, found, "of", len(rows), "with a period")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="search JSONL (appended)")
    parser.add_argument("--max-seconds", type=float, default=3300.0)
    parser.add_argument("--limit", type=int, default=None, help="first N census asteroids only")
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--finalize", type=Path, default=None)
    args = parser.parse_args()
    if args.finalize is not None:
        finalize(args.output, args.finalize)
        return 0
    remaining = run(args.output, args.max_seconds, args.limit, args.ids)
    # Exit code 3 tells the chunk driver that more chunks are needed.
    return 3 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
