"""Own rotation periods for the ZTF and ALCDEF comparisons.

    python -m repro.own_period.search_external --cohort ztf
    python -m repro.own_period.search_external --cohort alcdef

The GPU Fourier search of ``repro/period_search/search.py`` is applied, with
the settings of the 170-asteroid search, to exactly the observations that
DeLPHI receives.  Each prepared epoch is one curve, with magnitude
-2.5 log10 of the relative brightness.  No reference period is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from repro.period_search.search import Curve, estimate

from repro.own_period import DataPath

ROOT = DataPath("../../..")
FOLLOWUP = DataPath("../..")
PREPARED = {
    "ztf": FOLLOWUP / "data/generalization-20260910/mapped-ztf/prepared-current-endpoint",
    "alcdef": FOLLOWUP / "data/generalization-20260910/locked-cohort/alcdef-prepared-30-20260911",
}
OUT = DataPath()
PROVENANCE = "own-fourier-gpu-top1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ready_objects(cohort: str) -> list[tuple[str, Path, str]]:
    manifest = json.loads((PREPARED[cohort] / "manifest.json").read_text())
    rows = [row for row in manifest["objects"] if cohort == "alcdef" or row["status"] == "ready"]
    return [(row["object_id"], PREPARED[cohort] / row["path"], row["sha256"]) for row in rows]


def curves_of(path: Path) -> list[Curve]:
    curves = []
    for epoch in json.loads(path.read_text())["epochs"]:
        a = np.asarray(
            [[o["time_jd"], o["relative_brightness"]] for o in epoch["observations"]], float
        )
        a = a[np.isfinite(a).all(axis=1) & (a[:, 1] > 0)]
        a = a[np.argsort(a[:, 0])]
        if len(a) >= 2:
            curves.append(Curve(a[:, 0], -2.5 * np.log10(a[:, 1])))
    return curves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=sorted(PREPARED), required=True)
    args = parser.parse_args()
    destination = OUT / f"own-periods-{args.cohort}.json"
    if destination.exists():
        raise FileExistsError(destination)
    periods, details = {}, {}
    for object_id, path, digest in ready_objects(args.cohort):
        if sha256(path) != digest:
            raise ValueError(f"{object_id}: prepared-object hash mismatch")
        started = time.perf_counter()
        try:
            result = estimate(
                curves_of(path), "fourier_gpu", resolution=1, timeout=180, sparse_fallback=True
            )
        except TimeoutError as error:
            result = {"status": "timeout", "reason": str(error), "ranked_periods": []}
        ranked = result.get("ranked_periods") or []
        periods[object_id] = float(ranked[0]) if result.get("status") == "ok" and ranked else None
        details[object_id] = {
            "status": result.get("status"),
            "input_regime": result.get("input_regime"),
            "ranked_periods": ranked[:5],
            "seconds": time.perf_counter() - started,
        }
        print(
            object_id,
            details[object_id]["status"],
            periods[object_id],
            details[object_id]["input_regime"],
            flush=True,
        )
    destination.write_text(
        json.dumps(
            {
                "schema": "delphi.own-periods.v1",
                "provenance": PROVENANCE,
                "cohort": args.cohort,
                "search_source_sha256": sha256(
                    ROOT / "DeLPHI-followup/source-own-period/repro/period_search/search.py"
                ),
                "prepared_manifest_sha256": sha256(PREPARED[args.cohort] / "manifest.json"),
                "periods_hours": periods,
                "details": details,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(destination, sum(v is not None for v in periods.values()), "of", len(periods))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
