"""Own rotation periods for the 170 main-sample asteroids.

    python -m repro.own_period.manifest

The period of each asteroid is the top period of the GPU Fourier search
(alias P), taken from the candidate list that was built without reference
data.  The file written here is the only period input of the own-period runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from repro.own_period import DataPath

ROOT = DataPath("../../..")
FOLLOWUP = DataPath("../..")
CANDIDATES = FOLLOWUP / "experiments/period-refinement-20260921/end-to-end/candidates-v1.jsonl"
OUT = DataPath()
PROVENANCE = "own-fourier-gpu-top1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    periods = {}
    for row in map(json.loads, CANDIDATES.read_text().splitlines()):
        if row.get("selection_uses_reference") is not False:
            raise ValueError(f"{row['object_id']}: candidate list used reference data")
        top = [
            source["base_period_hours"]
            for candidate in row["candidates"]
            for source in candidate["sources"]
            if source["method"] == "fourier_gpu" and source["alias"] == "P"
        ]
        periods[row["object_id"]] = float(top[0]) if top else None
    if len(periods) != 170:
        raise ValueError("expected 170 asteroids")
    destination = OUT / "own-periods-170.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(
        json.dumps(
            {
                "schema": "delphi.own-periods.v1",
                "provenance": PROVENANCE,
                "source": str(CANDIDATES.relative_to(ROOT)),
                "source_sha256": sha256(CANDIDATES),
                "periods_hours": periods,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(destination, sum(v is not None for v in periods.values()), "periods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
