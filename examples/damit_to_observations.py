#!/usr/bin/env python3
"""Convert one DAMIT asteroid into a DeLPHI observations JSON file.

DAMIT (https://damit.cuni.cz) distributes the lightcurves of each asteroid as a
plain-text ``lc.txt`` file.  Its layout is::

    <number of lightcurves>
    <number of points> <calibrated flag 0/1>      # header of lightcurve 1
    JD  intensity  Sx Sy Sz  Ex Ey Ez               # one row per point
    ...
    <number of points> <calibrated flag 0/1>      # header of lightcurve 2
    ...

where JD is the light-time-corrected Julian date, ``intensity`` is linear
relative brightness (not magnitude), and ``S`` and ``E`` are the vectors from
the asteroid to the Sun and to the Earth in ecliptic J2000 coordinates, in au.
That is already the DeLPHI convention, so the conversion copies the numbers
unchanged and keeps every DAMIT lightcurve as one DeLPHI "epoch".

DeLPHI also needs a rotation period with a statement of where it came from.
Either pass it yourself (``--period-hours`` and ``--period-provenance``) or let
the script read it from the DAMIT model table (``--damit-id``), as the paper did.

Examples (run from the repository root)::

    # (a) a local lc.txt and a period you supply
    python examples/damit_to_observations.py --lc lc.txt \
        --object-id my-asteroid --period-hours 16.80059 \
        --period-provenance "DAMIT model 1816 (Hanus et al. 2017)" \
        --output observations.json

    # (b) download lightcurves and period of DAMIT asteroid 103, (5) Astraea
    python examples/damit_to_observations.py --damit-id 103 \
        --output observations.json

    # (c) as (b), but offline from a local DAMIT dump
    python examples/damit_to_observations.py --damit-id 103 \
        --lc DAMIT/files/asteroid_103/lc.txt \
        --models-table DAMIT/tables/asteroid_models.csv \
        --output observations.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

# Make the repository importable when the script is run as
# ``python examples/damit_to_observations.py`` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_pipeline.v2.data import parse_damit_lightcurve  # noqa: E402

DAMIT_BASE_URL = "https://damit.cuni.cz/projects/damit"
MODEL_TABLE_URL = f"{DAMIT_BASE_URL}/exports/table/asteroid_models"
ASTEROID_TABLE_URL = f"{DAMIT_BASE_URL}/exports/table/asteroids"
# All lightcurves of one asteroid, in the lc.txt layout described above.
LIGHTCURVE_URL = DAMIT_BASE_URL + "/light_curves/exportAllForAsteroid/{damit_id}/plaintext"
# The paper used models with a DAMIT quality flag of 3 or higher.
MINIMUM_QUALITY_FLAG = 3.0


def fetch(url: str) -> bytes:
    """Download one DAMIT export (tests replace this function)."""
    request = Request(url, headers={"User-Agent": "DeLPHI-example-converter/1.0"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def read_table(source: str | Path | None, url: str) -> list[dict[str, str]]:
    """Read a DAMIT CSV table from a local file, or download it."""
    # The live DAMIT exports start with a UTF-8 byte-order mark; utf-8-sig drops it.
    if source is not None:
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = fetch(url).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def choose_model(models: list[dict[str, str]], damit_id: int, model_id: int | None) -> dict[str, str]:
    """Pick the DAMIT model whose period is used.

    Default (the paper's rule): among the models of this asteroid with quality
    flag >= 3, take the one with the lowest model id.  ``--model-id`` overrides.
    """
    rows = [row for row in models if row.get("asteroid_id") == str(damit_id)]
    if not rows:
        raise SystemExit(f"DAMIT has no models for asteroid id {damit_id}")
    if model_id is not None:
        chosen = [row for row in rows if row["id"] == str(model_id)]
        if not chosen:
            raise SystemExit(f"model {model_id} does not belong to DAMIT asteroid {damit_id}")
        return chosen[0]

    def quality(row: dict[str, str]) -> float:
        try:
            return float(row.get("quality_flag") or "nan")
        except ValueError:
            return math.nan

    accepted = sorted((row for row in rows if quality(row) >= MINIMUM_QUALITY_FLAG),
                      key=lambda row: int(row["id"]))
    if not accepted:
        listing = ", ".join(
            f"model {row['id']} (P={row['period']} h, quality={row.get('quality_flag') or 'none'})"
            for row in rows
        )
        raise SystemExit(
            f"no model of asteroid {damit_id} has quality flag >= 3; "
            f"choose one with --model-id: {listing}"
        )
    return accepted[0]


def asteroid_label(asteroids: list[dict[str, str]] | None, damit_id: int) -> str:
    """Return e.g. '(5) Astraea' for the provenance text, if the table is available."""
    for row in asteroids or []:
        if row.get("id") == str(damit_id):
            number, name = row.get("number", ""), row.get("name") or row.get("designation", "")
            return f"({number}) {name}".strip() if number else name
    return ""


def build_document(epochs, *, object_id: str, period_hours: float, provenance: str) -> dict:
    """Assemble the delphi.k3-observations.v1 document."""
    return {
        "schema": "delphi.k3-observations.v1",
        "object_id": object_id,
        "known_period": {"hours": period_hours, "provenance": provenance},
        "epochs": [
            {
                # One DAMIT lightcurve = one DeLPHI epoch.  Keep the boundaries.
                "epoch_id": f"damit-lc-{index:04d}",
                "observations": [
                    {
                        "time_jd": row.time_jd,  # light-time-corrected JD
                        "relative_brightness": row.relative_brightness,  # linear
                        "sun_asteroid_ecliptic_j2000_au": list(row.sun_asteroid_ecliptic_j2000_au),
                        "observer_asteroid_ecliptic_j2000_au": list(row.observer_asteroid_ecliptic_j2000_au),
                        # DAMIT gives no per-point uncertainties, so
                        # "measured_error" is omitted, as in the paper.
                    }
                    for row in epoch.observations
                ],
            }
            for index, epoch in enumerate(epochs)
        ],
    }


def write_document(document: dict, path: Path) -> None:
    """Write readable JSON with one observation per line (smaller than indent=2)."""
    head = {key: value for key, value in document.items() if key != "epochs"}
    lines = ["{"]
    for key, value in head.items():
        lines.append(f"  {json.dumps(key)}: {json.dumps(value)},")
    lines.append('  "epochs": [')
    for e_index, epoch in enumerate(document["epochs"]):
        lines.append(f'    {{"epoch_id": {json.dumps(epoch["epoch_id"])}, "observations": [')
        rows = [json.dumps(obs, allow_nan=False) for obs in epoch["observations"]]
        lines.append(",\n".join(f"      {row}" for row in rows))
        lines.append("    ]}" + ("," if e_index < len(document["epochs"]) - 1 else ""))
    lines.append("  ]")
    lines.append("}")
    with path.open("x", encoding="utf-8") as handle:  # never overwrite
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="new observations JSON file")
    parser.add_argument("--lc", type=Path, help="local DAMIT lc.txt (default: download with --damit-id)")
    parser.add_argument("--damit-id", type=int,
                        help="DAMIT asteroid id (the number in asteroid_<id>), e.g. 103 for (5) Astraea")
    parser.add_argument("--object-id",
                        help="object_id to write (default: asteroid_<damit-id>, the benchmark ID)")
    parser.add_argument("--period-hours", type=float, help="rotation period in hours")
    parser.add_argument("--period-provenance", help="where the period comes from")
    parser.add_argument("--models-table", type=Path,
                        help="local DAMIT asteroid_models.csv (default: download)")
    parser.add_argument("--asteroids-table", type=Path,
                        help="local DAMIT asteroids.csv, only used to name the asteroid in the provenance")
    parser.add_argument("--model-id", type=int, help="DAMIT model whose period to use")
    parser.add_argument("--save-lc", type=Path, help="also save the downloaded lc.txt here")
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a new filename")
    if args.lc is None and args.damit_id is None:
        parser.error("give --lc, --damit-id, or both")
    if (args.period_hours is None) != (args.period_provenance is None):
        parser.error("--period-hours and --period-provenance must be given together")
    if args.period_hours is None and args.damit_id is None:
        parser.error("without --damit-id, give --period-hours and --period-provenance")
    object_id = args.object_id or (f"asteroid_{args.damit_id}" if args.damit_id is not None else None)
    if not object_id:
        parser.error("give --object-id (or --damit-id)")

    # 1. Lightcurves: local lc.txt, or the DAMIT export for this asteroid.
    if args.lc is not None:
        lc_path = args.lc
    else:
        lc_path = args.save_lc or args.output.with_suffix(".lc.txt")
        if lc_path.exists():
            parser.error(f"{lc_path} already exists; choose another --save-lc")
        lc_path.write_bytes(fetch(LIGHTCURVE_URL.format(damit_id=args.damit_id)))
        print(f"downloaded DAMIT lightcurves to {lc_path}")
    epochs = parse_damit_lightcurve(lc_path)

    # 2. Period: given on the command line, or read from the DAMIT model table.
    if args.period_hours is not None:
        period_hours, provenance = args.period_hours, args.period_provenance.strip()
    else:
        models = read_table(args.models_table, MODEL_TABLE_URL)
        model = choose_model(models, args.damit_id, args.model_id)
        period_hours = float(model["period"])
        table_source = (f"local copy {args.models_table}" if args.models_table
                        else f"{MODEL_TABLE_URL}, retrieved {date.today().isoformat()}")
        asteroids = None
        if args.asteroids_table is not None:
            asteroids = read_table(args.asteroids_table, ASTEROID_TABLE_URL)
        elif args.models_table is None:
            asteroids = read_table(None, ASTEROID_TABLE_URL)
        label = asteroid_label(asteroids, args.damit_id)
        provenance = (
            f"DAMIT model {model['id']} of "
            + (f"{label} (DAMIT asteroid {args.damit_id})" if label else f"DAMIT asteroid {args.damit_id}")
            + f", quality flag {model.get('quality_flag') or 'none'}; period from the "
            f"DAMIT asteroid_models table ({table_source})"
        )

    # 3. Write and re-read with DeLPHI's own validator.
    document = build_document(epochs, object_id=object_id, period_hours=period_hours,
                              provenance=provenance)
    write_document(document, args.output)
    from lc_pipeline.k3.predict import read_observations

    read_observations(args.output)
    n_obs = sum(len(epoch.observations) for epoch in epochs)
    print(f"wrote {args.output}: object_id={object_id}, {len(epochs)} lightcurves, "
          f"{n_obs} observations, period {period_hours} h")
    print(f"period provenance: {provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
