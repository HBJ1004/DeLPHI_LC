#!/usr/bin/env python3
"""Add JPL Horizons geometry to your own photometry (DAMIT conventions).

Input: a CSV file with one row per observation and the columns

    epoch_id, jd, mag            (magnitudes; optional column mag_err)
or  epoch_id, jd, flux           (linear relative flux; optional column flux_err)

where ``epoch_id`` names the lightcurve (e.g. one night with one telescope) and
``jd`` is the observation mid-time as a Julian date, as seen at the observer
(not light-time corrected).  Output: a ``delphi.k3-observations.v1`` file.

For each observation the script asks JPL Horizons (via astroquery) for

  1. the light time LT from the asteroid to the observer at ``jd``;
  2. at the light-time-corrected time t = jd - LT, the geometric heliocentric
     position of the asteroid r_helio and the geometric position of the
     asteroid relative to the observer r_obs, ecliptic J2000, au.

It then writes, as DAMIT lc.txt does,

    time_jd                              = jd - LT
    sun_asteroid_ecliptic_j2000_au       = -r_helio   (asteroid -> Sun)
    observer_asteroid_ecliptic_j2000_au  = -r_obs     (asteroid -> observer)

Requires ``pip install astroquery`` and internet access.  Example::

    python examples/horizons_geometry.py --target 5 --location 500@399 \
        --photometry my_photometry.csv --object-id my-astraea \
        --period-hours 16.80059 --period-provenance "DAMIT model 1816" \
        --output observations.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _vectors(target: str, location: str, epochs: list[float], aberrations: str):
    """One Horizons VECTORS query in the ecliptic J2000 frame, au and days."""
    from astroquery.jplhorizons import Horizons

    table = Horizons(id=target, id_type="smallbody", location=location, epochs=epochs).vectors(
        refplane="ecliptic", aberrations=aberrations
    )
    xyz = np.column_stack([np.asarray(table[name], dtype=float) for name in ("x", "y", "z")])
    return xyz, np.asarray(table["lighttime"], dtype=float)


def horizons_geometry(target: str, jd_observed, location: str = "500@399", chunk: int = 50):
    """Return (time_jd, sun_vectors, observer_vectors) in the DAMIT convention.

    target:   asteroid number or designation known to Horizons, e.g. "5".
    location: observer, e.g. "500@399" (geocentre, as DAMIT) or an MPC site
              code such as "568".  The difference is negligible for the pole.
    Horizons interprets the epochs of a vector query as TDB; for UTC input the
    ~1 minute offset moves the geometry by a negligible amount, and time_jd
    stays in your own time scale (only LT is subtracted).
    """
    jd_observed = np.asarray(jd_observed, dtype=float)
    times, suns, observers = [], [], []
    for start in range(0, jd_observed.size, chunk):
        jd = jd_observed[start : start + chunk].tolist()
        # 1. light time at the observation time (astrometric = light-time solved)
        _, light_time = _vectors(target, location, jd, "astrometric")
        emitted = (np.asarray(jd) - light_time).tolist()
        # 2. geometric positions at the light-time-corrected time
        helio, _ = _vectors(target, "500@10", emitted, "geometric")
        relative, _ = _vectors(target, location, emitted, "geometric")
        times.append(np.asarray(emitted))
        suns.append(-helio)  # asteroid -> Sun
        observers.append(-relative)  # asteroid -> observer
    return np.concatenate(times), np.vstack(suns), np.vstack(observers)


def read_photometry(path: Path):
    """Read the CSV; convert magnitudes to linear relative flux if needed."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    columns = set(rows[0])
    epoch_ids = [row["epoch_id"].strip() for row in rows]
    jd = np.asarray([float(row["jd"]) for row in rows])
    if "flux" in columns:
        flux = np.asarray([float(row["flux"]) for row in rows])
        error = [float(row["flux_err"]) if row.get("flux_err") else None for row in rows]
    elif "mag" in columns:
        mag = np.asarray([float(row["mag"]) for row in rows])
        # Linear relative flux from magnitude; the reference magnitude cancels
        # in DeLPHI (each lightcurve is centred), so the median is used here.
        flux = 10.0 ** (-0.4 * (mag - np.median(mag)))
        error = [
            0.4 * math.log(10.0) * f * float(row["mag_err"]) if row.get("mag_err") else None
            for f, row in zip(flux, rows)
        ]
    else:
        raise SystemExit("photometry CSV needs a 'flux' or a 'mag' column")
    return epoch_ids, jd, flux, error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", required=True, help="Horizons small-body ID, e.g. 5")
    parser.add_argument("--location", default="500@399", help="observer (default geocentre)")
    parser.add_argument("--photometry", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--period-hours", type=float, required=True)
    parser.add_argument("--period-provenance", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a new filename")

    epoch_ids, jd, flux, error = read_photometry(args.photometry)
    time_jd, sun, observer = horizons_geometry(args.target, jd, args.location)
    epochs: dict[str, list[dict]] = {}
    for index, epoch_id in enumerate(epoch_ids):
        row = {
            "time_jd": float(time_jd[index]),
            "relative_brightness": float(flux[index]),
            "sun_asteroid_ecliptic_j2000_au": [float(v) for v in sun[index]],
            "observer_asteroid_ecliptic_j2000_au": [float(v) for v in observer[index]],
        }
        if error[index] is not None:
            row["measured_error"] = float(error[index])
        epochs.setdefault(epoch_id, []).append(row)
    document = {
        "schema": "delphi.k3-observations.v1",
        "object_id": args.object_id,
        "known_period": {"hours": args.period_hours, "provenance": args.period_provenance},
        "epochs": [{"epoch_id": key, "observations": value} for key, value in epochs.items()],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")
    print(f"wrote {args.output}: {len(epochs)} lightcurves, {len(epoch_ids)} observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
