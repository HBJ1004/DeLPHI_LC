#!/usr/bin/env python3
"""List the six starting poles of a DeLPHI prediction, and optionally check it.

Each of the three candidate axes is a unit vector (x, y, z) in ecliptic J2000.
An axis has two ends, so the three axes give six starting poles for inversion:

    lambda = atan2(y, x) mod 360 deg,   beta = asin(z)
    opposite end: (lambda + 180 deg) mod 360 deg,  -beta

Usage (from the repository root)::

    python examples/prediction_to_poles.py prediction.json
    python examples/prediction_to_poles.py prediction.json \
        --compare examples/astraea/expected_prediction.json
    python examples/prediction_to_poles.py prediction.json --reference-pole 124 39

The candidates are unranked: start an inversion from all six poles and keep
the fit with the lowest residual.  ``--reference-pole`` only prints the angle
to a known solution (an evaluation quantity, not something to select with).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def axis_to_poles(axis_xyz) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return both ends of an axis as (lambda_deg, beta_deg) pairs."""
    x, y, z = (float(value) for value in axis_xyz)
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    lam = math.degrees(math.atan2(y, x)) % 360.0
    beta = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    return (lam, beta), ((lam + 180.0) % 360.0, -beta)


def pole_to_axis(lambda_deg: float, beta_deg: float) -> tuple[float, float, float]:
    lam, beta = math.radians(lambda_deg), math.radians(beta_deg)
    return (math.cos(beta) * math.cos(lam), math.cos(beta) * math.sin(lam), math.sin(beta))


def axial_angle_deg(a, b) -> float:
    """Angle between two axes, treating an axis and its opposite as equal (0-90 deg)."""
    dot = sum(float(p) * float(q) for p, q in zip(a, b))
    norms = math.sqrt(sum(float(p) ** 2 for p in a)) * math.sqrt(sum(float(q) ** 2 for q in b))
    return math.degrees(math.acos(min(1.0, abs(dot) / norms)))


def starting_poles(prediction: dict) -> list[dict]:
    """Six rows: candidate number, end (+/-), lambda, beta."""
    if prediction.get("status") != "ok":
        raise ValueError(f"prediction status is {prediction.get('status')!r}, not 'ok'")
    rows = []
    for number, axis in enumerate(prediction["axes"], start=1):
        for end, (lam, beta) in zip("+-", axis_to_poles(axis["axis_xyz"])):
            rows.append({"candidate": number, "end": end, "lambda_deg": lam, "beta_deg": beta})
    return rows


def max_difference_deg(prediction: dict, expected: dict) -> float:
    """Largest angle between each expected axis and the nearest predicted axis."""
    return max(
        min(axial_angle_deg(e["axis_xyz"], p["axis_xyz"]) for p in prediction["axes"])
        for e in expected["axes"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--compare", type=Path, help="expected prediction to compare with")
    parser.add_argument("--tolerance-deg", type=float, default=0.1,
                        help="largest acceptable axis difference for --compare (default 0.1)")
    parser.add_argument("--reference-pole", type=float, nargs=2, metavar=("LAMBDA", "BETA"),
                        help="print the angle of each axis to this pole (evaluation only)")
    args = parser.parse_args(argv)

    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    print(f"object {prediction.get('object_id')}  bundle {prediction.get('bundle_id')}")
    print("cand end  lambda(deg)  beta(deg)")
    for row in starting_poles(prediction):
        print(f"{row['candidate']:>4}  {row['end']}   {row['lambda_deg']:9.1f}  {row['beta_deg']:+9.1f}")
    if args.reference_pole:
        reference = pole_to_axis(*args.reference_pole)
        for number, axis in enumerate(prediction["axes"], start=1):
            angle = axial_angle_deg(axis["axis_xyz"], reference)
            print(f"candidate {number}: {angle:.1f} deg from the reference pole")
    if args.compare:
        expected = json.loads(args.compare.read_text(encoding="utf-8"))
        difference = max_difference_deg(prediction, expected)
        verdict = "matches" if difference <= args.tolerance_deg else "DIFFERS FROM"
        print(f"largest axis difference {difference:.4f} deg: {verdict} {args.compare}")
        return 0 if difference <= args.tolerance_deg else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
