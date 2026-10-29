#!/usr/bin/env python3
"""Read checksum-bound text PCK pole constants, without loading SPICE or models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[4] / "data/own-period-20261021/spacecraft"
OBJECTS = (
    ("Ceres", 1, 3414, "pck00011.tpc"),
    ("Vesta", 4, 3415, "pck00011.tpc"),
    ("Lutetia", 21, 113, "pck00011.tpc"),
    ("Ida", 243, 478, "pck00011.tpc"),
    ("Eros", 433, 1714, "pck00011.tpc"),
    ("Gaspra", 951, 477, "pck00011.tpc"),
    ("Itokawa", 25143, None, "itokawa_aizu504.tpc"),
    ("Ryugu", 162173, None, "ryugu_v10.tpc"),
    ("Bennu", 101955, None, "bennu_v17.tpc"),
)

# NAIF codes that do not follow 2000000 + number (Galileo-era flyby targets).
NAIF_BODY = {243: 2431010, 951: 9511010}
# Mathilde (253) has no pole in pck00011.tpc and is left out.


def constants(text: str) -> dict[str, list[float]]:
    """Only read numeric pole assignments inside SPICE begindata sections."""
    if not text.startswith("KPL/PCK"):
        raise ValueError("not a text PCK")
    active, lines = False, []
    for line in text.splitlines():
        marker = line.strip().lower()
        if marker == "\\begindata":
            active = True
        elif marker == "\\begintext":
            active = False
        elif active:
            lines.append(line)
    result = {}
    for match in re.finditer(r"\b(BODY\d+_POLE_(?:RA|DEC))\s*=\s*\(([^)]*)\)", "\n".join(lines), re.I):
        key, raw = match.groups()
        values = [float(value.replace("D", "E").replace("d", "e")) for value in raw.split()]
        result[key.upper()] = values  # Later '=' assignments supersede earlier ones.
    return result


def ecliptic(ra_deg: float, dec_deg: float) -> tuple[float, float, list[float]]:
    # SPICE ECLIPJ2000: fixed obliquity 84381.448 arcsec (IAU 1976).
    ra, dec, eps = map(math.radians, (ra_deg, dec_deg, 84381.448 / 3600))
    x, y, z = math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)
    vec = [x, math.cos(eps) * y + math.sin(eps) * z, -math.sin(eps) * y + math.cos(eps) * z]
    return math.degrees(math.atan2(vec[1], vec[0])) % 360, math.degrees(math.atan2(vec[2], math.hypot(vec[0], vec[1]))), vec


def build(output: Path) -> dict:
    receipt = json.loads((output / "raw/receipts.json").read_text())
    records = {row["path"]: row for row in receipt["files"]}
    rows = []
    for name, number, damit_id, kernel in OBJECTS:
        relative = "pck/" + kernel
        source = records.get(relative)
        if source is None:
            raise ValueError(f"missing kernel receipt: {kernel}; no substitute pole is allowed")
        payload = (output / "raw" / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != source["sha256"] or len(payload) != source["bytes"]:
            raise ValueError(f"kernel hash mismatch: {kernel}")
        values = constants(payload.decode("utf-8-sig"))
        body = NAIF_BODY.get(number, 2000000 + number)
        variables = [f"BODY{body}_POLE_{component}" for component in ("RA", "DEC")]
        coefficients = [values[key] for key in variables]
        if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in coefficients):
            raise ValueError(f"invalid pole polynomial: {name}")
        if any(value != 0 for polynomial in coefficients for value in polynomial[1:]):
            raise ValueError(f"nonconstant pole requires an explicit epoch policy: {name}")
        ra, dec = (polynomial[0] for polynomial in coefficients)
        if not 0 <= ra < 360 or not -90 <= dec <= 90:
            raise ValueError(f"invalid RA/Dec for {name}")
        longitude, latitude, vector = ecliptic(ra, dec)
        rows.append({"object": name, "mpc_number": number, "damit_id": damit_id,
                     "ra_deg": ra, "dec_deg": dec, "lambda_deg": longitude, "beta_deg": latitude,
                     "unit_vector": vector, "kernel_file": relative, "kernel_url": source["url"],
                     "sha256": source["sha256"], "variables": variables, "pole_coefficients": coefficients})
    return {"schema": "delphi.spacecraft-reference-poles.v1", "objects": rows,
            "equatorial_frame": "J2000", "ecliptic_frame": "ECLIPJ2000",
            "obliquity_arcsec": 84381.448, "epoch_jd_tdb": 2451545.0,
            "paper_crosscheck": "pending Claude review before Job 2; no scoring performed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(args.output_dir)
    with (args.output_dir / "reference-poles.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
