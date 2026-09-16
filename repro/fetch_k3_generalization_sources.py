"""Download a named public metadata/data snapshot with a byte-level receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

SOURCES = {
    "alcdef": {
        "alcdef-pds-v1.zip": "https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.alcdef-database_V1_0.zip",
    },
    "tess-metadata": {
        name: "https://archive.konkoly.hu/pub/tssys/dr1/" + name
        for name in ("README", "release.info", "release.merge")
    },
    "damit-identities": {
        "asteroids.csv": "https://damit.cuni.cz/projects/damit/exports/table/asteroids",
    },
    "gaia-dr3-spins": {
        "ReadMe": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/675/A24/ReadMe",
        "table3.dat": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/675/A24/table3.dat",
    },
    "synthetic-evidence": {
        "SHA256SUMS": "https://github.com/HBJ1004/DeLPHI_LC/releases/download/v1.0.0/SHA256SUMS",
        "delphi-k3-synthetic-data-v1.0.0.tar.gz": "https://github.com/HBJ1004/DeLPHI_LC/releases/download/v1.0.0/delphi-k3-synthetic-data-v1.0.0.tar.gz",
    },
    "publication-index": {
        "publication-archive-manifest.json": "https://github.com/HBJ1004/DeLPHI_LC/releases/download/v1.0.0/publication-archive-manifest.json",
    },
}


def fetch(source: str, output: Path) -> dict:
    if output.exists():
        raise ValueError("output directory must be new")
    output.mkdir(parents=True)
    records = []
    for name, url in SOURCES[source].items():
        path = output / name
        digest = hashlib.sha256()
        byte_count = 0
        with urlopen(Request(url, headers={"User-Agent": "DeLPHI-generalization-research/1.0"}), timeout=60) as response, path.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
            if byte_count == 0:
                raise ValueError("empty source response")
            resolved = urlsplit(response.url)
            public_final_url = urlunsplit((resolved.scheme, resolved.netloc, resolved.path, "", ""))
            records.append({"file": name, "requested_url": url, "final_url": public_final_url,
                            "http_date": response.headers.get("Date"), "content_type": response.headers.get("Content-Type"),
                            "bytes": byte_count, "sha256": digest.hexdigest()})
        print(json.dumps(records[-1]), flush=True)
    receipt = {"schema": "delphi.k3-source-fetch.v1", "source": source,
               "fetched_utc": datetime.now(timezone.utc).isoformat(), "files": records,
               "requested_role": (
                   "external_reference_spin_table" if source == "gaia-dr3-spins"
                   else "photometry_and_identity_metadata_not_reference_pole_tables"
               ),
               "scope": "raw public input data/identity metadata, not an independence attestation"}
    with (output / "fetch-receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=tuple(SOURCES), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        fetch(args.source, args.output_directory)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
