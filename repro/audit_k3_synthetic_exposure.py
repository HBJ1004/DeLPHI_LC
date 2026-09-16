"""Verify the public synthetic archive and extract only provenance JSON records."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path, PurePosixPath

from lc_pipeline.k3.generalization_diagnostics import sha256, write_new_json


def audit(archive: Path, package_manifest: Path, identity_table: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("synthetic exposure output must be new")
    package = json.loads(package_manifest.read_text())
    records = [row for row in package["files"] if row["role"] == "synthetic-data"]
    if len(records) != 1:
        raise ValueError("publication index does not identify one synthetic archive")
    expected = records[0]
    digest = sha256(archive)
    if digest != expected["sha256"] or archive.stat().st_size != expected["size_bytes"]:
        raise ValueError("synthetic archive differs from public publication manifest")
    output.mkdir(parents=True)
    manifests = []
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if not member.name.endswith(".json"):
                continue
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile() or member.size > 16 * 1024 * 1024:
                raise ValueError("unsafe archive metadata member")
            handle = stream.extractfile(member)
            if handle is None:
                raise ValueError("unreadable metadata member")
            payload = handle.read()
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("archive metadata must be a JSON object")
            path = output / "metadata" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as target:
                target.write(payload)
            manifests.append({"logical_path": member.name, "path": path.relative_to(output).as_posix(),
                              "sha256": sha256(path), "schema": value.get("schema"),
                              "keys": sorted(value)})
    aliases = []
    seen_internal, seen_number = set(), set()
    with identity_table.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            internal = int(row["id"])
            if internal in seen_internal:
                raise ValueError("duplicate official DAMIT ID")
            seen_internal.add(internal)
            if not row["number"].strip():
                continue
            number = int(row["number"])
            if number in seen_number:
                raise ValueError("duplicate official MPC number")
            seen_number.add(number)
            aliases.append({"object_id": f"asteroid_{internal}", "damit_id": internal,
                            "damit_identity": f"damit:{internal}",
                            "physical_identity": f"mpc:{number}",
                            "mpc_number": number, "name": row["name"]})
    write_new_json(output / "full-identity-aliases.json", {
        "schema": "delphi.k3-survey-identity-map.v1", "identity_table_sha256": sha256(identity_table),
        "purpose": "full_official_alias_registry_not_a_model_evaluation_cohort", "objects": aliases})
    result = {"schema": "delphi.k3-synthetic-exposure-metadata.v1", "archive_sha256": digest,
              "publication_manifest_sha256": sha256(package_manifest), "archive_matches_publication": True,
              "metadata_record_count": len(manifests), "files": manifests,
              "full_identity_aliases_sha256": sha256(output / "full-identity-aliases.json"),
              "training_arrays_extracted": False, "historical_exposure_registry_complete": False}
    write_new_json(output / "metadata-receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--publication-manifest", type=Path, required=True)
    parser.add_argument("--identity-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.archive, args.publication_manifest, args.identity_table, args.output)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))


if __name__ == "__main__":
    main()
