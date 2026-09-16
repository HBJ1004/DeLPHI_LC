#!/usr/bin/env python3
"""Materialize individually hash-bound ALCDEF prepared objects for prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(input_path: Path, output_directory: Path) -> dict[str, object]:
    if output_directory.exists():
        raise ValueError(f"output already exists: {output_directory}")
    try:
        source = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prepared manifest: {exc}") from exc
    if source.get("schema") != "delphi.k3-alcdef-prepared-manifest.v1" or not isinstance(source.get("objects"), list):
        raise ValueError("prepared manifest schema mismatch")
    rows = source["objects"]
    identities = [row.get("object_id") for row in rows if isinstance(row, dict)]
    if len(identities) != len(rows) or len(set(identities)) != len(rows) or any(not isinstance(item, str) for item in identities):
        raise ValueError("prepared manifest has invalid identities")
    output_directory.mkdir(parents=True)
    manifest_rows = []
    for row in rows:
        identity = str(row["object_id"])
        path = output_directory / "objects" / f"{identity.replace(':', '_')}.json"
        path.parent.mkdir(exist_ok=True)
        _write(path, row)
        manifest_rows.append({"object_id": identity, "path": path.relative_to(output_directory).as_posix(), "sha256": _sha256(path), "epoch_count": len(row["epochs"]), "observation_count": sum(len(epoch["observations"]) for epoch in row["epochs"])})
    result = {"schema": "delphi.k3-alcdef-prepared-directory.v1", "source_manifest_sha256": _sha256(input_path), "object_count": len(manifest_rows), "objects": manifest_rows}
    _write(output_directory / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = materialize(args.input, args.output_directory)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"object_count": result["object_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
