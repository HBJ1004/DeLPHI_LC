#!/usr/bin/env python3
"""Freeze independent JPL SBDB rotation-period receipts for a locked cohort.

The command reads only ``mpc:N`` identities in the cohort lock.  It does not
read Gaia values, model files, scores, or any error metric.  Each API response
is retained verbatim with its request URL and SHA-256, so an external period
input remains auditable independently of the Gaia reference axes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SBDB_API = "https://ssd-api.jpl.nasa.gov/sbdb.api"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_lock(path: Path) -> list[tuple[str, int]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read cohort lock: {exc}") from exc
    if document.get("schema") != "delphi.k3-alcdef-gaia-cohort-lock.v1":
        raise ValueError("cohort lock schema mismatch")
    rows = document.get("objects")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cohort lock has no objects")
    result: list[tuple[str, int]] = []
    for row in rows:
        identity = row.get("object_id") if isinstance(row, dict) else None
        if not isinstance(identity, str) or not identity.startswith("mpc:"):
            raise ValueError("cohort lock has an invalid object identity")
        try:
            number = int(identity.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("cohort lock has an invalid MPC number") from exc
        if number <= 0:
            raise ValueError("cohort lock has an invalid MPC number")
        result.append((identity, number))
    if len({item[0] for item in result}) != len(result):
        raise ValueError("cohort lock has duplicate identities")
    return result


def _extract_period(document: object, number: int) -> tuple[float | None, str | None]:
    if not isinstance(document, dict):
        raise ValueError("SBDB response is not an object")
    signature = document.get("signature")
    obj = document.get("object")
    if not isinstance(signature, dict) or "NASA/JPL Small-Body Database" not in str(signature.get("source")):
        raise ValueError("SBDB response has no expected signature")
    if not isinstance(obj, dict) or str(obj.get("des")) != str(number):
        raise ValueError("SBDB resolved a different target")
    values = document.get("phys_par")
    if not isinstance(values, list):
        return None, None
    matches = [row for row in values if isinstance(row, dict) and row.get("name") == "rot_per"]
    if len(matches) > 1:
        raise ValueError("SBDB response has duplicate rotation periods")
    if not matches:
        return None, None
    try:
        value = float(matches[0]["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SBDB rotation period is not numeric") from exc
    if not value > 0:
        raise ValueError("SBDB rotation period is not positive")
    return value, str(matches[0].get("ref")) if matches[0].get("ref") is not None else None


def fetch(lock_path: Path, output: Path, *, timeout_seconds: float) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    rows: list[dict[str, object]] = []
    for identity, number in _read_lock(lock_path):
        parameters = {"sstr": str(number), "phys-par": "1"}
        url = f"{SBDB_API}?{urlencode(parameters)}"
        request = Request(url, headers={"User-Agent": "DeLPHI-K3-followup/1.0 (period provenance)"})
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
                final_url = response.geturl()
                http_date = response.headers.get("Date")
        except OSError as exc:
            raise ValueError(f"SBDB request failed for {identity}: {exc}") from exc
        try:
            parsed = json.loads(payload)
            period, reference = _extract_period(parsed, number)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid SBDB response for {identity}: {exc}") from exc
        rows.append({
            "object_id": identity,
            "mpc_number": number,
            "request_url": url,
            "final_url": final_url,
            "http_date": http_date,
            "response_sha256": _sha256(payload),
            "response_bytes": len(payload),
            "raw_response_utf8": payload.decode("utf-8"),
            "known_period_hours": period,
            "period_reference": reference,
            "status": "ready" if period is not None else "no_public_period",
        })
    content = {
        "schema": "delphi.k3-alcdef-sbdb-periods.v1",
        "purpose": "independent externally supplied rotation-period inputs before Gaia-axis reference access",
        "cohort_lock_path": str(lock_path),
        "cohort_lock_sha256": _sha256(lock_path.read_bytes()),
        "endpoint": SBDB_API,
        "objects": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(content, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    try:
        result = fetch(args.cohort_lock, args.output, timeout_seconds=args.timeout_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"object_count": len(result["objects"]), "ready_count": sum(row["status"] == "ready" for row in result["objects"])}, sort_keys=True))


if __name__ == "__main__":
    main()
