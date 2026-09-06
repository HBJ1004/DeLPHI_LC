"""JSON command-line inference using a published or custom K3 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..v2.preprocessing import KnownPeriod, Observation, ObservationEpoch
from .bundle import load_predictor


def read_observations(path: Path) -> tuple[str, KnownPeriod, tuple[ObservationEpoch, ...]]:
    """Validate explicit epochs, positive relative fluxes, geometry, and period."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read observation file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("observation document must be a JSON object")
    if data.get("schema") != "delphi.k3-observations.v1":
        raise ValueError("input schema must be delphi.k3-observations.v1")
    try:
        object_id = data["object_id"]
        period = KnownPeriod(**data["known_period"])
        epochs = tuple(
            ObservationEpoch(
                row["epoch_id"],
                tuple(Observation(**observation) for observation in row["observations"]),
            )
            for row in data["epochs"]
        )
    except KeyError as exc:
        raise ValueError(f"missing required observation field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid observation document: {exc}") from exc
    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("object_id must be a nonempty string")
    return object_id, period, epochs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    try:
        object_id, period, epochs = read_observations(args.input)
        predictor = load_predictor(args.bundle, device=args.device)
        result = predictor.predict(epochs, known_period=period, object_id=object_id)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
