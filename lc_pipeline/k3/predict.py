"""JSON command-line inference using an evaluated five-seed fold ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..v2.preprocessing import KnownPeriod, Observation, ObservationEpoch
from .bundle import K3EnsemblePredictor


def read_observations(path: Path) -> tuple[str, KnownPeriod, tuple[ObservationEpoch, ...]]:
    """Validate explicit epochs, positive relative fluxes, geometry, and period."""
    data = json.loads(path.read_text())
    if data.get("schema") != "delphi.k3-observations.v1":
        raise ValueError("input schema must be delphi.k3-observations.v1")
    period = KnownPeriod(**data["known_period"])
    epochs = tuple(ObservationEpoch(row["epoch_id"], tuple(
        Observation(**observation) for observation in row["observations"]))
        for row in data["epochs"])
    return data["object_id"], period, epochs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    object_id, period, epochs = read_observations(args.input)
    predictor = K3EnsemblePredictor(args.bundle, device=args.device)
    result = predictor.predict(epochs, known_period=period, object_id=object_id)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
