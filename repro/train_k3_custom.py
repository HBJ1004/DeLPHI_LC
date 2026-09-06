"""Train a K3 scorer from user-supplied labelled JSONL examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from lc_pipeline.k3.config import K3ScoreModelConfig, K3TrainingConfig
from lc_pipeline.k3.model import CandidateConditionedScorer
from lc_pipeline.k3.tokenizer import tokenize_epochs
from lc_pipeline.k3.training import K3TrainingExample, fit_k3
from lc_pipeline.v2.preprocessing import KnownPeriod, Observation, ObservationEpoch


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _example(row: dict[str, Any], path: Path, line_number: int) -> K3TrainingExample:
    """Validate one documented JSONL row and construct tokenized model input."""
    try:
        if row["schema"] != "delphi.k3-training-example.v1":
            raise ValueError("schema must be delphi.k3-training-example.v1")
        period = KnownPeriod(**row["known_period"])
        epochs = tuple(
            ObservationEpoch(
                epoch["epoch_id"], tuple(Observation(**observation) for observation in epoch["observations"])
            )
            for epoch in row["epochs"]
        )
        return K3TrainingExample(
            object_id=str(row["object_id"]), tokenized=tokenize_epochs(epochs, known_period=period),
            target_axes=np.asarray(row["target_axes"], dtype=np.float32),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid training example: {exc}") from exc


def read_examples(path: Path) -> tuple[K3TrainingExample, ...]:
    """Read nonempty JSONL with unique object IDs and fail at the source row."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    rows = []
    for line_number, text in enumerate(lines, start=1):
        if text.strip():
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: every JSONL row must be an object")
            rows.append(_example(row, path, line_number))
    if not rows:
        raise ValueError(f"{path}: no training examples")
    identifiers = [row.object_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path}: object_id values must be unique")
    return tuple(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 137, 777, 2027), default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_directory.exists():
        parser.error("output directory already exists; choose a new directory")
    train, validation = read_examples(args.train), read_examples(args.validation)
    overlap = {example.object_id for example in train}.intersection(example.object_id for example in validation)
    if overlap:
        parser.error("train and validation object IDs overlap: " + ", ".join(sorted(overlap)[:5]))
    args.output_directory.mkdir(parents=True)
    configuration = K3TrainingConfig(seed=args.seed)
    model = CandidateConditionedScorer(K3ScoreModelConfig())
    result = fit_k3(model, train, validation, config=configuration, stage="custom", device=args.device,
                    checkpoint_path=args.output_directory / "custom-model.pt")
    report = {
        "schema": "delphi.k3-custom-training.v1",
        "scope": "user-supplied experimental training; not the frozen DAMIT benchmark",
        "train_examples": len(train), "validation_examples": len(validation),
        "train_sha256": _sha256(args.train), "validation_sha256": _sha256(args.validation),
        "model_config": model.config.as_mapping(), "training_config": configuration.as_mapping(),
        "result": {**result.__dict__, "history": list(result.history)},
    }
    (args.output_directory / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
