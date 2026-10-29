"""Single-network evaluations of the 170 test asteroids at a chosen period.

    python -m repro.own_period.network_controls --periods damit --out-dir D --folds 0
    python -m repro.own_period.network_controls --periods own-periods-170.json --out-dir O

For every fold, seed and control this writes ``real-fold-{f}-seed-{s}[-{control}].npz``
with the published layout (object_ids, score_grids, oracle_errors_deg).
The evaluation is ``publication_run._evaluate_examples`` with batch size 4,
exactly as published; only the period used to fold each lightcurve changes.

Controls
  raw            no change (the per-network result, baseline of every gap)
  input-swap     lightcurve exchange: each target receives the tokenized input of the
                 donor given by the published derangement (seed K3_INPUT_SWAP_SEED + fold);
                 the donor is folded with the donor's own period
  zero-numeric, brightness, geometry, period-scalar, sampling-summary
                 the published diagnostic removals (publication_run.ablate_k3_inputs)
  amplitude, fourier-even, fourier-odd, fourier-all, binned-brightness,
  photometric-errors, observer-aspect-only, solar-aspect-only, no-cross-term
                 the finer removals of repro/interpretation_ablations.py

No reference pole is used before the oracle error is computed inside the
published evaluation routine, as in the published runs.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from lc_pipeline.k3 import publication_run as run
from lc_pipeline.k3.training import K3TrainingExample, deterministic_derangement_indices
from repro import interpretation_ablations as finer

from .nettests_common import MODELS, SEEDS, fold_rows, load_examples, period_override

DIAGNOSTIC = run.K3_DIAGNOSTIC_CONTROLS
FINER = finer.CONTROLS
CONTROLS = ("raw", "input-swap") + DIAGNOSTIC + FINER


def output_name(fold: int, seed: int, control: str) -> str:
    suffix = "" if control == "raw" else f"-{control}"
    return f"real-fold-{fold}-seed-{seed}{suffix}.npz"


def evaluate(control, fold, seed, examples, test_ids, destination, device):
    checkpoint = MODELS / f"real-fold-{fold}-seed-{seed}.pt"
    kwargs = {"device": device, "batch_size": 4}
    if control == "raw":
        return run._evaluate_examples(checkpoint, examples, destination, **kwargs)
    if control == "input-swap":
        control_seed = run.K3_INPUT_SWAP_SEED + fold
        donors = deterministic_derangement_indices(len(examples), seed=control_seed)
        swapped = tuple(
            K3TrainingExample(
                object_id=target.object_id,
                tokenized=examples[int(donor)].tokenized,
                target_axes=target.target_axes,
            )
            for target, donor in zip(examples, donors, strict=True)
        )
        return run._evaluate_examples(
            checkpoint,
            swapped,
            destination,
            artifact_arrays={
                "input_donor_ids": np.asarray([test_ids[int(i)] for i in donors]),
                "derangement_seeds": np.full(len(swapped), control_seed, dtype=np.int64),
            },
            **kwargs,
        )
    if control in DIAGNOSTIC:
        return run._evaluate_examples(
            checkpoint,
            examples,
            destination,
            artifact_arrays={"diagnostic_control": np.full(len(examples), control)},
            input_transform=lambda inputs: run.ablate_k3_inputs(inputs, control),
            **kwargs,
        )
    transform = finer.photometric_transform(control) if control in finer.PHOTOMETRIC else None
    finer.set_geometric(control if control in finer.GEOMETRIC else None)
    try:
        return run._evaluate_examples(
            checkpoint, examples, destination, input_transform=transform, **kwargs
        )
    finally:
        finer.set_geometric(None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", required=True)
    parser.add_argument("--out-dir", type=lambda s: __import__("pathlib").Path(s), required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--controls", nargs="+", default=list(CONTROLS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    unknown = set(args.controls) - set(CONTROLS)
    if unknown:
        raise SystemExit(f"unknown controls {sorted(unknown)}")
    periods, provenance, label = period_override(args.periods)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.out_dir / "period-source.json"
    if stamp.exists():
        if json.loads(stamp.read_text())["period_source"] != label:
            raise SystemExit("output directory belongs to another period source")
    else:
        with stamp.open("x") as stream:
            json.dump({"period_source": label, "provenance": provenance, "spec": args.periods}, stream)
    for row in fold_rows():
        fold = int(row["fold"])
        if fold not in args.folds:
            continue
        test_ids = tuple(row["test_ids"])
        examples = load_examples(test_ids, periods, provenance)
        for seed in args.seeds:
            for control in args.controls:
                destination = args.out_dir / output_name(fold, seed, control)
                if destination.exists():
                    continue
                started = time.time()
                summary = evaluate(control, fold, seed, examples, test_ids, destination, args.device)
                print(
                    json.dumps(
                        {
                            "fold": fold,
                            "seed": seed,
                            "control": control,
                            "mean_error_deg": round(float(summary["mean_error_deg"]), 6),
                            "seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
