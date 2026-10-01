"""Finer inference-only input removals for the interpretation subsection.

Each control removes one physically defined part of the input from every
trained K3 network (5 folds x 5 seeds), without retraining, and records the
oracle error of the test asteroids.  The published controls in
``lc_pipeline.k3.publication_run`` are left untouched; this driver reuses
their evaluation routine with additional transforms.

Photometric controls zero input features, as the published controls do.
Geometric controls replace one of the axis-dependent terms that the scorer
computes for a trial axis p, (p.s)^2, (p.e)^2 or (p.s)(p.e), by its average
over the sphere (1/3, 1/3 and 0), so that the term no longer depends on p.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_pipeline.k3 import publication_run as run
from lc_pipeline.k3.model import CandidateConditionedScorer
from lc_pipeline.k3.tokenizer import K3_EPOCH_FEATURE_NAMES, K3_PHASE_FEATURE_NAMES

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parents[1]
ARTIFACTS = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092"
CATALOG = HERE / "data/damit-20250610T000301Z/catalog.jsonl"
SPLITS = HERE / "data/damit-20250610T000301Z/publication-splits-v2.3.json"
DUMP = FOLLOWUP.parent / "damit-20250610T000301Z"
SEEDS = (17, 42, 137, 777, 2027)

EVEN = tuple(f"fourier_h{h}_{p}" for h in (2, 4, 6, 8) for p in ("cos", "sin"))
ODD = tuple(f"fourier_h{h}_{p}" for h in (1, 3, 5, 7) for p in ("cos", "sin"))
PHOTOMETRIC = {
    "amplitude": ((), ("log_flux_peak_to_peak",)),
    "fourier-even": ((), EVEN),
    "fourier-odd": ((), ODD),
    "fourier-all": ((), EVEN + ODD),
    "binned-brightness": (("centered_log_flux_mean", "centered_log_flux_scatter"), ()),
    "photometric-errors": (
        ("mean_relative_photometric_error", "measured_error_fraction"),
        ("measured_error_fraction",),
    ),
}
# invariant channel -> constant; 0 (p.s)^2, 1 (p.e)^2, 2 (p.s)(p.e)
GEOMETRIC = {
    "observer-aspect-only": {0: 1.0 / 3.0, 2: 0.0},
    "solar-aspect-only": {1: 1.0 / 3.0, 2: 0.0},
    "no-cross-term": {2: 0.0},
}
CONTROLS = tuple(PHOTOMETRIC) + tuple(GEOMETRIC)

_ORIGINAL_DESCRIPTOR = CandidateConditionedScorer.__dict__["_candidate_invariants"]
_ORIGINAL_INVARIANTS = _ORIGINAL_DESCRIPTOR.__func__


def photometric_transform(control):
    phase_names, epoch_names = PHOTOMETRIC[control]
    phase_idx = [K3_PHASE_FEATURE_NAMES.index(n) for n in phase_names]
    epoch_idx = [K3_EPOCH_FEATURE_NAMES.index(n) for n in epoch_names]

    def transform(inputs):
        output = {key: value.clone() for key, value in inputs.items()}
        if phase_idx:
            output["phase_features"][..., phase_idx] = 0.0
        if epoch_idx:
            output["epoch_features"][..., epoch_idx] = 0.0
        return output

    return transform


def set_geometric(control):
    if control is None:
        CandidateConditionedScorer._candidate_invariants = _ORIGINAL_DESCRIPTOR
        return
    constants = GEOMETRIC[control]

    def invariants(geometry_features, candidates):
        values = _ORIGINAL_INVARIANTS(geometry_features, candidates).clone()
        for channel, constant in constants.items():
            values[..., channel] = constant
        return values

    CandidateConditionedScorer._candidate_invariants = staticmethod(invariants)


def evaluate(control, output_dir, device):
    split = json.loads(SPLITS.read_text())
    rows = []
    for fold_row in split["folds"]:
        fold = int(fold_row["fold"])
        examples = run.load_real_examples(CATALOG, DUMP, tuple(fold_row["test_ids"]))
        for seed in SEEDS:
            checkpoint = ARTIFACTS / f"models/real-fold-{fold}-seed-{seed}.pt"
            output = output_dir / f"real-fold-{fold}-seed-{seed}-{control}.npz"
            if output.exists():
                errors = np.load(output)["oracle_errors_deg"]
            else:
                transform = None
                if control in PHOTOMETRIC:
                    transform = photometric_transform(control)
                set_geometric(control if control in GEOMETRIC else None)
                try:
                    run._evaluate_examples(
                        checkpoint, examples, output, device=device, batch_size=4,
                        input_transform=transform,
                    )
                finally:
                    set_geometric(None)
                errors = np.load(output)["oracle_errors_deg"]
            base = np.load(ARTIFACTS / f"evaluations/real-fold-{fold}-seed-{seed}.npz")
            rows.append({"fold": fold, "seed": seed, "control": control,
                         "mean_error": float(errors.mean()),
                         "baseline_mean_error": float(base["oracle_errors_deg"].mean())})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controls", nargs="+", default=list(CONTROLS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for control in args.controls:
        rows = evaluate(control, args.output_dir, args.device)
        (args.output_dir / f"summary-{control}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
