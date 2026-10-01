"""Exact group Shapley values of the DeLPHI inputs.

Eight input groups are either present or removed from the trained networks.
For every one of the 2^8 = 256 combinations we score a grid of trial axes
(by default the HEALPix grid with 1,536 axes, about 3.7 degrees apart, which
is four times coarser than the 6,144-axis grid of DeLPHI, to keep the run
time practical) and record two quantities for each test asteroid:

  percentile  the fraction of trial axes that score below the best reference
              axis (0.5 when the score map carries no information);
  grid_error  the oracle error of the three peaks of the score map, read with
              the locked peak rule and without the final adjustment.

The Shapley value of a group is its marginal effect averaged over all orders
in which the groups can be added.  The values of the eight groups add up to
the difference between the full input and the fully removed input.

Removal follows repro/interpretation_ablations.py.  Photometric quantities are
set to zero.  An axis-dependent term is replaced by its average over all axis
directions (1/3 for the squared terms, 0 for their product).  The phase angle
and the two distances are replaced by their mean over the occupied bins of the
test asteroids of the same cross-validation run.  The networks are not
retrained.  The reported photometric uncertainties are identically zero for
the DAMIT lightcurves and are not treated as a group.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_pipeline.k3 import publication_run as run  # noqa: E402
from lc_pipeline.k3.config import K3ScoreModelConfig  # noqa: E402
from lc_pipeline.k3.grid import axial_angular_error_deg, axial_healpix_grid  # noqa: E402
from lc_pipeline.k3.model import CandidateConditionedScorer  # noqa: E402
from lc_pipeline.k3.tokenizer import K3_EPOCH_FEATURE_NAMES, K3_PHASE_FEATURE_NAMES  # noqa: E402
from lc_pipeline.k3.training import collate_examples  # noqa: E402

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parents[1]
ARTIFACTS = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092"
CATALOG = HERE / "data/damit-20250610T000301Z/catalog.jsonl"
SPLITS = HERE / "data/damit-20250610T000301Z/publication-splits-v2.3.json"
DUMP = FOLLOWUP.parent / "damit-20250610T000301Z"
SEEDS = (17, 42, 137, 777, 2027)
INPUT_KEYS = ("phase_features", "phase_mask", "geometry_features", "epoch_features", "epoch_mask")

GROUPS = (
    "binned_brightness",
    "lightcurve_shape_summaries",
    "sampling_summaries",
    "period",
    "sun_term",
    "observer_term",
    "product_term",
    "phase_angle_and_distances",
)
FOURIER = tuple(f"fourier_h{h}_{p}" for h in range(1, 9) for p in ("cos", "sin"))
PHOTOMETRIC = {
    0: (("centered_log_flux_mean", "centered_log_flux_scatter"), ()),
    1: ((), FOURIER + ("log_flux_peak_to_peak",)),
    2: (
        ("log1p_observation_count", "within_bin_phase_span_fraction"),
        ("occupied_phase_fraction", "log1p_observation_count", "log1p_duration_days"),
    ),
    3: ((), ("normalized_log_period_hours",)),
}
# group -> invariant channels it controls; channel order is (p.s)^2, (p.o)^2,
# (p.s)(p.o), s.o, phase angle / pi, log1p(sun distance), log1p(observer distance)
GEOMETRIC = {4: (0,), 5: (1,), 6: (2,), 7: (3, 4, 5, 6)}
AXIS_AVERAGES = {0: 1.0 / 3.0, 1: 1.0 / 3.0, 2: 0.0}

_ORIGINAL_DESCRIPTOR = CandidateConditionedScorer.__dict__["_candidate_invariants"]
_ORIGINAL_INVARIANTS = _ORIGINAL_DESCRIPTOR.__func__


class FastModes:
    """Vectorised copy of lc_pipeline.k3.grid.extract_axial_modes (K=3, 15 deg).

    It selects the same peaks as the locked function; this was checked on the
    170 saved score maps and 301 random maps, including tied and flat maps.
    """

    def __init__(self, grid, separation_deg=15.0, count=3):
        self.vectors = np.asarray(grid.vectors, dtype=np.float64)
        n = len(self.vectors)
        nb = np.asarray(grid.neighbour_indices)
        rows = []
        for i in range(n):
            u = np.unique(nb[i][nb[i] >= 0])
            u = u[u != i]
            rows.append(u)
        width = max(len(r) for r in rows)
        self.nb = np.full((n, width), -1, dtype=np.int64)
        for i, r in enumerate(rows):
            self.nb[i, : len(r)] = r
        self.valid = self.nb >= 0
        self.cos_limit = np.cos(np.radians(separation_deg))
        self.count = count
        self.index = np.arange(n)

    def __call__(self, scores):
        s = np.asarray(scores, dtype=np.float64)
        ns = np.where(self.valid, s[np.where(self.valid, self.nb, 0)], -np.inf)
        ge = np.all(s[:, None] >= ns, axis=1)
        tie_lower = np.any(
            self.valid & (ns == s[:, None]) & (self.nb < self.index[:, None]), axis=1
        )
        maxima = np.flatnonzero(ge & ~tie_lower)
        maxima = maxima[np.lexsort((maxima, -s[maxima]))]
        chosen = []

        def consider(i):
            if i in chosen:
                return
            v = self.vectors[i]
            # axial angle >= separation  <=>  |cos| <= cos(separation), with the same
            # rounding as axial_angular_error_deg checked below
            for o in chosen:
                if not self._far(v, self.vectors[o]):
                    return
            chosen.append(i)

        for i in maxima:
            consider(int(i))
            if len(chosen) == self.count:
                break
        if len(chosen) < self.count:
            for i in np.lexsort((self.index, -s)):
                consider(int(i))
                if len(chosen) == self.count:
                    break
        return np.array(chosen)

    def _far(self, a, b):
        return float(axial_angular_error_deg(a, b)) >= 15.0


def remove_photometry(inputs, present):
    output = {key: value.clone() for key, value in inputs.items()}
    for group, (phase_names, epoch_names) in PHOTOMETRIC.items():
        if group in present:
            continue
        if phase_names:
            output["phase_features"][
                ..., [K3_PHASE_FEATURE_NAMES.index(n) for n in phase_names]
            ] = 0.0
        if epoch_names:
            output["epoch_features"][
                ..., [K3_EPOCH_FEATURE_NAMES.index(n) for n in epoch_names]
            ] = 0.0
    return output


def set_geometry(present, shared_means):
    """Patch the invariant terms so that removed groups carry no information."""
    constants = {}
    for group, channels in GEOMETRIC.items():
        if group in present:
            continue
        for channel in channels:
            constants[channel] = AXIS_AVERAGES.get(channel, shared_means.get(channel))
    if not constants:
        CandidateConditionedScorer._candidate_invariants = _ORIGINAL_DESCRIPTOR
        return

    def invariants(geometry_features, candidates):
        values = _ORIGINAL_INVARIANTS(geometry_features, candidates).clone()
        for channel, constant in constants.items():
            values[..., channel] = constant
        return values

    CandidateConditionedScorer._candidate_invariants = staticmethod(invariants)


def shared_channel_means(batches, device):
    """Mean of the axis-independent terms over occupied bins of one fold."""
    totals, count = np.zeros(4), 0.0
    probe = torch.tensor([[[0.0, 0.0, 1.0]]], device=device)
    for inputs in batches:
        geometry = inputs["geometry_features"].to(device)
        mask = inputs["phase_mask"].to(device)
        values = _ORIGINAL_INVARIANTS(geometry, probe.expand(geometry.shape[0], -1, -1))[..., 0, 3:]
        weights = mask[..., None].to(values.dtype)
        totals += (values * weights).sum(dim=(0, 1, 2)).cpu().numpy()
        count += float(weights.sum().cpu())
    return {3 + index: float(totals[index] / count) for index in range(4)}


def axial_error(axes, targets):
    axes = np.asarray(axes, dtype=float)
    targets = np.asarray(targets, dtype=float)
    cosine = np.abs(axes @ targets.T)
    return float(np.degrees(np.arccos(np.clip(cosine.max(), -1.0, 1.0))))


def coalition_values(model, examples, device, shared_means, grid_tensor, fast_modes, batch_size=4):
    """Return percentile and grid error, arrays [asteroid, 256]."""
    count = len(examples)
    percentile = np.empty((count, 256))
    grid_error = np.empty((count, 256))
    for start in range(0, count, batch_size):
        chunk = examples[start : start + batch_size]
        raw = collate_examples(chunk)
        base = {key: raw[key] for key in INPUT_KEYS}
        targets = [np.asarray(example.target_axes, dtype=float) for example in chunk]
        width = max(len(t) for t in targets)
        padded = np.zeros((len(chunk), width, 3))
        for index, t in enumerate(targets):
            padded[index, : len(t)] = t
            padded[index, len(t) :] = t[0]
        target_tensor = torch.as_tensor(padded, dtype=base["phase_features"].dtype, device=device)
        for photometric_mask in range(16):
            present_photometry = {g for g in range(4) if photometric_mask >> g & 1}
            inputs = {
                k: v.to(device) for k, v in remove_photometry(base, present_photometry).items()
            }
            with torch.no_grad():
                interactions, latent = model._encode_photometry(
                    inputs["phase_features"],
                    inputs["phase_mask"],
                    inputs["epoch_features"],
                    inputs["epoch_mask"],
                )
                for geometric_mask in range(16):
                    present = present_photometry | {
                        4 + g for g in range(4) if geometric_mask >> g & 1
                    }
                    coalition = photometric_mask | (geometric_mask << 4)
                    set_geometry(present, shared_means)
                    try:
                        scores = []
                        for offset in range(0, len(grid_tensor), 256):
                            candidates = (
                                grid_tensor[offset : offset + 256]
                                .unsqueeze(0)
                                .expand(len(chunk), -1, -1)
                            )
                            scores.append(
                                model.score_encoded(
                                    interactions,
                                    latent,
                                    inputs["phase_mask"],
                                    inputs["geometry_features"],
                                    inputs["epoch_mask"],
                                    candidates,
                                ).scores
                            )
                        reference = model.score_encoded(
                            interactions,
                            latent,
                            inputs["phase_mask"],
                            inputs["geometry_features"],
                            inputs["epoch_mask"],
                            target_tensor,
                        ).scores
                    finally:
                        set_geometry(set(range(8)), shared_means)
                    grid_scores = torch.cat(scores, dim=1).double().cpu().numpy()
                    reference = reference.double().cpu().numpy()
                    for index in range(len(chunk)):
                        row = grid_scores[index]
                        best = reference[index, : len(targets[index])].max()
                        percentile[start + index, coalition] = np.mean(row < best) + 0.5 * np.mean(
                            row == best
                        )
                        axes = fast_modes.vectors[fast_modes(row)]
                        grid_error[start + index, coalition] = axial_error(axes, targets[index])
    return percentile, grid_error


def shapley(values):
    """Exact Shapley values from v(S) indexed by bitmask, last axis of length 256."""
    n = 8
    weights = {
        size: math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
        for size in range(n)
    }
    result = np.zeros(values.shape[:-1] + (n,))
    for group in range(n):
        bit = 1 << group
        for mask in range(256):
            if mask & bit:
                continue
            size = bin(mask).count("1")
            result[..., group] += weights[size] * (values[..., mask | bit] - values[..., mask])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="one fold and one seed only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--nside", type=int, default=16, help="HEALPix resolution of the scored grid"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLITS.read_text())
    grid = axial_healpix_grid(args.nside)
    fast_modes = FastModes(grid)
    grid_tensor = torch.as_tensor(
        np.array(grid.vectors, copy=True), dtype=torch.float32, device=args.device
    )
    folds = split["folds"][:1] if args.pilot else split["folds"]
    seeds = SEEDS[:1] if args.pilot else SEEDS
    cache = {}
    # Seeds on the outside, so that one full pass over all 170 asteroids
    # (one network per fold) is complete before the next seed starts.
    for seed in seeds:
        for fold_row in folds:
            fold = int(fold_row["fold"])
            output = args.output_dir / f"fold-{fold}-seed-{seed}.npz"
            if output.exists():
                continue
            if fold not in cache:
                examples = run.load_real_examples(CATALOG, DUMP, tuple(fold_row["test_ids"]))
                batches = [
                    {
                        k: v
                        for k, v in collate_examples(examples[i : i + 4]).items()
                        if k in INPUT_KEYS
                    }
                    for i in range(0, len(examples), 4)
                ]
                cache[fold] = (examples, shared_channel_means(batches, args.device))
            examples, shared_means = cache[fold]
            checkpoint = torch.load(
                ARTIFACTS / f"models/real-fold-{fold}-seed-{seed}.pt",
                map_location="cpu",
                weights_only=False,
            )
            model = CandidateConditionedScorer(K3ScoreModelConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(args.device).eval()
            started = time.time()
            percentile, grid_error = coalition_values(
                model, examples, args.device, shared_means, grid_tensor, fast_modes
            )
            np.savez_compressed(
                output,
                object_ids=np.array([example.object_id for example in examples]),
                groups=np.array(GROUPS),
                nside=np.array(args.nside),
                percentile=percentile,
                grid_error=grid_error,
                shapley_percentile=shapley(percentile),
                shapley_grid_error=shapley(grid_error),
                shared_means=np.array([shared_means[c] for c in (3, 4, 5, 6)]),
            )
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "seed": seed,
                        "seconds": round(time.time() - started, 1),
                        "full_percentile": float(percentile[:, 255].mean()),
                        "empty_percentile": float(percentile[:, 0].mean()),
                        "full_grid_error": float(grid_error[:, 255].mean()),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
