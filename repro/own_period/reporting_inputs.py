"""Reporting consumers of the RT3 runners, using the central network switch.

Only network-derived paths move. Catalogs, saved searches, classical references
and historical drawing source remain explicit read-only inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from repro import network_root as network
from repro.own_period import DataPath as FixedDataPath


def substitutions():
    """Longest-prefix translation to the outputs actually made by RT3a--d1."""
    own = Path(FixedDataPath())
    followup = own.parent.parent
    results = Path(network.NetworkPath("results"))
    pairs = {
        own / "median-stats.json": results / "median-stats.json",
        own / "paired-gains.json": results / "paired-gains.json",
        own / "v67/v7-validation-guided-gpu/validation-vs-published.json": results
        / "reporting/v7-validation.json",
        own / "maps/own-period": results / "headline/maps/own",
        own / "maps/damit-period": results / "headline/maps/damit",
        own / "own-period-scores-main.json": results / "headline/own-period-scores-main.json",
        own / "own-period-scores.json": results / "ztf/own-period-scores.json",
        own / "benchmark": results / "delphi-inversions/own",
        own / "refine-chain": results / "refine-chain",
        own / "v67/v6-own-period-score.json": results / "delphi-inversions/v6-own-score.json",
        own / "v67/v7-own-period-score.json": results / "delphi-inversions/v7-own-score.json",
        own / "v67/v7-own-period": results / "delphi-inversions/v7-own",
        own / "lowq/scores-own": results / "lower-quality-census/scores-own",
        own / "lowq/scores-published-check": results / "lower-quality-census/scores-reference",
        own / "failed-excluded-derived": results / "derived-comparators",
        own / "nettests/no-uncertainty": results / "external-variants/no-uncertainty",
        own / "nettests/ztf-variants": results / "external-variants/ztf-variants",
        own / "nettests/shapley-own-summary.json": results / "shapley/own-summary.json",
        own / "nettests/shapley-own": results / "shapley/own",
        own / "nettests/train-vs-test-own-report.json": results
        / "internal-validation/own-report.json",
        own / "nettests/train-vs-test-own": results / "internal-validation/own",
        own / "nettests/controls-own-summary.json": results / "input-removals/own-summary.json",
        own / "nettests/controls-own": results / "input-removals/own",
        own / "reliability/score": results / "reliability/score",
        own / "reliability/RESULTS.md": results / "reliability/RESULTS.md",
        own / "spacecraft": results / "spacecraft",
        own / "census7": results / "census7",
        own / "temporal": results / "temporal",
        followup / "data/direct-workflow-benchmark": results / "delphi-inversions/damit",
        followup / "data/reliability-20260915/run-1/scored-rows.json": results
        / "period-sensitivity/run-1/scored-rows.json",
        followup / "data/reliability-20260915/run-1/reliability-report.csv": results
        / "period-sensitivity/run-1/sampling-table.csv",
        followup / "data/interpretation-20260926/mirror-symmetry.json": results
        / "headline/maps/damit/analyses/mirror-symmetry.json",
        followup / "data/interpretation-20260930/label-shuffle-spread.json": results
        / "controls-label-shuffle/label-shuffle-spread.json",
        followup / "data/publication-revision-20260917-r2/lowq-analysis.json": results
        / "mirror/lowq-analysis.json",
        followup / "experiments/period-refinement-20260921/end-to-end/benchmark-v8": results
        / "period-peak-pilot/benchmark",
        followup / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations": Path(
            network.NetworkPath("evaluations")
        ),
    }
    # The raw evaluations for input-swap are identical to the raw removal cells;
    # the swap itself is produced by the separate control family.
    for fold in range(5):
        for seed in network.SEEDS:
            name = f"real-fold-{fold}-seed-{seed}-input-swap.npz"
            pairs[own / "nettests/controls-own" / name] = results / "controls-input-swap/own" / name
    return sorted(pairs.items(), key=lambda pair: len(pair[0].parts), reverse=True)


def select(path):
    path = Path(os.path.abspath(path))
    if network.switched():
        for old, new in substitutions():
            if path.is_relative_to(old):
                return new / path.relative_to(old)
    return path


def dependent(path):
    if not network.switched():
        return False
    selected = select(path)
    return selected.is_relative_to(Path(network.NetworkPath("results"))) or selected.is_relative_to(
        Path(network.NetworkPath("evaluations"))
    )


def check(path):
    """Check dict, row-list and sealed native JSON formats, without rebinding."""
    path = select(path)
    if not dependent(path):
        return
    network.guard_output(path)
    if path.suffix == ".npz":
        network.check_npz(path)
    elif path.suffix in (".json", ".jsonl"):
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            value = json.loads(path.read_text())
            rows = value if isinstance(value, list) else [value]
            if isinstance(value, dict) and "payload" in value:
                from lc_pipeline.k3.reliability_study import unseal

                unseal(path)
                rows = (
                    [value["payload"]] if isinstance(value["payload"], dict) else value["payload"]
                )
        expected = network.json_metadata()["network_binding_sha256"]
        if not rows or any(row.get("network_binding_sha256") != expected for row in rows):
            raise ValueError(f"network binding hash mismatch in reporting input: {path}")
    elif path.suffix == ".csv":
        # Native CSVs have no metadata column. Verify their accompanying bound
        # report before using them; runner receipts also lock their file hashes.
        if path.name == "sampling-table.csv":
            check(path.with_name("report.json"))
        elif "lineage-" in str(path.parent):
            check(
                path.parent.parent
                / ("census-analysis-" + path.parent.name.removeprefix("lineage-") + ".json")
            )


def load_npz(path, *args, **kwargs):
    path = select(path)
    if network.switched():
        followup = Path(FixedDataPath("../.."))
        reference_only = {
            followup / "source/repro/baselines_out/reference-baselines-per-object.npz",
            followup / "inputs/frozen-artifacts/k3-2408c563/comparators/v1-comparators.npz",
            followup / "data/interpretation-20260926/random-axes-cdf.npz",
        }
        if path not in reference_only:
            network.check_npz(path)
    return np.load(path, *args, **kwargs)


def read_json(path):
    path = select(path)
    check(path)
    return json.loads(path.read_text(encoding="utf-8"))


class InputPath(os.PathLike):
    """Keep joins lazy, and validate reads after selecting the RT3 result."""

    def __init__(self, path):
        self.original = path

    def path(self):
        return select(self.original)

    def __fspath__(self):
        return str(self.path())

    def __str__(self):
        return str(self.path())

    def __hash__(self):
        return hash(self.path())

    def __eq__(self, other):
        if not isinstance(other, (Path, InputPath, FixedDataPath)):
            return NotImplemented
        return self.path() == Path(other)

    def __truediv__(self, part):
        return InputPath(self.original / part)

    @property
    def parent(self):
        return InputPath(self.original.parent)

    def with_name(self, name):
        return InputPath(self.original.with_name(name))

    def read_text(self, *args, **kwargs):
        check(self.path())
        return self.path().read_text(*args, **kwargs)

    def open(self, *args, **kwargs):
        check(self.path())
        return self.path().open(*args, **kwargs)

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        return getattr(self.path(), key)


def DataPath(*parts):
    return InputPath(FixedDataPath(*parts))
