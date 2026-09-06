"""Export the 25 hash-bound local training checkpoints as five safe ensembles.

Only use this exporter with the trusted, frozen local training archive. Public
inference loads safetensors and never unpickles a downloaded checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from lc_pipeline.k3.bundle import SEEDS, K3EnsemblePredictor, sha256
from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
from lc_pipeline.k3.tokenizer import K3_TOKENIZER_SCHEMA_SHA256


def export(root: Path, splits: Path, output: Path, repository: Path) -> dict:
    """Verify source hashes, preserve every tensor, and package evaluated folds."""
    if output.exists():
        raise ValueError("output must be a new directory")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository):
        raise ValueError("commit release code before exporting weights")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    summary = json.loads((root / "release/publication-summary.json").read_text())
    if summary["protocol_sha256"] != K3_PROTOCOL_SHA256:
        raise ValueError("source protocol mismatch")
    if sha256(splits) != summary["downstream"]["splits_sha256"]:
        raise ValueError("source split mismatch")
    ensemble = root / "evaluations/real-oof-ensemble.npz"
    calibrated = root / "evaluations/real-oof-ensemble-calibrated.npz"
    for path, key in ((ensemble, "oof_ensemble"), (calibrated, "calibrated_ensemble")):
        if sha256(path) != summary["source_sha256"][key]:
            raise ValueError(f"source checksum mismatch: {path.name}")
    with np.load(ensemble, allow_pickle=False) as data:
        hashes = dict(zip(data["checkpoint_names"].astype(str),
                          data["checkpoint_sha256"].astype(str), strict=True))
    expected = {f"real-fold-{fold}-seed-{seed}.pt" for fold in range(5) for seed in SEEDS}
    if set(hashes) != expected:
        raise ValueError("source does not bind exactly 25 evaluated models")
    with np.load(calibrated, allow_pickle=False) as data:
        cone90, cone95 = data["fold_cone90_deg"].copy(), data["fold_cone95_deg"].copy()
    fold_rows = json.loads(splits.read_text())["folds"]
    output.mkdir(parents=True)
    assets = {}
    for fold in range(5):
        directory = output / f"k3-oof-fold-{fold}"
        directory.mkdir()
        members, configurations = [], []
        for seed in SEEDS:
            source = root / "models" / f"real-fold-{fold}-seed-{seed}.pt"
            if sha256(source) != hashes[source.name]:
                raise ValueError(f"checkpoint checksum mismatch: {source.name}")
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            provenance = checkpoint["run_provenance"]
            if (checkpoint["stage"] != "real-oof" or checkpoint["config"]["seed"] != seed
                    or provenance["implementation_commit"] != summary["implementation_commit"]
                    or provenance["protocol_sha256"] != K3_PROTOCOL_SHA256):
                raise ValueError("checkpoint provenance mismatch")
            tensors = {key: value.detach().cpu().contiguous()
                       for key, value in checkpoint["model_state_dict"].items()}
            destination = directory / f"seed-{seed}.safetensors"
            save_file(tensors, str(destination))
            restored = load_file(str(destination))
            if set(restored) != set(tensors) or any(
                    not torch.equal(value, restored[key]) for key, value in tensors.items()):
                raise ValueError("export changed a tensor")
            configurations.append(checkpoint["model_config"])
            members.append({"seed": seed, "file": destination.name,
                            "sha256": sha256(destination),
                            "source_checkpoint_sha256": hashes[source.name]})
        if any(config != configurations[0] for config in configurations):
            raise ValueError("ensemble model configurations differ")
        row = next(row for row in fold_rows if row["fold"] == fold)
        manifest = {
            "schema": "delphi.k3-ensemble-bundle.v1", "bundle_id": directory.name,
            "fold": fold, "protocol_sha256": K3_PROTOCOL_SHA256,
            "tokenizer_sha256": K3_TOKENIZER_SCHEMA_SHA256,
            "training_commit": summary["implementation_commit"], "export_commit": commit,
            "splits_sha256": sha256(splits), "model_config": configurations[0],
            "object_roles": {key: row[key] for key in
                             ("train_ids", "validation_ids", "calibration_ids", "test_ids")},
            "calibration": {"cone90_deg": float(cone90[fold]), "cone95_deg": float(cone95[fold]),
                            "source_sha256": sha256(calibrated)},
            "members": members,
        }
        (directory / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        K3EnsemblePredictor(directory)
        archive = output / (directory.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(directory, arcname=directory.name)
        assets[archive.name] = sha256(archive)
    (output / "SHA256SUMS").write_text("".join(f"{value}  {key}\n" for key, value in assets.items()))
    return {"export_commit": commit, "tensor_parity": "exact", "assets": assets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "splits", "output", "repository"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.artifact_root, args.splits, args.output, args.repository), indent=2))


if __name__ == "__main__":
    main()
