"""Portable evaluated ensembles using JSON metadata and safetensors weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from ..v2.preprocessing import KnownPeriod, ObservationEpoch
from .config import K3ScoreModelConfig
from .evaluation import ensemble_score_grids, modes_from_score_grid
from .inference import refine_ensemble_axes, score_axial_grid
from .model import CandidateConditionedScorer
from .protocol import K3_PROTOCOL_SHA256, repository_root
from .tokenizer import K3_TOKENIZER_SCHEMA_SHA256, pad_tokenized_objects, tokenize_epochs

SEEDS = (17, 42, 137, 777, 2027)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class K3EnsemblePredictor:
    """Load one evaluated fold ensemble and return three refined axial modes.

    Calibration radii describe the archived fold experiment. They do not
    establish coverage for new populations, nor provide an individual risk.
    """

    def __init__(self, directory: str | Path, *, device: str = "cpu") -> None:
        root = Path(directory).resolve()
        self.manifest = json.loads((root / "bundle.json").read_text())
        data = self.manifest
        if (data.get("schema") != "delphi.k3-ensemble-bundle.v1"
                or data.get("protocol_sha256") != K3_PROTOCOL_SHA256
                or data.get("tokenizer_sha256") != K3_TOKENIZER_SCHEMA_SHA256
                or type(data.get("fold")) is not int or data["fold"] not in range(5)):
            raise ValueError("incompatible ensemble identity or protocol")
        members = data.get("members", [])
        if [member.get("seed") for member in members] != list(SEEDS):
            raise ValueError("bundle must contain the five ordered evaluated seeds")
        roles = data["object_roles"]
        split_path = repository_root() / "repro/data/damit-20250610T000301Z/publication-splits-v2.3.json"
        if data.get("splits_sha256") != sha256(split_path):
            raise ValueError("bundle split checksum mismatch")
        frozen = json.loads(split_path.read_text())["folds"][data["fold"]]
        expected_sizes = {"train_ids": 108, "validation_ids": 14,
                          "calibration_ids": 14, "test_ids": 34}
        seen: set[str] = set()
        for role, size in expected_sizes.items():
            ids = roles[role]
            if ids != frozen[role]:
                raise ValueError("bundle roles differ from the frozen fold")
            if len(ids) != size or len(set(ids)) != size or seen.intersection(ids):
                raise ValueError("fold object roles are inconsistent")
            seen.update(ids)
        radii = data["calibration"]
        if not 0 <= radii["cone90_deg"] <= radii["cone95_deg"] == 90:
            raise ValueError("invalid fold containment radii")
        config = K3ScoreModelConfig(**data["model_config"])
        self.device = torch.device(device)
        self.models = []
        for member in members:
            path = (root / member["file"]).resolve()
            if path.parent != root or sha256(path) != member["sha256"]:
                raise ValueError("weight file path or checksum mismatch")
            model = CandidateConditionedScorer(config)
            model.load_state_dict(load_file(str(path)), strict=True)
            self.models.append(model.to(self.device).eval())

    def predict(self, epochs: Sequence[ObservationEpoch], *, known_period: KnownPeriod,
                object_id: str) -> dict:
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("object_id is required")
        roles = self.manifest["object_roles"]
        if any(object_id in roles[role] for role in
               ("train_ids", "validation_ids", "calibration_ids")):
            raise ValueError("known benchmark object requires its held-out fold bundle")
        tokenized = tokenize_epochs(epochs, known_period=known_period)
        arrays = pad_tokenized_objects((tokenized,))
        inputs = {key: torch.from_numpy(arrays[key]).to(self.device) for key in
                  ("phase_features", "phase_mask", "geometry_features", "epoch_features", "epoch_mask")}
        # Match the chunk size used to generate the archived deployed ensemble.
        maps = [score_axial_grid(model, inputs, chunk_size=1024) for model in self.models]
        mean = ensemble_score_grids(np.stack(maps)[:, None, :])[0]
        modes = modes_from_score_grid(mean)
        axes, scores = refine_ensemble_axes(
            self.models, inputs, np.asarray([mode.axis_xyz for mode in modes]))
        return {
            "schema": "delphi.k3-ensemble-prediction.v1", "status": "ok",
            "object_id": object_id, "fold": self.manifest["fold"],
            "bundle_id": self.manifest["bundle_id"],
            "tokenizer_sha256": K3_TOKENIZER_SCHEMA_SHA256,
            "period_provenance": known_period.provenance,
            "axes": [{"axis_xyz": axis.tolist(), "score": float(score),
                      "grid_index": int(mode.grid_index)}
                     for axis, score, mode in zip(axes, scores, modes, strict=True)],
            "calibration": {**self.manifest["calibration"],
                            "scope": "archived DAMIT fold; transfer coverage is not established"},
            "risk_deg": None,
        }
