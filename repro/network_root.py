"""Lazy network paths and fail-closed provenance checks for local reruns.

DELPHI_DATA still names the published, fixed inputs. DELPHI_NETWORK_ROOT
selects models/ and evaluations/; optional DELPHI_MODEL_BINDING and
DELPHI_BUNDLE_ROOT override model-binding.json and bundles/ in that root.
With all three unset, each caller retains its historical paths and behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from repro.own_period import DataPath

SEEDS = (17, 42, 137, 777, 2027)
LOCAL_SCHEMA = "delphi.k3-local-rerun-models.v1"
VARIABLES = ("DELPHI_NETWORK_ROOT", "DELPHI_MODEL_BINDING", "DELPHI_BUNDLE_ROOT")


def switched() -> bool:
    return any(os.environ.get(key) for key in VARIABLES)


class NetworkPath(os.PathLike[str]):
    """Resolve environment variables on use, including after module import."""

    def __init__(self, role: str, *parts: str, legacy_followup=None, legacy_path=None):
        self.role, self.parts, self.legacy_followup = role, parts, legacy_followup
        self.legacy_path = legacy_path

    def path(self) -> Path:
        override_key = {"binding": "DELPHI_MODEL_BINDING", "bundles": "DELPHI_BUNDLE_ROOT"}.get(self.role)
        if override_key and os.environ.get(override_key):
            return Path(os.environ[override_key]).expanduser().joinpath(*self.parts).absolute()
        followup = self.legacy_followup
        if followup is None:
            followup = DataPath("../..")
        selected = os.environ.get("DELPHI_NETWORK_ROOT")
        if self.role == "results" and not selected and self.legacy_path is not None:
            return Path(self.legacy_path).absolute()
        if selected:
            root = Path(selected).expanduser()
        else:
            root = Path(followup) / "inputs/frozen-artifacts/k3-definitive-7874092"
        if self.role == "binding":
            override = os.environ.get("DELPHI_MODEL_BINDING")
            base = Path(override).expanduser() if override else (
                root / "model-binding.json" if selected
                else Path(followup) / "data/external-models-final.json")
        elif self.role == "bundles":
            override = os.environ.get("DELPHI_BUNDLE_ROOT")
            base = Path(override).expanduser() if override else (
                root / "bundles" if selected else
                Path(followup).parent / "DeLPHI-publication/DeLPHI-k3-release-assets-20260906")
        elif self.role == "results":
            if selected:
                base = root.parent / "results"
            elif self.legacy_path is not None:
                return Path(self.legacy_path).absolute()
            else:
                raise ValueError("results paths require DELPHI_NETWORK_ROOT")
        else:
            base = root if self.role == "artifacts" else root / self.role
        return base.joinpath(*self.parts).absolute()

    def __fspath__(self):
        return str(self.path())

    def __str__(self):
        return str(self.path())

    def __truediv__(self, part):
        return NetworkPath(self.role, *self.parts, os.fspath(part),
                           legacy_followup=self.legacy_followup,
                           legacy_path=(self.legacy_path / part if self.legacy_path is not None else None))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.path(), name)


def sha256(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def guard_output(path) -> None:
    if not switched():
        return
    destination = Path(path).resolve()
    parts = destination.parts
    if ("own-period-20261021" in parts or "frozen-artifacts" in parts
            or "DeLPHI-k3-release-assets-20260906" in parts
            or any(p in parts for p in ("interpretation-20260923", "interpretation-20260926",
                                       "interpretation-20260930", "publication-revision-20260917-r2"))):
        raise ValueError(f"network switch refuses published output folder: {destination}")


def binding() -> dict:
    value = json.loads(NetworkPath("binding").read_text())
    from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256

    expected = {f"real-fold-{fold}-seed-{seed}.pt" for fold in range(5) for seed in SEEDS}
    records = value.get("models", [])
    if (value.get("schema") not in (LOCAL_SCHEMA, "delphi.k3-external-model-manifest.v1")
            or value.get("protocol_sha256") != K3_PROTOCOL_SHA256
            or len(records) != 25 or {r.get("filename") for r in records} != expected):
        raise ValueError("network binding must identify the chosen protocol and all 25 models")
    return value


def check_checkpoint(path) -> None:
    if not switched():
        return
    path = Path(path)
    record = next((r for r in binding()["models"] if r["filename"] == path.name), None)
    if (record is None or sha256(path) != record["sha256"]
            or path.stat().st_size != record["size_bytes"]):
        raise ValueError(f"network binding hash mismatch: {path}")


def network_metadata() -> dict:
    return {"network_binding_sha256": np.asarray(sha256(NetworkPath("binding")))} if switched() else {}


def json_metadata() -> dict:
    return {key: str(value) for key, value in network_metadata().items()}


def evaluation_metadata(count: int) -> dict:
    """The unchanged evaluation writer requires extras aligned with examples."""
    return {key: np.full(count, value) for key, value in json_metadata().items()}


def check_jsonl(path) -> None:
    """Every row, including failures, must belong to the selected networks."""
    if not switched():
        return
    path = Path(path)
    guard_output(path)
    expected = json_metadata()
    seen = set()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = row.get("run_id", row.get("object_id"))
            if identity is None or identity in seen or any(row.get(k) != v for k, v in expected.items()):
                raise ValueError(f"network binding mismatch or duplicate identity: {path}")
            seen.add(identity)


def check_npz(path) -> None:
    """Accept a bound frozen file or a rerun carrying the chosen binding hash."""
    if not switched():
        return
    path = Path(path)
    value = binding()
    with np.load(path, allow_pickle=False) as arrays:
        known = {r["filename"]: r["sha256"] for r in value["models"]}
        if "checkpoint_names" in arrays:
            for name, digest in zip(arrays["checkpoint_names"].astype(str),
                                    arrays["checkpoint_sha256"].astype(str), strict=True):
                if known.get(name) != digest:
                    raise ValueError(f"network binding hash mismatch in {path}: {name}")
        elif "checkpoint_sha256" in arrays:
            digest = arrays["checkpoint_sha256"].item()
            if digest not in known.values():
                raise ValueError(f"network binding hash mismatch in {path}")
        if "network_binding_sha256" in arrays:
            marker = arrays["network_binding_sha256"]
            if not marker.size or not np.all(marker == sha256(NetworkPath("binding"))):
                raise ValueError(f"network binding hash mismatch: {path}")
            return
    root = Path(NetworkPath("artifacts")).resolve()
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"unbound network artifact: {path}") from exc
    hashes = value.get("artifact_sha256", {})
    if (value["schema"] == LOCAL_SCHEMA and relative ==
            "evaluations/synthetic-label-shuffle-test-seed-17.npz"):
        # The training queue evaluates this auxiliary network after headline binding.
        # Bind its native receipt to the very same 31-checkpoint audit, not a new switch.
        from repro.prepare_k3_followup_inputs import local_evaluation_receipt

        audit_path = Path(NetworkPath("binding")).parent / "checkpoint-audit.json"
        if sha256(audit_path) != value.get("checkpoint_audit_sha256"):
            raise ValueError("network binding checkpoint audit hash mismatch")
        audit = json.loads(audit_path.read_text())
        checkpoint = root / "models/synthetic-label-shuffle-seed-17.pt"
        digest = audit["checkpoint_sha256"].get(checkpoint.name)
        receipt = local_evaluation_receipt(root, path, digest)
        if not digest or sha256(checkpoint) != digest or receipt.get("checkpoint_sha256") != digest:
            raise ValueError(f"network binding auxiliary checkpoint mismatch: {path}")
        return
    if not hashes:
        index_path = root / "release/archive-index.json"
        if sha256(index_path) != value.get("archive_index_sha256"):
            raise ValueError("network binding archive-index hash mismatch")
        index = json.loads(index_path.read_text())
        hashes = {r["logical_path"]: r["sha256"] for r in index["files"]}
    if hashes.get(relative) != sha256(path):
        raise ValueError(f"network binding artifact hash mismatch: {path}")


def check_json(path) -> None:
    if not switched():
        return
    path = Path(path)
    guard_output(path)  # Published results are also forbidden as rerun inputs.
    value = json.loads(path.read_text())
    if value.get("network_binding_sha256") != sha256(NetworkPath("binding")):
        raise ValueError(f"network binding hash mismatch: {path}")


def load_models(model_directory, manifest_path, folds, device):
    """Preserve the published loader; local bindings carry no publication claim."""
    from lc_pipeline.k3.ztf_prediction import _load_models

    if not switched() or binding()["schema"] != LOCAL_SCHEMA:
        return _load_models(model_directory, manifest_path, folds, device)
    import torch

    from lc_pipeline.k3.config import K3ScoreModelConfig
    from lc_pipeline.k3.model import CandidateConditionedScorer

    value = binding()
    models, names, records = [], [], []
    for fold in folds:
        for seed in SEEDS:
            name = f"real-fold-{fold}-seed-{seed}.pt"
            path = Path(model_directory) / name
            check_checkpoint(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            record = next(r for r in value["models"] if r["filename"] == name)
            if (checkpoint.get("stage") != "real-oof"
                    or checkpoint.get("run_provenance") != record["run_provenance"]
                    or int(checkpoint.get("config", {}).get("seed", -1)) != seed):
                raise ValueError(f"network checkpoint metadata mismatch: {name}")
            model = CandidateConditionedScorer(K3ScoreModelConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model_state_dict"])
            models.append(model.to(device).eval())
            names.append(name)
            records.append(record)
    return models, names, records
