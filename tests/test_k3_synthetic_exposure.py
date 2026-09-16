import hashlib
import io
import json
import tarfile

import pytest

from repro.audit_k3_synthetic_exposure import audit


def _inputs(tmp_path, *, metadata_name="synthetic-data/train/train-0.npz.manifest.json"):
    archive = tmp_path / "synthetic.tar.gz"
    metadata = json.dumps({
        "schema": "delphi.k3-synthetic-shard.v1",
        "shape_donor_ids": ["asteroid_101/model_100"],
        "geometry_donor_ids": ["asteroid_101"],
    }).encode()
    with tarfile.open(archive, "w:gz") as target:
        for name, payload in ((metadata_name, metadata), ("synthetic-data/train/train-0.npz", b"private-array-payload")):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            target.addfile(member, io.BytesIO(payload))
    manifest = tmp_path / "publication.json"
    manifest.write_text(json.dumps({"files": [{
        "role": "synthetic-data", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "size_bytes": archive.stat().st_size,
    }]}))
    identities = tmp_path / "asteroids.csv"
    identities.write_text("id,number,name\n101,2,Pallas\n102,3,Juno\n", encoding="utf-8-sig")
    return archive, manifest, identities


def test_exposure_audit_verifies_archive_and_extracts_metadata_only(tmp_path):
    archive, manifest, identities = _inputs(tmp_path)
    output = tmp_path / "audit"
    result = audit(archive, manifest, identities, output)
    assert result["archive_matches_publication"] is True
    assert result["metadata_record_count"] == 1
    assert result["training_arrays_extracted"] is False
    assert result["historical_exposure_registry_complete"] is False
    assert not list(output.rglob("*.npz"))
    mapping = json.loads((output / "full-identity-aliases.json").read_text())
    first = mapping["objects"][0]
    assert (first["object_id"], first["damit_identity"], first["physical_identity"]) == (
        "asteroid_101", "damit:101", "mpc:2"
    )
    with pytest.raises(ValueError, match="must be new"):
        audit(archive, manifest, identities, output)


def test_exposure_audit_rejects_wrong_published_digest_before_extraction(tmp_path):
    archive, manifest, identities = _inputs(tmp_path)
    document = json.loads(manifest.read_text())
    document["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="differs from public"):
        audit(archive, manifest, identities, output)
    assert not output.exists()


def test_exposure_audit_rejects_path_traversal(tmp_path):
    archive, manifest, identities = _inputs(tmp_path, metadata_name="../escape.json")
    with pytest.raises(ValueError, match="unsafe archive"):
        audit(archive, manifest, identities, tmp_path / "audit")
    assert not (tmp_path / "escape.json").exists()
