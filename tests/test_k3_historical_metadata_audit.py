import json

from repro.audit_k3_historical_metadata import audit


def test_metadata_audit_does_not_clear_unenumerated_phase40_history(tmp_path):
    identities = tmp_path / "identities.json"
    identities.write_text(json.dumps({"schema": "delphi.k3-survey-identity-map.v1", "objects": [{"object_id": f"asteroid_{index}"} for index in range(1, 171)]}))
    phase30 = tmp_path / "phase30"
    phase30.mkdir()
    (phase30 / "manifest.json").write_text('{"object_ids": ["asteroid_2", "asteroid_4"]}')
    phase40 = tmp_path / "phase40"
    phase40.mkdir()
    (phase40 / "config.yaml").write_text("data:\n  synthetic:\n    manifest_path: data/synthetic/manifest.jsonl\n")
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"schema": "delphi.k3-survey-identity-map.v1", "objects": [{"object_id": "asteroid_2", "physical_identity": "mpc:20"}, {"object_id": "asteroid_4", "physical_identity": "mpc:40"}]}))
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"schema": "delphi.k3-alcdef-gaia-eligibility.v1", "candidates": [{"object_id": "mpc:40"}]}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"object_id": "synth_00000", "shard": "shard_00000.npz"}\n')
    result = audit(phase30, phase40, identities, aliases, candidates, manifest)
    assert result["phase30"]["all_enumerated_ids_are_original"] is True
    assert result["phase40"]["identity_enumeration_available"] is False
    assert result["phase40"]["manifest"]["donor_identity_enumerable"] is False
    assert result["phase30"]["candidate_inventory_overlap"] == ["mpc:40"]
    assert result["conclusion"]["phase40_clears_external_candidates"] is False
