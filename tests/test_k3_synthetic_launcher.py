"""The regeneration preflight must reject recipe and donor drift without rendering."""

import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from lc_pipeline.k3.synthetic import (
    EmpiricalNoiseRecipe,
    SyntheticDataError,
    SyntheticRecipe,
    canonical_json,
    sha256_file,
)
from repro import generate_k3_synthetic as launcher


@pytest.mark.parametrize("split,count,seed", [
    ("train", 20000, 20260911), ("validation", 2000, 20260912), ("test", 2000, 20260913),
])
def test_historical_recipes_match_all_three_frozen_hashes(split, count, seed):
    recipe = SyntheticRecipe(split, count, seed, EmpiricalNoiseRecipe(launcher.NOISE_PROVENANCE))
    proof = launcher.recipe_versions(recipe, launcher.OLD_HASHES[split])
    assert proof["changed_fields"] == ["schema"]
    assert proof["v1_sha256"] != proof["v2_sha256"]
    assert {k: v for k, v in proof["v1"].items() if k != "schema"} == {
        k: v for k, v in proof["v2"].items() if k != "schema"}
    with pytest.raises(SyntheticDataError, match="v1 recipe hash mismatch"):
        launcher.recipe_versions(replace(recipe, master_seed=seed + 1), launcher.OLD_HASHES[split])


@pytest.fixture
def frozen_split(tmp_path, monkeypatch):
    recipe = SyntheticRecipe("smoke", 4, 123, EmpiricalNoiseRecipe("test-only"))
    old_recipe = {**asdict(recipe), "schema": launcher.OLD_SCHEMA}
    old_hash = hashlib.sha256(canonical_json(old_recipe).encode()).hexdigest()
    monkeypatch.setitem(launcher.OLD_HASHES, "smoke", old_hash)
    donors = ("asteroid_a", "asteroid_b")
    shapes = {"asteroid_a": "asteroid_a/model_1", "asteroid_b": "asteroid_b/model_2"}
    for index in range(2):
        path = tmp_path / f"smoke-{index:04d}.npz"
        # Independent, explicit round-robin geometry and affine shape pairing.
        np.savez(path, object_ids=[f"synthetic-smoke-{i:06d}" for i in range(index * 2, index * 2 + 2)],
                 seeds=np.arange(123 + index * 2, 125 + index * 2),
                 geometry_donor_ids=["asteroid_a", "asteroid_b"],
                 shape_donor_ids=["asteroid_b/model_2", "asteroid_a/model_1"])
        manifest = {"schema": launcher.LEGACY_SYNTHETIC_SHARD_SCHEMA, "split": "smoke",
                    "shard_index": index, "object_count": 2, "recipe_sha256": old_hash,
                    "data_file": path.name, "data_sha256": sha256_file(path),
                    "geometry_donor_ids": list(donors), "shape_donor_ids": sorted(shapes.values())}
        path.with_suffix(".npz.manifest.json").write_text(json.dumps(manifest))
    return tmp_path, recipe, donors, shapes


def test_check_accepts_frozen_pairs_and_seeds_without_rendering(frozen_split, monkeypatch):
    def forbidden_render(**_kwargs):
        pytest.fail("--check must never render")

    monkeypatch.setattr(launcher, "generate_synthetic_record", forbidden_render)
    root, recipe, donors, shapes = frozen_split
    proof = launcher.check_frozen_split(root, recipe, donors, shapes, shard_size=2)
    assert proof["objects"] == 4
    assert proof["all_record_ids_seeds_and_shape_pairs_match"] is True


@pytest.mark.parametrize("change", ["pair", "seed", "manifest_donor", "missing_shard", "hash"])
def test_check_rejects_frozen_drift(frozen_split, change):
    root, recipe, donors, shapes = frozen_split
    path = root / "smoke-0000.npz"
    manifest_path = path.with_suffix(".npz.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if change in {"pair", "seed"}:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if change == "pair":
            arrays["shape_donor_ids"] = arrays["shape_donor_ids"][::-1]
        else:
            arrays["seeds"] += 1
        np.savez(path, **arrays)
        manifest["data_sha256"] = sha256_file(path)
    elif change == "manifest_donor":
        manifest["geometry_donor_ids"] = ["asteroid_c"]
    elif change == "missing_shard":
        path.unlink()
    else:
        with path.open("ab") as handle:
            handle.write(b"tamper")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SyntheticDataError):
        launcher.check_frozen_split(root, recipe, donors, shapes, shard_size=2)


def test_check_cli_does_not_generate_or_create_output(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "check_inputs", lambda _args: ({"status": "passed"}, {}, {}, {}))

    def forbidden_generation(*_args):
        pytest.fail("--check must not generate")

    monkeypatch.setattr(launcher, "generate", forbidden_generation)
    output = tmp_path / "absent"
    assert launcher.main(["--check", "--output-dir", str(output),
                          "--dump-root", str(tmp_path / "dump"),
                          "--frozen-dir", str(tmp_path / "frozen"),
                          "--noise-bank", str(tmp_path / "noise.npz")]) == 0
    assert not output.exists()


@pytest.mark.parametrize("missing", ["--dump-root", "--frozen-dir", "--noise-bank"])
def test_cli_requires_portable_input_paths(missing, tmp_path, capsys):
    args = ["--check"]
    for flag in ("--dump-root", "--frozen-dir", "--noise-bank"):
        if flag != missing:
            args.extend((flag, str(tmp_path / flag[2:])))
    with pytest.raises(SystemExit) as exc:
        launcher.main(args)
    assert exc.value.code == 2
    assert f"the following arguments are required: {missing}" in capsys.readouterr().err
