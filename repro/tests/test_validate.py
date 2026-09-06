from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from repro.validate import (
    DEFAULT_RELEASE_SPEC,
    ContractError,
    load_json,
    load_release_spec,
    sha256_file,
    validate_bundle,
    validate_manifest,
)

REPOSITORY = "https://github.com/example/delphi"
COMMIT = "1" * 40


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def passing_observed(metric: str, operator: str, threshold: object) -> object:
    range_midpoints = {
        "period_empirical_coverage_68": 0.68,
        "period_empirical_coverage_95": 0.95,
    }
    if metric in range_midpoints:
        return range_midpoints[metric]
    if operator in {"eq", "lte", "gte"}:
        return copy.deepcopy(threshold)
    if operator == "lt":
        return float(threshold) - 1.0
    if operator == "gt":
        return float(threshold) + 1.0
    raise AssertionError(operator)


def build_valid_bundle(root: Path) -> tuple[list[Path], Path]:
    artifact_root = root / "bytes"
    (artifact_root / "data").mkdir(parents=True)
    (artifact_root / "results").mkdir(parents=True)
    source_bytes = b"canonical test source\n"
    prediction_bytes = b"directed pole density test output\n"
    source_file = artifact_root / "data" / "objects.parquet"
    prediction_file = artifact_root / "results" / "predictions.json"
    source_file.write_bytes(source_bytes)
    prediction_file.write_bytes(prediction_bytes)

    source_artifact = {
        "artifact_id": "data.objects",
        "logical_path": "data/objects.parquet",
        "sha256": digest(source_bytes),
        "size_bytes": len(source_bytes),
        "media_type": "application/vnd.apache.parquet",
        "role": "source",
    }
    prediction_artifact = {
        "artifact_id": "results.predictions",
        "logical_path": "results/predictions.json",
        "sha256": digest(prediction_bytes),
        "size_bytes": len(prediction_bytes),
        "media_type": "application/json",
        "role": "predictions",
    }

    data_manifest = {
        "schema_version": "2.0.0",
        "manifest_type": "data",
        "manifest_id": "damit.qf3.v2",
        "dataset": {
            "name": "DAMIT quality-filtered directed poles",
            "version": "2025-06-10",
            "created_at_utc": "2026-01-01T00:00:00Z",
        },
        "source": {
            "name": "DAMIT",
            "snapshot_id": "damit-20250610T000301Z",
            "canonical_uri": "https://damit.cuni.cz/",
            "retrieved_at_utc": "2025-06-10T00:03:01Z",
            "license": "CC-BY-4.0",
            "citation": "DAMIT and each original shape-model publication",
            "sha256": digest(b"damit-20250610T000301Z source archive"),
        },
        "label_policy": {
            "minimum_quality_flag": 3,
            "coordinate_frame": "ecliptic_j2000",
            "directed_poles": True,
            "antipodal_equivalence": False,
        },
        "splits": [
            {"name": "train", "n_objects": 80, "object_ids_sha256": digest(b"train ids")},
            {"name": "validation", "n_objects": 10, "object_ids_sha256": digest(b"validation ids")},
            {"name": "prospective_test", "n_objects": 10, "object_ids_sha256": digest(b"prospective ids")},
        ],
        "files": [source_artifact],
        "builder": {
            "git_repository": REPOSITORY,
            "git_commit": COMMIT,
            "git_dirty": False,
            "command": ["python3", "-m", "lc_pipeline.data.build_damit_v2", "--manifest", "repro/data.json"],
        },
    }
    data_path = root / "data.json"
    write_json(data_path, data_manifest)

    spec_hash = sha256_file(DEFAULT_RELEASE_SPEC)
    run_input = {**source_artifact, "role": "input"}
    run_manifest = {
        "schema_version": "2.0.0",
        "manifest_type": "run",
        "run_id": "seed17.final",
        "created_at_utc": "2026-01-02T00:00:00Z",
        "release_spec_sha256": spec_hash,
        "git": {"repository": REPOSITORY, "commit": COMMIT, "dirty": False},
        "environment": {
            "python": "3.12.3",
            "platform": "linux-x86_64",
            "cuda": "12.4",
            "dependency_lock_sha256": digest(b"dependency lock"),
        },
        "randomness": {"training_seed": 17, "deterministic_algorithms": True},
        "data_manifests": [{"manifest_id": "damit.qf3.v2", "sha256": sha256_file(data_path)}],
        "command": ["python3", "-m", "lc_pipeline.train", "--seed", "17"],
        "inputs": [run_input],
        "outputs": [prediction_artifact],
    }
    run_path = root / "run.json"
    write_json(run_path, run_manifest)

    archived_prediction = {**prediction_artifact, "generated_by_run_id": "seed17.final"}
    artifact_manifest = {
        "schema_version": "2.0.0",
        "manifest_type": "artifact",
        "record_id": "delphi.v2.artifacts",
        "created_at_utc": "2026-01-03T00:00:00Z",
        "archive": {
            "provider": "zenodo",
            "doi": "10.5281/zenodo.12345678",
            "record_url": "https://zenodo.org/records/12345678",
            "published_at_utc": "2026-01-03T00:00:00Z",
            "immutable": True,
        },
        "license": "MIT",
        "creators": [{"name": "Researcher, Test", "orcid": "https://orcid.org/0000-0002-1825-0097"}],
        "related_identifiers": [
            {"identifier": f"{REPOSITORY}/commit/{COMMIT}", "relation": "isSupplementTo"}
        ],
        "artifacts": [archived_prediction],
    }
    artifact_path = root / "artifact.json"
    write_json(artifact_path, artifact_manifest)

    spec = load_release_spec()
    evidence = {"artifact_id": prediction_artifact["artifact_id"], "sha256": prediction_artifact["sha256"]}
    gate_results = [
        {
            "gate_id": gate["gate_id"],
            "metric": gate["metric"],
            "observed": passing_observed(gate["metric"], gate["operator"], gate["threshold"]),
            "operator": gate["operator"],
            "threshold": copy.deepcopy(gate["threshold"]),
            "passed": True,
            "evidence": [copy.deepcopy(evidence)],
        }
        for gate in spec["required_gates"]
    ]
    verdict_manifest = {
        "schema_version": "2.0.0",
        "manifest_type": "verdict",
        "release_id": "v2.0.0",
        "evaluated_at_utc": "2026-01-05T00:00:00Z",
        "release_spec_sha256": spec_hash,
        "run_manifest": {"manifest_id": "seed17.final", "sha256": sha256_file(run_path)},
        "artifact_manifest": {"manifest_id": "delphi.v2.artifacts", "sha256": sha256_file(artifact_path)},
        "prospective_test": {
            "split_lock_sha256": digest(b"prospective split lock"),
            "locked_at_utc": "2026-01-01T00:00:00Z",
            "labels_revealed_at_utc": "2026-01-04T00:00:00Z",
            "custodian_name": "Independent Custodian",
            "custodian_independent": True,
            "no_test_set_tuning": True,
        },
        "gate_results": gate_results,
        "overall_verdict": "PASS",
        "approved_for_release": True,
    }
    verdict_path = root / "verdict.json"
    write_json(verdict_path, verdict_manifest)
    return [data_path, run_path, artifact_path, verdict_path], artifact_root


class ReleaseSpecTests(unittest.TestCase):
    def test_locked_release_spec_is_valid(self) -> None:
        spec = load_release_spec()
        self.assertEqual(spec["spec_version"], "2.1.0")
        self.assertEqual(spec["randomness"]["training_seeds"], [17, 42, 137, 777, 2027])
        self.assertEqual(spec["randomness"]["bootstrap_seed"], 20260901)
        self.assertEqual(spec["evaluation"]["primary_metrics"]["pole"], "directed_top1_angular_error_deg")

    def test_directional_superiority_and_reporting_are_preregistered(self) -> None:
        evaluation = load_release_spec()["evaluation"]
        self.assertEqual(evaluation["hypothesis_tests"]["familywise_alpha"], 0.01)
        self.assertEqual(
            evaluation["hypothesis_tests"]["directional_superiority_alternative"],
            "one_sided_preregistered",
        )
        self.assertEqual(evaluation["hypothesis_tests"]["multiplicity_correction"], "holm")
        self.assertEqual(evaluation["reporting"]["headline_pole_metric"], "directed_top1_angular_error_deg")
        self.assertEqual(evaluation["reporting"]["matched_k_reporting_role"], "secondary_baseline_not_headline")

    def test_requested_gate_thresholds_are_explicit(self) -> None:
        spec = load_release_spec()
        gates = {gate["gate_id"]: gate for gate in spec["required_gates"]}
        expected = {
            "MODEL_DERANGED_INPUT_POINT_GAP": ("gte", 5.0),
            "MODEL_DERANGED_INPUT_CI95_LOWER": ("gte", 2.0),
            "MODEL_DERANGED_INPUT_ONE_SIDED_HOLM_P": ("lte", 0.01),
            "MODEL_MATCHED_K_PRIOR_POINT_GAP": ("gte", 5.0),
            "MODEL_MATCHED_K_PRIOR_CI95_LOWER": ("gte", 2.0),
            "MODEL_MATCHED_K_PRIOR_ONE_SIDED_HOLM_P": ("lte", 0.01),
            "MODEL_LABEL_SHUFFLE_CI95_LOWER": ("gte", 2.0),
            "MODEL_LABEL_SHUFFLE_ONE_SIDED_HOLM_P": ("lte", 0.01),
            "MODEL_SEEDS_POSITIVE_COUNT": ("gte", 4),
            "MODEL_SEEDS_MINIMUM_GAP": ("gte", -1.0),
            "MODEL_ALL_SIGNAL_ABLATION_CI95_LOWER": ("gte", 2.0),
            "MODEL_EACH_CLAIMED_CHANNEL_CI95_LOWER": ("gt", 0.5),
            "MODEL_EACH_CLAIMED_CHANNEL_ONE_SIDED_HOLM_P": ("lte", 0.01),
            "NUMERIC_CPU_GPU_MAX_ANGULAR_DISAGREEMENT": ("lte", 0.05),
            "NUMERIC_CPU_GPU_AGGREGATE_DISAGREEMENT": ("lte", 0.01),
        }
        for gate_id, (operator, threshold) in expected.items():
            self.assertIn(gate_id, gates)
            self.assertEqual(gates[gate_id]["operator"], operator)
            self.assertEqual(gates[gate_id]["threshold"], threshold)
        self.assertNotIn("NUMERIC_CPU_GPU_PARITY", gates)

    def test_conditional_consensus_policy_is_locked(self) -> None:
        policy = load_release_spec()["evaluation"]["period_consensus"]
        self.assertEqual(policy["minimum_acc5_absolute_gain_percentage_points"], 5.0)
        self.assertEqual(policy["minimum_median_relative_error_fractional_reduction"], 0.10)
        self.assertEqual(policy["effect_size_logic"], "either")
        self.assertEqual(policy["maximum_one_sided_holm_adjusted_p_value"], 0.01)

    def test_locked_gate_threshold_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release_spec.yaml"
            spec = load_json(DEFAULT_RELEASE_SPEC)
            gate = next(
                item for item in spec["required_gates"]
                if item["gate_id"] == "MODEL_DERANGED_INPUT_POINT_GAP"
            )
            gate["threshold"] = 4.99
            write_json(path, spec)
            with self.assertRaisesRegex(ContractError, "locked gate value"):
                load_release_spec(path)

    def test_cli_validates_spec_without_optional_packages(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "repro.validate", "--spec-only"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "VALID")


class ManifestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.paths, self.artifact_root = build_valid_bundle(self.root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_complete_bundle_and_artifact_bytes_validate(self) -> None:
        validate_bundle(self.paths, artifact_root=self.artifact_root)

    def test_missing_required_field_fails_closed(self) -> None:
        document = load_json(self.paths[0])
        del document["builder"]
        with self.assertRaisesRegex(ContractError, "missing required field 'builder'"):
            validate_manifest(document)

    def test_malformed_hash_fails_closed(self) -> None:
        document = load_json(self.paths[0])
        document["files"][0]["sha256"] = "abc"
        with self.assertRaisesRegex(ContractError, "64-character SHA-256"):
            validate_manifest(document)

    def test_all_zero_placeholder_hash_fails_closed(self) -> None:
        document = load_json(self.paths[0])
        document["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractError, "all-zero placeholder"):
            validate_manifest(document)

    def test_dirty_git_state_fails_closed(self) -> None:
        document = load_json(self.paths[1])
        document["git"]["dirty"] = True
        with self.assertRaisesRegex(ContractError, "clean Git commit"):
            validate_manifest(document)

    def test_integer_cannot_impersonate_boolean_contract(self) -> None:
        document = load_json(self.paths[1])
        document["randomness"]["deterministic_algorithms"] = 1
        with self.assertRaisesRegex(ContractError, "must equal True"):
            validate_manifest(document)

    def test_short_git_commit_fails_closed(self) -> None:
        document = load_json(self.paths[1])
        document["git"]["commit"] = COMMIT[:8]
        with self.assertRaisesRegex(ContractError, "40-character Git commit"):
            validate_manifest(document)

    def test_absolute_posix_path_fails_closed(self) -> None:
        document = load_json(self.paths[0])
        document["files"][0]["logical_path"] = "/private/data.parquet"
        with self.assertRaisesRegex(ContractError, "absolute local paths"):
            validate_manifest(document)

    def test_absolute_windows_path_fails_closed(self) -> None:
        document = load_json(self.paths[0])
        document["files"][0]["logical_path"] = "C:\\private\\data.parquet"
        with self.assertRaisesRegex(ContractError, "absolute local paths"):
            validate_manifest(document)

    def test_release_spec_byte_mismatch_fails_closed(self) -> None:
        document = load_json(self.paths[1])
        document["release_spec_sha256"] = digest(b"different release policy")
        with self.assertRaisesRegex(ContractError, "exact release specification bytes"):
            validate_manifest(document)

    def test_conflicting_artifact_identity_across_manifests_fails(self) -> None:
        run_document = load_json(self.paths[1])
        run_document["inputs"][0]["sha256"] = digest(b"conflicting bytes")
        write_json(self.paths[1], run_document)
        with self.assertRaisesRegex(ContractError, "conflicts with another manifest"):
            validate_bundle(self.paths)

    def test_mutated_artifact_bytes_fail_hash_verification(self) -> None:
        (self.artifact_root / "results" / "predictions.json").write_bytes(b"tampered output\n")
        with self.assertRaisesRegex(ContractError, "mismatch"):
            validate_bundle(self.paths, artifact_root=self.artifact_root)

    def test_missing_cross_manifest_reference_fails_closed(self) -> None:
        with self.assertRaisesRegex(ContractError, "referenced data manifest is not present"):
            validate_bundle(self.paths[1:])

    def test_gate_result_cannot_self_certify_false_observation(self) -> None:
        verdict = load_json(self.paths[3])
        verdict["gate_results"][0]["observed"] = False
        with self.assertRaisesRegex(ContractError, "must be False"):
            validate_manifest(verdict)

    def test_duplicate_metric_bounds_must_use_one_observed_value(self) -> None:
        verdict = load_json(self.paths[3])
        upper = next(
            result for result in verdict["gate_results"]
            if result["gate_id"] == "PERIOD_COVERAGE_68_UPPER"
        )
        upper["observed"] = 0.69
        with self.assertRaisesRegex(ContractError, "conflicts with the same metric"):
            validate_manifest(verdict)

    def test_omitted_gate_prevents_release(self) -> None:
        verdict = load_json(self.paths[3])
        verdict["gate_results"].pop()
        with self.assertRaisesRegex(ContractError, "missing required gates"):
            validate_manifest(verdict)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        path = self.root / "duplicate.json"
        path.write_text('{"manifest_type":"data","manifest_type":"run"}', encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "duplicate JSON object key"):
            load_json(path)

    def test_zenodo_template_is_invalid_until_materialized(self) -> None:
        template = load_json(Path("repro/artifact_manifest.zenodo.template.json"))
        with self.assertRaises(ContractError):
            validate_manifest(template)


if __name__ == "__main__":
    unittest.main()
