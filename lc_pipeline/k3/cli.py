"""Command-line entry points for K3 smoke and definitive publication runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .config import K3ScoreModelConfig, K3TrainingConfig
from .manifest import (
    K3RunManifest,
    configuration_sha256,
    implementation_commit,
    require_clean_repository,
    write_manifest,
)
from .model import CandidateConditionedScorer
from .protocol import K3_PROTOCOL_SHA256, load_protocol
from .publication_run import K3_DIAGNOSTIC_CONTROLS
from .renderer import validate_random_agreement
from .synthetic import EmpiricalNoiseRecipe, SyntheticRecipe, generate_synthetic_record
from .training import examples_from_synthetic_records, fit_k3


def _run_provenance(repository_root: Path) -> dict[str, str]:
    return {
        "protocol_sha256": K3_PROTOCOL_SHA256,
        "implementation_commit": implementation_commit(repository_root),
    }


def _smoke_geometry() -> tuple:
    from ..v2.preprocessing import Observation, ObservationEpoch

    epochs = []
    for epoch_index in range(3):
        observations = []
        for index in range(24):
            angle = 0.025 * index + 0.3 * epoch_index
            observations.append(
                Observation(
                    time_jd=2450000.0 + 30 * epoch_index + index / 24.0,
                    relative_brightness=1.0,
                    sun_asteroid_ecliptic_j2000_au=(1.5 * math.cos(angle), 1.5 * math.sin(angle), 0.2),
                    observer_asteroid_ecliptic_j2000_au=(1.1 * math.cos(angle + 0.4), 1.1 * math.sin(angle + 0.4), -0.1),
                )
            )
        epochs.append(ObservationEpoch(f"smoke-epoch-{epoch_index}", tuple(observations)))
    return tuple(epochs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="delphi-k3")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-protocol")
    renderer = commands.add_parser("renderer-smoke")
    renderer.add_argument("--cases", type=int, default=8)
    renderer.add_argument("--tolerance", type=float, default=1e-5)
    smoke = commands.add_parser("train-smoke")
    smoke.add_argument("--objects", type=int, default=8)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--repository-root", type=Path, default=Path.cwd())
    smoke.add_argument("--allow-dirty", action="store_true")
    publication = commands.add_parser("train-publication-synthetic")
    publication.add_argument("--train-directory", type=Path, required=True)
    publication.add_argument("--validation-directory", type=Path, required=True)
    publication.add_argument("--output-directory", type=Path, required=True)
    publication.add_argument("--seed", type=int, required=True)
    publication.add_argument("--device", default="cuda")
    publication.add_argument("--repository-root", type=Path, default=Path.cwd())
    publication.add_argument("--resume", action="store_true")
    label_shuffle = commands.add_parser("train-publication-synthetic-label-shuffle")
    label_shuffle.add_argument("--train-directory", type=Path, required=True)
    label_shuffle.add_argument("--validation-directory", type=Path, required=True)
    label_shuffle.add_argument("--output-directory", type=Path, required=True)
    label_shuffle.add_argument("--seed", type=int, required=True)
    label_shuffle.add_argument("--device", default="cuda")
    label_shuffle.add_argument("--repository-root", type=Path, default=Path.cwd())
    label_shuffle.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate-publication-synthetic")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--test-directory", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--batch-size", type=int, default=4)
    evaluation.add_argument("--repository-root", type=Path, default=Path.cwd())
    synthetic_swap = commands.add_parser(
        "evaluate-publication-synthetic-input-swap"
    )
    synthetic_swap.add_argument("--checkpoint", type=Path, required=True)
    synthetic_swap.add_argument("--test-directory", type=Path, required=True)
    synthetic_swap.add_argument("--raw-artifact", type=Path, required=True)
    synthetic_swap.add_argument("--output", type=Path, required=True)
    synthetic_swap.add_argument("--seed", type=int, required=True)
    synthetic_swap.add_argument("--device", default="cuda")
    synthetic_swap.add_argument("--batch-size", type=int, default=4)
    synthetic_swap.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    label_evaluation = commands.add_parser("evaluate-publication-synthetic-label-shuffle")
    label_evaluation.add_argument("--checkpoint", type=Path, required=True)
    label_evaluation.add_argument("--test-directory", type=Path, required=True)
    label_evaluation.add_argument("--output", type=Path, required=True)
    label_evaluation.add_argument("--seed", type=int, required=True)
    label_evaluation.add_argument("--device", default="cuda")
    label_evaluation.add_argument("--batch-size", type=int, default=4)
    label_evaluation.add_argument("--repository-root", type=Path, default=Path.cwd())
    real = commands.add_parser("train-publication-real-oof")
    real.add_argument("--pretrained-checkpoint", type=Path, required=True)
    real.add_argument("--synthetic-train-directory", type=Path, required=True)
    real.add_argument("--catalog", type=Path, required=True)
    real.add_argument("--dump-root", type=Path, required=True)
    real.add_argument("--splits", type=Path, required=True)
    real.add_argument("--output-directory", type=Path, required=True)
    real.add_argument("--fold", type=int, required=True)
    real.add_argument("--seed", type=int, required=True)
    real.add_argument("--device", default="cuda")
    real.add_argument("--repository-root", type=Path, default=Path.cwd())
    real.add_argument("--resume", action="store_true")
    real_evaluation = commands.add_parser("evaluate-publication-real-oof")
    real_evaluation.add_argument("--checkpoint", type=Path, required=True)
    real_evaluation.add_argument("--catalog", type=Path, required=True)
    real_evaluation.add_argument("--dump-root", type=Path, required=True)
    real_evaluation.add_argument("--splits", type=Path, required=True)
    real_evaluation.add_argument("--output", type=Path, required=True)
    real_evaluation.add_argument("--fold", type=int, required=True)
    real_evaluation.add_argument("--seed", type=int, required=True)
    real_evaluation.add_argument("--device", default="cuda")
    real_evaluation.add_argument("--batch-size", type=int, default=4)
    real_evaluation.add_argument("--repository-root", type=Path, default=Path.cwd())
    calibration_evaluation = commands.add_parser(
        "evaluate-publication-real-calibration"
    )
    calibration_evaluation.add_argument("--checkpoint", type=Path, required=True)
    calibration_evaluation.add_argument("--catalog", type=Path, required=True)
    calibration_evaluation.add_argument("--dump-root", type=Path, required=True)
    calibration_evaluation.add_argument("--splits", type=Path, required=True)
    calibration_evaluation.add_argument("--output", type=Path, required=True)
    calibration_evaluation.add_argument("--fold", type=int, required=True)
    calibration_evaluation.add_argument("--seed", type=int, required=True)
    calibration_evaluation.add_argument("--device", default="cuda")
    calibration_evaluation.add_argument("--batch-size", type=int, default=4)
    calibration_evaluation.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    swap_evaluation = commands.add_parser("evaluate-publication-real-oof-input-swap")
    swap_evaluation.add_argument("--checkpoint", type=Path, required=True)
    swap_evaluation.add_argument("--catalog", type=Path, required=True)
    swap_evaluation.add_argument("--dump-root", type=Path, required=True)
    swap_evaluation.add_argument("--splits", type=Path, required=True)
    swap_evaluation.add_argument("--output", type=Path, required=True)
    swap_evaluation.add_argument("--fold", type=int, required=True)
    swap_evaluation.add_argument("--seed", type=int, required=True)
    swap_evaluation.add_argument("--device", default="cuda")
    swap_evaluation.add_argument("--batch-size", type=int, default=4)
    swap_evaluation.add_argument("--repository-root", type=Path, default=Path.cwd())
    diagnostic_evaluation = commands.add_parser(
        "evaluate-publication-real-oof-diagnostic"
    )
    diagnostic_evaluation.add_argument("--checkpoint", type=Path, required=True)
    diagnostic_evaluation.add_argument("--catalog", type=Path, required=True)
    diagnostic_evaluation.add_argument("--dump-root", type=Path, required=True)
    diagnostic_evaluation.add_argument("--splits", type=Path, required=True)
    diagnostic_evaluation.add_argument("--output", type=Path, required=True)
    diagnostic_evaluation.add_argument("--fold", type=int, required=True)
    diagnostic_evaluation.add_argument("--seed", type=int, required=True)
    diagnostic_evaluation.add_argument(
        "--control", choices=K3_DIAGNOSTIC_CONTROLS, required=True
    )
    diagnostic_evaluation.add_argument("--device", default="cuda")
    diagnostic_evaluation.add_argument("--batch-size", type=int, default=4)
    diagnostic_evaluation.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    aggregate = commands.add_parser("aggregate-publication-real-oof")
    aggregate.add_argument("--evaluation-directory", type=Path, required=True)
    aggregate.add_argument("--splits", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--seed", type=int, required=True)
    aggregate.add_argument("--repository-root", type=Path, default=Path.cwd())
    comparator = commands.add_parser("build-publication-v1-comparators")
    comparator.add_argument("--v1-oof-root", type=Path, required=True)
    comparator.add_argument("--splits", type=Path, required=True)
    comparator.add_argument("--output", type=Path, required=True)
    comparator.add_argument("--repository-root", type=Path, default=Path.cwd())
    checkpoint_audit = commands.add_parser("audit-publication-checkpoints")
    checkpoint_audit.add_argument("--model-directory", type=Path, required=True)
    checkpoint_audit.add_argument("--output", type=Path, required=True)
    checkpoint_audit.add_argument(
        "--allow-missing-label-shuffle", action="store_true"
    )
    checkpoint_audit.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    calibration = commands.add_parser("calibrate-publication-real-oof")
    calibration.add_argument("--calibration-directory", type=Path, required=True)
    calibration.add_argument("--splits", type=Path, required=True)
    calibration.add_argument("--raw-aggregate", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--seed", type=int, required=True)
    calibration.add_argument("--repository-root", type=Path, default=Path.cwd())
    swap_aggregate = commands.add_parser("aggregate-publication-real-oof-input-swap")
    swap_aggregate.add_argument("--evaluation-directory", type=Path, required=True)
    swap_aggregate.add_argument("--splits", type=Path, required=True)
    swap_aggregate.add_argument("--raw-aggregate", type=Path, required=True)
    swap_aggregate.add_argument("--output", type=Path, required=True)
    swap_aggregate.add_argument("--seed", type=int, required=True)
    swap_aggregate.add_argument("--repository-root", type=Path, default=Path.cwd())
    diagnostic_aggregate = commands.add_parser(
        "aggregate-publication-real-oof-diagnostic"
    )
    diagnostic_aggregate.add_argument(
        "--evaluation-directory", type=Path, required=True
    )
    diagnostic_aggregate.add_argument("--splits", type=Path, required=True)
    diagnostic_aggregate.add_argument("--raw-aggregate", type=Path, required=True)
    diagnostic_aggregate.add_argument("--output", type=Path, required=True)
    diagnostic_aggregate.add_argument("--seed", type=int, required=True)
    diagnostic_aggregate.add_argument(
        "--control", choices=K3_DIAGNOSTIC_CONTROLS, required=True
    )
    diagnostic_aggregate.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    ensemble = commands.add_parser("evaluate-publication-real-oof-ensemble")
    ensemble.add_argument("--evaluation-directory", type=Path, required=True)
    ensemble.add_argument("--model-directory", type=Path, required=True)
    ensemble.add_argument("--catalog", type=Path, required=True)
    ensemble.add_argument("--dump-root", type=Path, required=True)
    ensemble.add_argument("--splits", type=Path, required=True)
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.add_argument("--seeds", type=int, nargs="+", required=True)
    ensemble.add_argument("--device", default="cuda")
    ensemble.add_argument("--repository-root", type=Path, default=Path.cwd())
    calibration_ensemble = commands.add_parser(
        "evaluate-publication-real-calibration-ensemble"
    )
    calibration_ensemble.add_argument(
        "--evaluation-directory", type=Path, required=True
    )
    calibration_ensemble.add_argument("--model-directory", type=Path, required=True)
    calibration_ensemble.add_argument("--catalog", type=Path, required=True)
    calibration_ensemble.add_argument("--dump-root", type=Path, required=True)
    calibration_ensemble.add_argument("--splits", type=Path, required=True)
    calibration_ensemble.add_argument("--output", type=Path, required=True)
    calibration_ensemble.add_argument("--seeds", type=int, nargs="+", required=True)
    calibration_ensemble.add_argument("--device", default="cuda")
    calibration_ensemble.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    calibrated_ensemble = commands.add_parser(
        "calibrate-publication-real-oof-ensemble"
    )
    calibrated_ensemble.add_argument(
        "--calibration-ensemble", type=Path, required=True
    )
    calibrated_ensemble.add_argument("--splits", type=Path, required=True)
    calibrated_ensemble.add_argument("--oof-ensemble", type=Path, required=True)
    calibrated_ensemble.add_argument("--output", type=Path, required=True)
    calibrated_ensemble.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    downstream = commands.add_parser("run-publication-fixed-period-benchmark")
    downstream.add_argument("--executable", type=Path, required=True)
    downstream.add_argument("--source-root", type=Path, required=True)
    downstream.add_argument("--source-archive", type=Path, required=True)
    downstream.add_argument("--ensemble", type=Path, required=True)
    downstream.add_argument("--catalog", type=Path, required=True)
    downstream.add_argument("--dump-root", type=Path, required=True)
    downstream.add_argument("--splits", type=Path, required=True)
    downstream.add_argument("--output-directory", type=Path, required=True)
    downstream.add_argument("--timeout-seconds", type=float, default=3600.0)
    downstream.add_argument("--repository-root", type=Path, default=Path.cwd())
    release = commands.add_parser("build-publication-release")
    release.add_argument("--artifact-root", type=Path, required=True)
    release.add_argument("--comparators", type=Path, required=True)
    release.add_argument("--output-json", type=Path, required=True)
    release.add_argument("--output-tex", type=Path, required=True)
    release.add_argument("--repository-root", type=Path, default=Path.cwd())
    figures = commands.add_parser("build-publication-figures")
    figures.add_argument("--artifact-root", type=Path, required=True)
    figures.add_argument("--comparators", type=Path, required=True)
    figures.add_argument("--release-summary", type=Path, required=True)
    figures.add_argument("--output-directory", type=Path, required=True)
    figures.add_argument("--repository-root", type=Path, default=Path.cwd())
    environment = commands.add_parser("capture-publication-environment")
    environment.add_argument("--output-directory", type=Path, required=True)
    environment.add_argument("--analysis-commit", required=True)
    environment.add_argument("--analysis-pid", type=int)
    environment.add_argument("--repository-root", type=Path, default=Path.cwd())
    archive = commands.add_parser("build-publication-archive-index")
    archive.add_argument("--artifact-root", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    archive.add_argument(
        "--external",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="hash an external file/tree under external/LABEL (repeatable)",
    )
    archive.add_argument("--repository-root", type=Path, default=Path.cwd())
    verify_archive = commands.add_parser("verify-publication-archive-index")
    verify_archive.add_argument("--artifact-root", type=Path, required=True)
    verify_archive.add_argument("--index", type=Path, required=True)
    verify_archive.add_argument(
        "--external", action="append", default=[], metavar="LABEL=PATH"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate-protocol":
        protocol = load_protocol()
        print(json.dumps({"schema": protocol["schema"], "sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "renderer-smoke":
        agreement = validate_random_agreement(cases=arguments.cases, tolerance=arguments.tolerance)
        print(json.dumps(agreement.__dict__, sort_keys=True))
        return 0 if agreement.passed else 1
    if arguments.command == "train-publication-synthetic":
        require_clean_repository(arguments.repository_root)
        from .publication_run import run_synthetic_pretraining

        result = run_synthetic_pretraining(
            arguments.train_directory,
            arguments.validation_directory,
            arguments.output_directory,
            seed=arguments.seed,
            device=arguments.device,
            resume=arguments.resume,
            run_provenance=_run_provenance(arguments.repository_root),
        )
        print(json.dumps({"result": result.__dict__, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True, default=list))
        return 0
    if arguments.command == "evaluate-publication-synthetic":
        require_clean_repository(arguments.repository_root)
        from .publication_run import evaluate_synthetic_checkpoint

        summary = evaluate_synthetic_checkpoint(
            arguments.checkpoint,
            arguments.test_directory,
            arguments.output,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "train-publication-synthetic-label-shuffle":
        require_clean_repository(arguments.repository_root)
        from .publication_run import run_synthetic_label_shuffle_control

        result = run_synthetic_label_shuffle_control(
            arguments.train_directory,
            arguments.validation_directory,
            arguments.output_directory,
            seed=arguments.seed,
            device=arguments.device,
            resume=arguments.resume,
            run_provenance=_run_provenance(arguments.repository_root),
        )
        print(json.dumps({"result": result.__dict__, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True, default=list))
        return 0
    if arguments.command == "evaluate-publication-synthetic-input-swap":
        require_clean_repository(arguments.repository_root)
        from .publication_run import evaluate_synthetic_input_swap_checkpoint

        summary = evaluate_synthetic_input_swap_checkpoint(
            arguments.checkpoint,
            arguments.test_directory,
            arguments.raw_artifact,
            arguments.output,
            seed=arguments.seed,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "evaluate-publication-synthetic-label-shuffle":
        require_clean_repository(arguments.repository_root)
        from .publication_run import evaluate_synthetic_label_shuffle_checkpoint

        summary = evaluate_synthetic_label_shuffle_checkpoint(
            arguments.checkpoint,
            arguments.test_directory,
            arguments.output,
            seed=arguments.seed,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "train-publication-real-oof":
        require_clean_repository(arguments.repository_root)
        from .publication_run import run_real_oof_finetuning

        result = run_real_oof_finetuning(
            arguments.pretrained_checkpoint,
            arguments.synthetic_train_directory,
            arguments.catalog,
            arguments.dump_root,
            arguments.splits,
            arguments.output_directory,
            fold=arguments.fold,
            seed=arguments.seed,
            device=arguments.device,
            resume=arguments.resume,
            run_provenance=_run_provenance(arguments.repository_root),
        )
        print(json.dumps({"result": result.__dict__, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True, default=list))
        return 0
    if arguments.command == "evaluate-publication-real-oof":
        require_clean_repository(arguments.repository_root)
        from .publication_run import evaluate_real_oof_checkpoint

        summary = evaluate_real_oof_checkpoint(
            arguments.checkpoint,
            arguments.catalog,
            arguments.dump_root,
            arguments.splits,
            arguments.output,
            fold=arguments.fold,
            seed=arguments.seed,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "evaluate-publication-real-calibration":
        require_clean_repository(arguments.repository_root)
        from .publication_run import evaluate_real_calibration_checkpoint

        summary = evaluate_real_calibration_checkpoint(
            arguments.checkpoint,
            arguments.catalog,
            arguments.dump_root,
            arguments.splits,
            arguments.output,
            fold=arguments.fold,
            seed=arguments.seed,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "evaluate-publication-real-oof-input-swap":
        require_clean_repository(arguments.repository_root)
        from .publication_run import evaluate_real_oof_input_swap_checkpoint

        summary = evaluate_real_oof_input_swap_checkpoint(
            arguments.checkpoint,
            arguments.catalog,
            arguments.dump_root,
            arguments.splits,
            arguments.output,
            fold=arguments.fold,
            seed=arguments.seed,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "evaluate-publication-real-oof-diagnostic":
        require_clean_repository(arguments.repository_root)
        from .publication_run import evaluate_real_oof_diagnostic_checkpoint

        summary = evaluate_real_oof_diagnostic_checkpoint(
            arguments.checkpoint,
            arguments.catalog,
            arguments.dump_root,
            arguments.splits,
            arguments.output,
            fold=arguments.fold,
            seed=arguments.seed,
            control=arguments.control,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "aggregate-publication-real-oof":
        require_clean_repository(arguments.repository_root)
        from .publication_run import aggregate_real_oof_artifacts

        summary = aggregate_real_oof_artifacts(
            arguments.evaluation_directory,
            arguments.splits,
            arguments.output,
            seed=arguments.seed,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "build-publication-v1-comparators":
        require_clean_repository(arguments.repository_root)
        from .publication_run import build_v1_comparator_artifact

        summary = build_v1_comparator_artifact(
            arguments.v1_oof_root,
            arguments.splits,
            arguments.output,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "audit-publication-checkpoints":
        require_clean_repository(arguments.repository_root)
        from .publication_run import audit_publication_checkpoint_set

        summary = audit_publication_checkpoint_set(
            arguments.model_directory,
            arguments.output,
            require_label_shuffle=not arguments.allow_missing_label_shuffle,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "calibrate-publication-real-oof":
        require_clean_repository(arguments.repository_root)
        from .publication_run import calibrate_real_oof_artifacts

        summary = calibrate_real_oof_artifacts(
            arguments.calibration_directory,
            arguments.splits,
            arguments.raw_aggregate,
            arguments.output,
            seed=arguments.seed,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "aggregate-publication-real-oof-input-swap":
        require_clean_repository(arguments.repository_root)
        from .publication_run import aggregate_real_oof_input_swap_artifacts

        summary = aggregate_real_oof_input_swap_artifacts(
            arguments.evaluation_directory,
            arguments.splits,
            arguments.raw_aggregate,
            arguments.output,
            seed=arguments.seed,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "aggregate-publication-real-oof-diagnostic":
        require_clean_repository(arguments.repository_root)
        from .publication_run import aggregate_real_oof_diagnostic_artifacts

        summary = aggregate_real_oof_diagnostic_artifacts(
            arguments.evaluation_directory,
            arguments.splits,
            arguments.raw_aggregate,
            arguments.output,
            seed=arguments.seed,
            control=arguments.control,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "evaluate-publication-real-oof-ensemble":
        require_clean_repository(arguments.repository_root)
        from .publication_run import evaluate_real_oof_ensemble

        summary = evaluate_real_oof_ensemble(
            arguments.evaluation_directory,
            arguments.model_directory,
            arguments.catalog,
            arguments.dump_root,
            arguments.splits,
            arguments.output,
            seeds=arguments.seeds,
            device=arguments.device,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "evaluate-publication-real-calibration-ensemble":
        require_clean_repository(arguments.repository_root)
        from .publication_run import evaluate_real_calibration_ensemble

        summary = evaluate_real_calibration_ensemble(
            arguments.evaluation_directory,
            arguments.model_directory,
            arguments.catalog,
            arguments.dump_root,
            arguments.splits,
            arguments.output,
            seeds=arguments.seeds,
            device=arguments.device,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "calibrate-publication-real-oof-ensemble":
        require_clean_repository(arguments.repository_root)
        from .publication_run import calibrate_real_oof_ensemble_artifacts

        summary = calibrate_real_oof_ensemble_artifacts(
            arguments.calibration_ensemble,
            arguments.splits,
            arguments.oof_ensemble,
            arguments.output,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "run-publication-fixed-period-benchmark":
        require_clean_repository(arguments.repository_root)
        from .downstream import run_fixed_period_benchmark

        summary = run_fixed_period_benchmark(
            executable=arguments.executable,
            source_root=arguments.source_root,
            source_archive=arguments.source_archive,
            ensemble_path=arguments.ensemble,
            catalog_path=arguments.catalog,
            dump_root=arguments.dump_root,
            split_path=arguments.splits,
            output_directory=arguments.output_directory,
            timeout_seconds=arguments.timeout_seconds,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "build-publication-release":
        require_clean_repository(arguments.repository_root)
        from .release import build_publication_release

        summary = build_publication_release(
            artifact_root=arguments.artifact_root,
            comparator_path=arguments.comparators,
            output_json=arguments.output_json,
            output_tex=arguments.output_tex,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "build-publication-figures":
        require_clean_repository(arguments.repository_root)
        from .figures import build_publication_figures

        summary = build_publication_figures(
            artifact_root=arguments.artifact_root,
            comparator_path=arguments.comparators,
            release_summary_path=arguments.release_summary,
            output_directory=arguments.output_directory,
        )
        print(json.dumps({"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256}, sort_keys=True))
        return 0
    if arguments.command == "capture-publication-environment":
        require_clean_repository(arguments.repository_root)
        from .environment import capture_publication_environment

        summary = capture_publication_environment(
            output_directory=arguments.output_directory,
            repository_root=arguments.repository_root,
            analysis_commit=arguments.analysis_commit,
            analysis_pid=arguments.analysis_pid,
        )
        print(
            json.dumps(
                {"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256},
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "build-publication-archive-index":
        require_clean_repository(arguments.repository_root)
        from .archive import K3ArchiveError, build_publication_archive_index

        external_sources: dict[str, Path] = {}
        for value in arguments.external:
            label, separator, raw_path = value.partition("=")
            if not separator or not label or not raw_path or label in external_sources:
                raise K3ArchiveError(
                    "--external must be a unique nonempty LABEL=PATH value"
                )
            external_sources[label] = Path(raw_path)
        summary = build_publication_archive_index(
            artifact_root=arguments.artifact_root,
            output=arguments.output,
            tool_commit=implementation_commit(arguments.repository_root),
            external_sources=external_sources,
        )
        print(
            json.dumps(
                {"summary": summary, "protocol_sha256": K3_PROTOCOL_SHA256},
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "verify-publication-archive-index":
        from .archive import K3ArchiveError, verify_publication_archive_index

        external_sources = {}
        for value in arguments.external:
            label, separator, raw_path = value.partition("=")
            if not separator or not label or not raw_path or label in external_sources:
                raise K3ArchiveError(
                    "--external must be a unique nonempty LABEL=PATH value"
                )
            external_sources[label] = Path(raw_path)
        summary = verify_publication_archive_index(
            artifact_root=arguments.artifact_root,
            index=arguments.index,
            external_sources=external_sources,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if arguments.command == "train-smoke":
        if arguments.objects < 4:
            raise SystemExit("--objects must be at least four for a train/validation smoke split")
        if not arguments.allow_dirty:
            require_clean_repository(arguments.repository_root)
        recipe = SyntheticRecipe(
            split="smoke",
            object_count=arguments.objects,
            master_seed=17,
            noise=EmpiricalNoiseRecipe("declared-smoke-noise"),
        )
        from .renderer import ConvexFacets, sample_convex_ellipsoids

        generator = torch.Generator().manual_seed(17)
        normals, areas, _ = sample_convex_ellipsoids(arguments.objects, generator=generator, facet_count=64)
        records = tuple(
            generate_synthetic_record(
                object_id=f"smoke-{index:04d}",
                geometry_donor_id=f"smoke-geometry-{index:04d}",
                geometry_epochs=_smoke_geometry(),
                shape=ConvexFacets(normals[index].numpy(), areas[index].numpy(), f"smoke-shape-{index:04d}"),
                recipe=recipe,
                seed=1000 + index,
            )
            for index in range(arguments.objects)
        )
        examples = examples_from_synthetic_records(records)
        config = K3TrainingConfig(
            seed=17, batch_size=2, synthetic_max_epochs=2, synthetic_patience=1,
            real_max_epochs=2, real_patience=1, uniform_negatives=4, hard_negatives=2,
        )
        model_config = K3ScoreModelConfig(
            hidden_dim=32, attention_heads=4, convolution_blocks=1, set_encoder_layers=1,
            dropout=0.0, score_chunk_size=1024,
        )
        model = CandidateConditionedScorer(model_config)
        result = fit_k3(
            model, examples[:-2], examples[-2:], config=config, device="cpu",
            checkpoint_path=arguments.output.with_suffix(".pt"),
            run_provenance=_run_provenance(arguments.repository_root),
        )
        payload = {"result": result.__dict__, "protocol_sha256": K3_PROTOCOL_SHA256, "training_config_sha256": configuration_sha256(config.as_mapping())}
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(payload, sort_keys=True, default=list) + "\n", encoding="utf-8")
        manifest = K3RunManifest(
            run_id=arguments.output.stem,
            stage="synthetic",
            status="complete",
            protocol_sha256=K3_PROTOCOL_SHA256,
            implementation_commit=implementation_commit(arguments.repository_root),
            configuration_sha256=configuration_sha256({"training": config.as_mapping(), "model": model_config.as_mapping()}),
            artifacts=(),
        )
        write_manifest(arguments.output.with_suffix(".manifest.json"), manifest)
        print(json.dumps(payload, sort_keys=True, default=list))
        return 0
    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
