# Reproducing results and figures

The measured experiment is bound to training commit
`7874092b81d9456e5bf52fba9adb699b8218046f` and protocol SHA-256
`84dee816d08a60b7780a6d1d400add16136fa96c0677e84929700cee8bf22a50`.
The training commit records the local analysis state but is not publicly
resolvable on GitHub. The released source at commit `4d1bf1359` regenerates all
reported numeric results from the archived artifacts. An export commit
identifies later packaging/interface changes separately from training. Do not
relabel a new training run as the frozen experiment.

## Rebuild from frozen predictions (CPU)

Download these assets from the [v1.0.0 GitHub release](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.0.0):
`delphi-k3-artifacts-v1.0.0.tar.gz`, `delphi-k3-synthetic-data-v1.0.0.tar.gz`,
`delphi-k3-source-v1.0.0.tar.gz`, `delphi-k3-v1-comparator-v1.0.0.npz`,
`publication-archive-manifest.json`, and `VERIFY.md`. A checkout alone does
not contain training weights, raw DAMIT photometry, or the frozen evidence.
Before extracting, read `VERIFY.md` and verify the release-asset hashes against
`publication-archive-manifest.json`. The output directory must not already exist.

```bash
python -m repro.reproduce_k3_paper \
  --artifact-root /path/to/k3-definitive-7874092 \
  --comparators /path/to/v1-comparators.npz \
  --catalog repro/data/damit-20250610T000301Z/catalog.jsonl \
  --output outputs/paper-reproduction
```

This checks artifact bindings, regenerates the publication summary and TeX values,
and requires exact matches to the frozen versions. It rebuilds the three primary
plots plus architecture, descriptive error-property, candidate-sky, perturbation
and containment plots, exporting object-level CSV and checksums. Descriptive
plots are not additional confirmatory tests or evidence of causal feature effects.

The maintenance revision also exposes quantities already present in the frozen
per-object evidence:

```bash
python -m repro.derive_k3_disclosures \
  --ensemble /path/to/artifact-root/evaluations/real-oof-ensemble.npz \
  --controls /path/to/delphi-k3-v1-comparator-v1.0.0.npz \
  --downstream-rows /path/to/artifact-root/downstream/fixed-period/fixed-period-rows.json \
  --json-output outputs/publication-disclosures.json \
  --tex-output outputs/k3-disclosures.tex
```

The command derives the non-learned-control contrasts, fixed-work timing
decomposition, absolute recovery, completed-in-both sensitivity, and refinement
effect. It does not retrain a model or rerun inversion.

## Train or reevaluate

Use `delphi-k3 --help` and each subcommand's `--help` for validated arguments.
Run `train-smoke --objects 8 --output outputs/smoke --allow-dirty` to exercise
the small development path; smoke metrics are not scientific benchmarks.

Full training requires the frozen synthetic train/validation/test directories
and raw `damit-20250610T000301Z` dump, with catalog and split hashes matching
the specification. The sequence is:

1. `train-publication-synthetic` for each seed 17, 42, 137, 777, 2027.
2. `train-publication-real-oof` for every fold 0–4 and each seed, using the
   matching synthetic parent checkpoint and frozen split.
3. Run the synthetic/real evaluation, input-swap, label-shuffle and diagnostic
   subcommands with the same declared partitions and model identities.
4. Run OOF and calibration ensemble subcommands, then calibrated-ensemble
   aggregation. Do not transfer single-model calibration to an ensemble.
5. Run `run-publication-fixed-period-benchmark` with the official DAMIT
   convexinv v0.2.1 source archive, matching compiled executable and frozen inputs.
6. Run `build-publication-release` and the figure reproduction command above.

Full training commands require a clean Git checkout and record its commit.
Interrupted CUDA training is not guaranteed to resume bit-exactly; restart a
definitive seed from epoch zero if exact uninterrupted provenance is required.
Do not select checkpoints, hyperparameters or folds on test metrics.

The frozen archive includes broader model-selection controls that are not part
of the K3-only manuscript. Those retained files are provenance evidence, not
additional current-model claims. K3 uses `repro/k3_redesign_spec.yaml`; older
specifications remain only to document the analysis lineage.

## Export trusted evaluated models

```bash
python -m repro.export_k3_bundles \
  --artifact-root /path/to/k3-definitive-7874092 \
  --splits repro/data/damit-20250610T000301Z/publication-splits-v2.3.json \
  --repository . --output outputs/model-bundles
```

The exporter verifies the 25 original checkpoint hashes before loading, checks
tensor-exact safetensors round trips, and emits five fold archives plus SHA256SUMS.
Only use it with trusted local checkpoints. The deployment path never needs
optimizer state or pickle checkpoint files.

## Scope of reproduced analyses

Earlier period-estimation results, K sweeps, low-quality catalog analyses, and
integrated-gradient illustrations are not K3 benchmarks. The primary K3
scope fixes K=3 and requires a known period. Old ZTF products lacking verified
per-observation geometry cannot be passed through K3 with geometry replaced by
zeros. A valid cross-survey study needs real geometry and each object's held-out
fold; it would be a new experiment, not reproduction of the frozen result.

The catalog manifest's `release_eligible: false` and `release_blockers` fields
are frozen internal development gates. They record that an immutable third-party
data URI, independent custodian review, and a prospective temporal test were
not completed at the analysis freeze; they do not describe whether the GitHub
software package can be distributed. The prospective temporal cohort was not
executed and is not a result of the manuscript.

Input-only acquisition and label-separation procedures for the frozen follow-up
are documented in [K3 follow-up external data](followup-data.md). These commands
do not score the follow-up or change the frozen manuscript results.

For scientific interpretation and limitations, cite Jo, Ishiguro and Lee, in prep.

## Verify the complete release package

The versioned GitHub release package is distributed as separate artifact-root,
synthetic-data, source, and retained control-archive files so that each component can
be checked independently. After downloading the complete deposit, run:

```bash
python -m repro.verify_publication_package \
  --package-directory /path/to/downloaded-deposit \
  --extract-to /path/to/new-verification-directory \
  --output /path/to/publication-archive-verification.json
```

The destination must not already exist. The command first checks the outer
download hashes, rejects unsafe archive members, and then verifies the exact
9,797-file scientific index. A successful report does not change or recompute
any published result. The release is a public versioned distribution rather
than a DOI-backed preservation archive.

<!-- BEGIN K3 CONVERGENCE FOLLOW-UP WORKFLOW -->

## Frozen K3 convergence follow-up

The convergence follow-up is a phase-separated new experiment; it does not
replace the fixed-work result in the frozen manuscript. Run it with
`python -m repro.run_k3_convergence_study`. Final JSON outputs must not already
exist. A completed repeat may be resumed only when its plan and all cell hashes
match; a partial repeat must be preserved for audit and restarted in a new
study root. The runner verifies the detached study-spec checksum, exact 30/140
cohort manifests, frozen split and ensemble, official solver archive, matching source
payload and source-tree hash, compiler identity, executable hash, and timing
artifact before accepting work.

Before phase 1, prepare one label-free neural-timing JSON object with exactly
these fields:

```json
{
  "schema": "delphi.k3-convergence-neural-timing.v1",
  "source_ensemble_sha256": "<SHA-256 of the frozen OOF ensemble>",
  "object_ids": ["<all 170 IDs in frozen split order>"],
  "warm_wall_seconds": ["<170 finite positive values>"],
  "cold_wall_seconds": ["<170 finite positive values>"]
}
```

Both vectors are seconds and must be aligned element-for-element with
`object_ids`. Warm timings are used in the primary guided-arm runtime;
cold-start timings are reported as the prespecified sensitivity. Neither this
artifact nor any execution command may contain reference axes or recovery
labels.

For the examples below, define the common, immutable resources once:

```bash
COMMON=(
  --spec repro/k3_followup_study_spec.yaml
  --spec-checksum repro/k3_followup_study_spec.sha256
  --splits repro/data/damit-20250610T000301Z/publication-splits-v2.3.json
  --development-manifest repro/data/k3-followup-20260910/development-objects.json
  --locked-manifest repro/data/k3-followup-20260910/locked-evaluation-objects.json
  --blind-inputs repro/data/k3-followup-20260910/convergence-blind-inputs.json
  --ensemble /path/to/k3-definitive-7874092/evaluations/real-oof-ensemble.npz
  --neural-timing /path/to/frozen-neural-timing.json
  --source-archive /path/to/damit-version_0.2.1.tar.gz
  --source-root /path/to/version_0.2.1/convexinv
  --executable /path/to/version_0.2.1/convexinv/convexinv
)
```

### Phase 1: freeze the pre-execution lock

```bash
python -m repro.run_k3_convergence_study lock "${COMMON[@]}" \
  --output outputs/convergence/study-lock.json
```

The lock is immutable. Any subsequent change to a manifest, source file,
binary, compiler version, timing vector, or other frozen input invalidates it.

### Phase 2: execute and score the development grid

Execution is label blind and runs every frozen tolerance. It writes and hashes
fit selection by minimum final relative RMS before scoring can open the
reference catalog.

```bash
python -m repro.run_k3_convergence_study execute-development-grid \
  "${COMMON[@]}" \
  --lock outputs/convergence/study-lock.json \
  --dump-root /path/to/damit-20250610T000301Z \
  --output-root outputs/convergence/development-executions
```

Then invoke `score-development` once for each of the five reported execution
paths, using a distinct output path:

```bash
python -m repro.run_k3_convergence_study score-development \
  --spec repro/k3_followup_study_spec.yaml \
  --spec-checksum repro/k3_followup_study_spec.sha256 \
  --splits repro/data/damit-20250610T000301Z/publication-splits-v2.3.json \
  --development-manifest repro/data/k3-followup-20260910/development-objects.json \
  --execution /path/reported/by/execute-development-grid/blind-execution.json \
  --reference-catalog repro/data/damit-20250610T000301Z/catalog.jsonl \
  --lock outputs/convergence/study-lock.json \
  --output outputs/convergence/development-scores/tolerance-N.json
```

All six starts in both arms and all three repeats remain in the execution
ledger, including timeouts, parse failures, non-finite outputs, and numerical
failures. Do not reuse timing from a partially completed repeat.

### Phase 3: freeze the development selection

```bash
python -m repro.run_k3_convergence_study select-development \
  --spec repro/k3_followup_study_spec.yaml \
  --spec-checksum repro/k3_followup_study_spec.sha256 \
  --lock outputs/convergence/study-lock.json \
  --score outputs/convergence/development-scores/tolerance-1.json \
  --score outputs/convergence/development-scores/tolerance-2.json \
  --score outputs/convergence/development-scores/tolerance-3.json \
  --score outputs/convergence/development-scores/tolerance-4.json \
  --score outputs/convergence/development-scores/tolerance-5.json \
  --output outputs/convergence/development-selection.json
```

The frozen rule considers all five scores, applies the development eligibility
conditions, maximizes the warm-runtime lower acceptance bound, and breaks ties
toward the smaller tolerance. If no tolerance is eligible, stop: locked
execution is prohibited.

### Phase 4: one-shot locked execution and scoring

Only after phase 3 succeeds, run the locked cohort. The command has no
tolerance option: it reads the hash-bound development selection and revalidates
all five development scores. The selection also fixes the only allowed output
directory and creates an exclusive one-shot claim there. The command still
receives no reference catalog.

```bash
python -m repro.run_k3_convergence_study execute-locked \
  "${COMMON[@]}" \
  --lock outputs/convergence/study-lock.json \
  --dump-root /path/to/damit-20250610T000301Z \
  --development-selection outputs/convergence/development-selection.json \
  --development-score outputs/convergence/development-scores/tolerance-1.json \
  --development-score outputs/convergence/development-scores/tolerance-2.json \
  --development-score outputs/convergence/development-scores/tolerance-3.json \
  --development-score outputs/convergence/development-scores/tolerance-4.json \
  --development-score outputs/convergence/development-scores/tolerance-5.json
```

After that execution artifact is complete and frozen, score it exactly once:

```bash
python -m repro.run_k3_convergence_study score-locked \
  --spec repro/k3_followup_study_spec.yaml \
  --spec-checksum repro/k3_followup_study_spec.sha256 \
  --splits repro/data/damit-20250610T000301Z/publication-splits-v2.3.json \
  --locked-manifest repro/data/k3-followup-20260910/locked-evaluation-objects.json \
  --execution outputs/convergence/locked-evaluation/blind-execution.json \
  --reference-catalog repro/data/damit-20250610T000301Z/catalog.jsonl \
  --lock outputs/convergence/study-lock.json \
  --development-selection outputs/convergence/development-selection.json \
  --development-score outputs/convergence/development-scores/tolerance-1.json \
  --development-score outputs/convergence/development-scores/tolerance-2.json \
  --development-score outputs/convergence/development-scores/tolerance-3.json \
  --development-score outputs/convergence/development-scores/tolerance-4.json \
  --development-score outputs/convergence/development-scores/tolerance-5.json \
  --output outputs/convergence/locked-score.json
```

Recovery and completion are object-level majority-of-three endpoints with all
failures retained. Their noninferiority decision uses the prespecified
simultaneous conservative exact paired bound and a -0.05 margin. RMS uses only
jointly completed/selectable repeat pairs, never reference recovery, requires
at least two joint repeats per object and exact 30/140 object support. Runtime
and RMS bootstrap 5th/95th percentiles are stratified empirical acceptance
bounds, not confidence intervals; the final claim requires every locked gate.

<!-- END K3 CONVERGENCE FOLLOW-UP WORKFLOW -->
