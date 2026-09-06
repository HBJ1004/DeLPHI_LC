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
