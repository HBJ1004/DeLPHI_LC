# Reproducing results and figures

The measured experiment is bound to training commit
`7874092b81d9456e5bf52fba9adb699b8218046f` and protocol SHA-256
`84dee816d08a60b7780a6d1d400add16136fa96c0677e84929700cee8bf22a50`.
An export commit identifies packaging/interface changes separately from training.
Do not relabel a new training run as the frozen experiment.

## Rebuild from frozen predictions (CPU)

Required external artifacts: the complete `k3-definitive-7874092` archive,
`v1-comparators.npz`, and the catalog included in this repository. A checkout
alone does not contain training weights, raw DAMIT photometry or the full archive.
The output directory must not already exist.

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

V1 comparator orchestration is in `lc_pipeline.publication.runner`, with model
implementation in `lc_pipeline.v2.baselines`. The frozen V1 contract also requires
its external cache files. The shipped cache manifest is provenance metadata,
not the cache itself. `repro/release_spec.yaml` describes the older V2 experiment;
K3 uses `repro/k3_redesign_spec.yaml` and its own gates.

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

Historical V1 period-estimation results, K sweeps, low-quality catalog analyses,
and integrated-gradient illustrations are not K3 benchmarks. The primary K3
scope fixes K=3 and requires a known period. Old ZTF products lacking verified
per-observation geometry cannot be passed through K3 with geometry replaced by
zeros. A valid cross-survey study needs real geometry and each object's held-out
fold; it would be a new experiment, not reproduction of the frozen result.

For scientific interpretation and limitations, cite Jo, Ishiguro and Lee, in prep.
