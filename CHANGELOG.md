# Release notes

## 1.2.0 (2026-11-02)

No numerical results change.

- Remove unused inference, tensor-cache, loss and figure code and unreferenced reproduction scripts.
  The removed scripts remain available at tag v1.1.0.
- Keep the existing K3 result types as deprecated imports and export `load_predictor` and `read_observations`.
- Share equivalent SHA-256 and canonical JSON helpers while preserving their formats and errors.
- Remove no-op exception wrappers and duplicate seed definitions.
- Keep `risk_deg` as a reserved prediction field with value `null`.
- Correct docstrings, help and error text, and remove development notes from report text.
- Clarify installation constraints, packaged split data and prediction provenance fields.
- Merge the results headings and explain historical statistic names and run labels.
- Correct configuration guidance and name the commit at which the deposited locks verify.
- Simplify citation wording and release notes.

## 1.1.1 (2026-11-02)

No numerical results change.

- Support PyTorch 2.2 out-of-memory handling and default shipped commands to the
  caller's available CPUs, recording actual affinity in reports.
- Make CPU-affinity tests portable and skip worker-count cases when CPUs are
  unavailable. Pin Ruff for lint checks and clarify the historical constraints.
- Require explicit opt-in for historical v1.0.0 networks trained on simulations
  that reused random seeds. Add software version, training commit and network
  hashes to predictions.
- Distinguish the paper's v1.1.0 software DOI from the concept DOI for the latest version.
- Add the disclosed overlap between simulated training, validation and test objects.
- Guard historical DeLPHI helpers, select current coverage and timing inputs, document stale figures and their redraw commands, and keep figure reports inside the output folder.
- Document the compatible GCC solver build and find its test fixture through `DELPHI_DATA`.
- Clarify documentation and metadata wording.

## 1.1.0 (2026-11-01)

- Retrained all 25 ensemble networks after correcting reused simulation seeds
  that linked the pole to the period and initial phase.
  Pole, period, phase and noise use independent random streams within each object.
  Each simulated validation and test object shares its pole, period, phase and
  noise draw with a training object, with different observing geometry and shape.
- Reran all network results and retimed all reported comparisons on one PC.
- Record GPU numerics settings in timing receipts, check fixed-period
  inversion files before resuming, validate scorer inputs, and include the
  retimed exploratory-ladder scorer in the repository.
- The wheel and source archive now carry every module and data file that the
  commands use.
- The released `median_figures` command redraws its 13 figures with the
  manuscript folder required for its checks and pole-fit times taken from the
  same-PC retimed scores. Period-search and map figures have separate commands.
- Timing receipts record `DELPHI_SCORE_CHUNK_MAX`, transient GPU errors are
  retried on resume instead of being sealed as failures, and the synthetic
  data generator no longer has machine-specific default paths.
- New data record version 1.1.0 (DOI 10.5281/zenodo.23857881) with the
  retrained networks, all results and the historical source snapshots.

## 1.0.0 (2026-10-29)

DeLPHI estimates the rotation period with its own Fourier search and proposes
three unranked pole-axis candidates from asteroid lightcurves, the observing
geometry and that period (or a supplied one), for use as six signed
starting poles in convex lightcurve inversion. This is the initial software
release for the paper. Version 1.1.0 supersedes its networks and results.

- K3 scoring, simulated pretraining, real-data training, prediction, safe
  inference bundles, custom-data training and protocol checks.
- Own-period predictions and inversion benchmarks; score-map, candidate-count,
  lower-quality census, reliability, network and withheld-lightcurve analyses;
  median statistics and figures, with reproduction guides and automated tests.
- Code at tag `v1.0.0`; trained networks, synthetic data, predictions, results
  and saved benchmark fits in the single [Zenodo data record,
  DOI 10.5281/zenodo.23757942](https://doi.org/10.5281/zenodo.23757942).

Earlier development releases (1.0, 1.0.1, 1.1/paper-v1 and the dated evidence
releases) were withdrawn and superseded by this release.
