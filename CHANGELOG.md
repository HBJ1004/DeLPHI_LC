# Release notes

## 1.1.0 (2026-11-01)

- Retrained all 25 ensemble networks after correcting reused simulation seeds
  that linked the pole to the period and initial phase.
- Reran all network results and retimed all reported comparisons on one PC.
- Review fixes (d0f586a): record GPU numerics settings in timing receipts,
  check V6 files before resuming, validate scorer inputs, and include the
  RT5 timing scorer in the repository.
- The wheel and source archive now carry every module and data file that the
  commands use.
- The released figure command redraws every printed figure, with pole-fit
  times taken from the same-PC retimed scores.
- Timing receipts record `DELPHI_SCORE_CHUNK_MAX`, transient GPU errors are
  retried on resume instead of being sealed as failures, and the synthetic
  data generator no longer has machine-specific default paths.
- New data record version 1.1.0 (DOI 10.5281/zenodo.23857881) with the
  retrained networks, all results and the historical source snapshots.

## 1.0.0 (2026-10-29)

DeLPHI estimates the rotation period with its own Fourier search and proposes
three unranked pole-axis candidates from asteroid lightcurves, the observing
geometry and that period (or a supplied one), for use as six signed
starting poles in convex lightcurve inversion. This is the software behind
Paper I as revised.

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
