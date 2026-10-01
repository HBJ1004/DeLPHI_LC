# Release notes

## paper-v1 (software version 1.1) — 2026-10-01

The code and evidence for the paper (Jo, Ishiguro and Lee, "DeLPHI: Pole-Axis
Candidates for Asteroid Lightcurve Inversion", submitted to PSJ). The trained
networks and their test predictions are unchanged from `v1.0.0`; this release
adds everything else the paper reports.

- Evidence archive `delphi-paper-v1-evidence.tar.gz`, whose README lists the
  files behind each section of the paper.
- Analysis scripts under `repro/` for the inversion comparison, the
  interpretation of the score maps, the reduced-data tests, the experimental
  period search, and the DAMIT census.
- Rewritten user guides: a first-prediction walk-through with a worked example,
  a DAMIT converter, geometry from JPL Horizons, an explanation of every output
  field, and a reproduction guide organized by paper section.
- A table of the 170 benchmark asteroids with the cross-validation run in which
  each is a test asteroid (`repro/data/benchmark-asteroids.csv`), and a clearer
  error when a benchmark asteroid is given to a network set that trained on it.
- Internal development logs were removed from `docs/`; they remain in the git
  history and in the earlier tags.

All releases before `paper-v1`, except the trained networks in `v1.0.0`, are
superseded.

## v1.0 — 2026-09-07 (superseded by paper-v1)

A maintenance release of the DeLPHI K3 software. It closes the independent
third-audit compatibility findings by:

- supporting the declared NumPy 1.26 floor throughout both the K3 and retained
  V2 utility modules; and
- limiting the Matplotlib/pyparsing warning exception to the affected
  Matplotlib module without importing a version-specific warning class; and
- supporting gradient scaling across the declared PyTorch 2.2--2.x range.

The earlier `v1.0.0` scientific evidence release and `v1.0.1` audit candidate
remain available as immutable provenance. This release does not retrain the
published models or alter the frozen scientific results.

## v1.0.1 — 2026-09-06

- Restored the declared NumPy 1.26 compatibility and added a minimum-dependency
  CI job.
- Added safe, checksum-bound inference bundles for models trained with
  `repro.train_k3_custom`; the standard prediction command now loads published
  and custom bundles.
- Marked the three returned candidates as an unordered axial proposal set and
  clarified that internal scores are not a validated physical-pole ranking.
- Converted malformed prediction inputs into concise command-line errors.
- Added a reproducible derivation of control, fixed-work timing, recovery,
  sensitivity, and refinement disclosures from the frozen evidence.
- Expanded tests for arbitrary 3-D rotation invariance, bundle tampering,
  command-line validation, and disclosure provenance.
- Clarified the fixed-work inversion benchmark (later superseded by the
  complete pole-search comparison of the paper) and CUDA-versus-CPU numerical
  parity limits in the user and reproduction guides.

The v1.0.1 maintenance release does not retrain the published models or alter
the frozen v1.0.0 scientific predictions.

## v1.0.0 — 2026-09-06

Initial public K3 implementation, evaluated fold bundles, and frozen
publication evidence.
