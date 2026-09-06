# Release notes

## v1.0 — 2026-09-07

This is the author-designated final publication release of the DeLPHI K3
software. It closes the independent third-audit compatibility findings by:

- supporting the declared NumPy 1.26 floor throughout both the K3 and retained
  V2 utility modules; and
- limiting the Matplotlib/pyparsing warning exception to the affected
  Matplotlib module without importing a version-specific warning class.

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
- Clarified the fixed-work inversion benchmark and CUDA-versus-CPU numerical
  parity limits in the user and reproduction guides.

The v1.0.1 maintenance release does not retrain the published models or alter
the frozen v1.0.0 scientific predictions.

## v1.0.0 — 2026-09-06

Initial public K3 implementation, evaluated fold bundles, and frozen
publication evidence.
