# Solver internal-capacity correction

The historical convergence solver allocates fixed arrays from `constants.h`.
After it has accepted native lightcurve input, both `convexinv` and
`period_scan` append one regularization lightcurve containing three
observations.  A native input at a declared limit is therefore not safe at the
same allocation limit.

The correction reserves this internal append before admitting an input:

```
POINTS_MAX = max(native maximum points, 3)
MAX_N_OBS = native total observations + 3
MAX_LC = native lightcurve count + 1
```

The padding version is `convexinv-regularization-append-v1`.  A receipt that
does not bind this version and these effective requirements is not safe for a
new locked execution.  Earlier capacity receipts are historical records only;
they must not be interpreted as an approval for the corrected solver.

The correction is create-once.  It copies the supplied solver source to a new
directory, patches only these three defines, and builds both `convexinv` and
`period_scan`.  It does not modify the supplied source, existing solver
copies, frozen inputs, weights, or results.  It runs one small supported-input
parity check and ASAN/UBSAN checks on the two known boundary inputs.  Those
checks are capacity checks, not a convergence benchmark.

Run from the source repository with the supported interpreter:

```bash
python -m repro.prepare_k3_solver_capacity \
  --blind-inputs repro/data/k3-followup-20260910/convergence-blind-inputs.json \
  --development-manifest repro/data/k3-followup-20260910/development-objects.json \
  --locked-manifest repro/data/k3-followup-20260910/locked-evaluation-objects.json \
  --dump-root ../../damit-20250610T000301Z \
  --solver-root ../inputs/solver/version_0.2.1 \
  --output-root ../data/solver-internal-capacity-corrected/20260911
```

The report is `solver-capacity-preflight.json` under the new output root.  It
records raw structures, effective internal requirements, the padding version,
the hashes of both normal binaries, the sanitizer-binary hashes, and the
parity and sanitizer results.  The dated output root must not be reused or
overwritten.
