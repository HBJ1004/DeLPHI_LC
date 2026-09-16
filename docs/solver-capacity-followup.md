# Solver-capacity correction for the convergence follow-up

The frozen 170-object convergence design cannot be executed with the original
`convexinv` v0.2.1 static input dimensions.  This is an input-capacity
finding, **not** evidence that no convergence tolerance works.  No object is
silently removed to make the RMS support requirement appear satisfied.

The original `convexinv/constants.h` checks `MAX_LC=100`,
`MAX_N_OBS=10000`, and `POINTS_MAX=1000`.  `conjgradinv/constants.h` was also
inspected (`MAX_LC=200`, `MAX_N_OBS=10000`, `POINTS_MAX=1000`), but the frozen
protocol names `damit_convexinv_0.2.1`; only its copied source is rebuilt.

## Recorded preflight result

`../data/solver-capacity-revision/solver-capacity-preflight.json` was built
from the frozen blind input's lightcurve paths and hashes only.  It does not
read reference solutions, K3 axes, prediction outcomes, score files, or any
locked solver result.
The audit verifies the frozen internal-DAMIT mapping
`files/<object_id>/lc.txt` for every blind row; these identifiers are not
treated as external catalog numbers.

| Cohort | Required RMS support | Fits original static limits | Result |
| --- | ---: | ---: | --- |
| Development | 30 | 25 | blocked |
| Locked evaluation | 140 | 119 | blocked |

The five development and 21 locked inputs that exceed at least one original
limit are enumerated in the report with structural counts and macro names.
The report fails closed: both full declared supports must fit before either
cohort is authorized.  The observed all-cohort maxima are `MAX_LC=209`,
`MAX_N_OBS=26609`, and `POINTS_MAX=1549`.

## Bounded replacement binary

The preparation command creates a new copy at
`../data/solver-capacity-revision/expanded-solver/convexinv`; it never builds,
cleans, patches, or overwrites `../inputs/solver/version_0.2.1`.  Only the
three static bounds above are changed, and they are increased only to the
metadata maxima (never reduced).  The report records the original and copied
source-tree hashes, expanded binary hash, compiler path/version/hash, and a
canonical patch hash.  The same recorded expanded `convexinv` binary must be
used by both baseline and guided arms.

The expanded executable's measured static size is 43,224 text bytes, 800 data
bytes, and 7,648,528 BSS bytes.  Its observation buffers remain dynamically
allocated and scale with `MAX_N_OBS`; run solver processes serially and retain
the report's stack-limit and binary-size values in a revised execution lock.
The worst-size check is structural only at the three maxima—no synthetic
photometry and no locked solver execution was performed.

A two-iteration, original-supported development smoke (`asteroid_227`) ran
the original and copied binaries with a fixed official `convexinv` parameter
file.  Return code and hashes of stdout, stderr, modelled lightcurve, solution
parameters, and areas matched.  Raw smoke products are discarded; only hashes
are retained.  This confirms supported-input parity for the bounded source
change, not scientific equivalence at every execution condition.

A separate two-iteration development-only check used `asteroid_3415`, whose
169 lightcurves exceed the original 100-lightcurve limit. The expanded binary
completed with return code zero. Its inputs, output products and hashes are
retained under `../data/solver-capacity-revision/high-index-development-smoke/`.
This checks one high-index input, not every maximum-size boundary, convergence,
or speed. The 140 locked objects have not been executed.

## Revised execution lock

The phase-separated runner now accepts a capacity revision only when the
preflight report binds the requested copied source and binary, the original
archive remains the base source, the report proves all 30 development and all
140 locked objects fit structurally, and the archived source differs only in
the three static capacity definitions. The default path remains strict archive
equality. A report hash, patch hash, capacities, binary hash, compiler and
current copied-tree hash are included in the lock.

`../data/solver-capacity-revision/revised-convergence-study-lock.json` was
created without solver execution. Its SHA-256 is
`6388670f1b3c49862176ebcab5669c86d662e8854536b6328069213315fd9e8a`.
It permits the prescribed label-blind development grid to use the expanded
binary. It does not select a tolerance, open a reference catalog, execute a
locked object, or establish acceleration.

## Reproduction

Run from `DeLPHI-followup/source` with the focused environment:

```bash
python -m repro.prepare_k3_solver_capacity \
  --blind-inputs repro/data/k3-followup-20260910/convergence-blind-inputs.json \
  --development-manifest repro/data/k3-followup-20260910/development-objects.json \
  --locked-manifest repro/data/k3-followup-20260910/locked-evaluation-objects.json \
  --dump-root ../../damit-20250610T000301Z \
  --solver-root ../inputs/solver/version_0.2.1 \
  --output-root ../data/solver-capacity-revision \
  --compiler cc
```

The output root is intentionally create-once.  A pre-existing output path, an
output nested inside the original source tree, an altered lightcurve hash, a
missing cohort object, a malformed lightcurve frame, a failed compile, failed
parity smoke, or insufficient expanded support stops the command.  Do not use
the locked-execution command until the revised protocol explicitly binds the
expanded source, binary, compiler, patch, and this capacity report.

For the later development-only grid, provide the same report explicitly with
`--capacity-revision ../data/solver-capacity-revision/solver-capacity-preflight.json`
and the revised lock. The locked command remains unavailable until all five
development tolerances have been executed, scored, and selected by the frozen
rule.

## Superseded pre-capacity development record

An earlier development directory exists under `../data/convergence/`. Its
selection artifact has status `no_eligible_tolerance_stop`; its reported warm
runtime lower bounds range from 0.487 to 0.908, so it does not support an
acceleration claim and it correctly did not open the locked cohort. That
record is not a valid result for the revised execution design. Its lock binds
source-tree SHA-256 `18389707...` and executable SHA-256 `7965542f...`, whereas
the capacity-corrected lock binds `aaf95c46...` and `d3fa5929...`. The old
source cannot satisfy the current full-cohort structural capacity check.

The old record is retained as historical evidence of a stopped development
attempt. It must not be pooled with or substituted for a capacity-corrected
development grid. No locked acceleration result has been executed.

## Capacity-corrected development result

The capacity-corrected, label-blind development grid was completed on 30
development objects at the five locked tolerances 0.01, 0.003, 0.001, 0.0003,
and 0.0001, with three repeat cells per object and arm. All 450 execution
cells were retained. The five score artifacts were then supplied together to
the frozen selection rule in
`../data/convergence-capacity-corrected/development/selection.json` (SHA-256
`68bddcea938fdfef9a61347de7eac6611531e86a3122151ee3984d1e7e0b7d84`). Its
status is `no_eligible_tolerance_stop`.

None of the tolerances met every predeclared safeguard. The warm runtime
acceptance lower bounds were 0.673, 0.699, 0.784, 0.936, and 1.031,
respectively as the tolerance decreased. However, the RMS comparison required
jointly completed and selectable fits for all 30 development objects; the
available support was 29, 27, 26, 25, and 24 objects, respectively. The 0.003
and 0.0001 tolerances also missed the completion-difference safeguard.

Accordingly, the selection rule did not select a tolerance and did not
authorize the 140-object locked evaluation. The locked-evaluation directory is
empty. This result does not establish acceleration, and it should not be
combined with the fixed-work manuscript timing benchmark or presented as a
locked-cohort speed result.
