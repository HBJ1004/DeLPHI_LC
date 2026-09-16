# Luna verification of internal solver-capacity requirements

This is the independent review of the capacity correction. It does not modify
the native solver, execute the real cohort, or run the sanitizer boundary
checks. The only executable checks in this pass are disposable unit fixtures.

## Native-source basis

Both `convexinv.c` and `period_scan.c` use the three input limits while reading
the input stream. After that read, both programs append a regularization
lightcurve:

```text
Lcurves = Lcurves + 1;
Lpoints[Lcurves] = 3;
ndata++;  /* repeated three times */
```

The arrays are statically declared as `Lpoints[MAX_LC+1]` and the observation
arrays are sized from `MAX_N_OBS`. The safe admission requirements are therefore
`input lightcurves + 1`, `input observations + 3`, and
`max(input points, 3)`. The same consumers occur in both source programs; no
additional input-dimension macro was found in the native `convexinv` or
`period_scan` sources. Other macros (`MAX_N_FAC`, `MAX_N_PAR`, `MAX_LM`, and
`MAX_N_ITER`) govern fixed model or iteration allocations and are not derived
from the lightcurve input census.

For the current declared cohort maxima (209 lightcurves, 26,609 observations,
and 1,549 points in one lightcurve), the minimum safe capacities are therefore

```text
MAX_LC = 210
MAX_N_OBS = 26612
POINTS_MAX = 1549
```

The lower bounds are strict: exact raw-input limits are unsafe because the
three-point row and one additional lightcurve are appended after parsing.

## Independent tests

`tests/test_k3_solver_capacity_independent.py` tests the implementation at the
exact cohort boundary, one below each safe bound, the small-input three-point
case, and parser framing/truncation. It does not depend on the implementation's
capacity formula to calculate an expected value. It also checks that the parser
reports raw input counts separately from the later internal padding.

Verification with the supported interpreter:

```text
python -m pytest -q \
  tests/test_k3_solver_capacity_independent.py
```

Result: **8 passed**. Ruff and `git diff --check` pass for the independent test
file.

The existing capacity test module was also inspected. Its real-source test
fixture now includes a `period_scan` target and disables the sanitizer for the
small fixture. Any remaining expected values for the old raw-only limits must be
updated by the implementation owner to reflect the internal padding; this
review did not edit that file.

## Remaining evidence needed

The independent tests establish the arithmetic and the two native consumers.
They do not establish that the rebuilt binaries are safe at the real maxima.
That requires one recorded ASAN/UBSAN run for the two boundary inputs named in
the handoff (`asteroid_2512` and `asteroid_109`), followed by source-tree and
binary hash recording. The full convergence benchmark remains a separate,
authorized step.
