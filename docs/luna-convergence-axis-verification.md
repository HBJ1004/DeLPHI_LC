# Luna verification of the convergence-axis correction

This is the independent verification pass for Packet A. It does not execute a
solver, open the locked cohort, alter archived outputs, or read reference axes.

## Mathematical check

The native `blmatrix.c` implementation sets its third row to

```text
(sin(beta) cos(lambda), sin(beta) sin(lambda), cos(beta)).
```

Before converting to radians, `convexinv.c` replaces the internal `beta` by
`90 - printed_beta`. Therefore the direction represented by the two printed
values is

```text
(cos(printed_beta) cos(printed_lambda),
 cos(printed_beta) sin(printed_lambda),
 sin(printed_beta)).
```

This is a directed vector. A printed latitude outside the conventional range
must be reduced through the trigonometric representation and then converted to
standard coordinates; it must not be clipped. For example, `(10, 100)` degrees
is the same direction as `(190, 80)` degrees, not `(10, 90)` and not the
antipode.

The independent tests implement this matrix translation directly. They do not
call the decoder to calculate the expected vector.

## Implementation reviewed

The decoder is `lc_pipeline/k3/convergence_axis.py` and the integration points
are `_completion`, `_selectable`, selected-fit construction, and scoring. The
decoder rejects nonfinite values, strings, and booleans; returns raw and
standard angles, a normalized directed vector, and an interpretation version;
and does not mutate its inputs. Selection and scoring use the same decoded
vector. Raw solver records remain unchanged.

The independent test file is
`tests/test_k3_convergence_axis_independent.py`. It covers:

- wrapped and out-of-range latitude examples, including the recorded solver
  example;
- north and south poles, where the vector is checked rather than longitude;
- 128 deterministic full-turn random angle pairs;
- nonfinite, string, boolean, and missing-value rejection; and
- input-record immutability and interpretation-version presence.

The existing Terra tests additionally cover completion rejection for timeout,
nonzero exit, missing iteration log, iteration cap, invalid RMS, invalid
period, invalid angle, invalid output hash, and invalid output products.

## Verification command

Using the project Python environment:

```text
python -m pytest -q tests/test_k3_convergence_axis_independent.py \
  tests/test_k3_convergence_axis.py \
  tests/test_k3_convergence_benchmark.py \
  tests/test_k3_convergence_study.py \
  tests/test_k3_convergence_timing.py
```

Result at the verification pass: **84 passed**. Ruff and `git diff --check`
also passed for the independent test file. No benchmark or replay was run.

This verifies the interpretation and its current in-memory integration. It does
not certify the archived-development replay; that requires Packet B's separate
provenance and phase-separation checks.
