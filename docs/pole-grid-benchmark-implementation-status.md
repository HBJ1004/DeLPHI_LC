# Implementation and execution status — 2026-09-11

The known-period grid benchmark is implemented. The first five-object pilot
completed, projected 318,095 seconds (88.4 hours), and correctly stopped under
its locked 24-hour ceiling. The full 170-object experiment and its scientific
decision are not complete. This file is a dated status record, not a claim that
a later execution succeeded.

## Completed checks

- Internal native capacity corrected to `MAX_LC=210`, `MAX_N_OBS=26612`, and
  `POINTS_MAX=1549`. Both `convexinv` and `period_scan` passed ASAN/UBSAN checks
  on asteroids 109 and 2512. The supported asteroid 227 check matched the
  official solver's output hashes.
- Historical capacity-v1 receipts cannot authorize a new execution. The older
  convergence report now warns that its timings and tolerance selection used
  unsafe internal capacities and are not valid acceleration evidence.
- Full repository test run: 413 passed. After the final validation additions,
  the combined capacity/grid/scoring/import tests passed: 80 passed. Repository
  Ruff check passed. Tests were completed before timing began.
- A synthetic-only CUDA warmup succeeded using the released fold-0 ensemble.
  No evaluation object was used for this warmup.
- Independent tests cover grid cardinality, signed-axis geometry, candidate
  sign/order invariance, cap boundaries, duplicate removal, invalid native
  outcomes, artifact tampering, setup evidence, deadline handling, and scoring
  contracts. Cold classical imports do not load Torch.

## Frozen execution

Output root, relative to the source directory:

`../data/pole-grid-benchmark/20260911-internal-capacity-v2/`

Lock SHA-256:

`bbc8bc676f25db49576f4cf4f3a2e4d16acdcbc72c921655e9452551cd2c50ca`

Native-capacity report SHA-256:

`9070f5a4e776bfd8a88e6615a2df2012c5dcd156c7cfa595b0b913cc16c183d0`

Pilot objects in outer-fold order: `asteroid_1261`, `asteroid_3415`,
`asteroid_2513`, `asteroid_1715`, `asteroid_176`. These were selected by a fixed
hash rule from the original 30-object development cohort, not by pilot results.
The experiment has 150 pilot cases and 5,100 full-cohort cases.

The execution process retained each start's input parameters, native products,
logs and checksums, as well as label-blind selected fits and directly measured
case wall times. The original continuation wrote `pilot-score.json` and
`continuation-stop.json`; it did not launch a full experiment. The exact v1
runtime source was archived as `runtime-source-v1.tar.gz` with SHA-256
`cf1c44907c01e875473bd86b5eceb7bdbfd2f089432855a1d2ec29d00bf152e5`.

The user subsequently authorized a separate v2 execution with a seven-day hard
ceiling. Its 150-case pilot completed and projected 312,627 seconds (86.8
hours), within that ceiling. Before the full execution started, the system
Python executable changed and the frozen timing environment check refused to
continue. The v2 pilot remains retained. A v3 environment-relocked protocol
will use a benchmark-owned interpreter and repeat the engineering pilot. Its
scientific endpoints and decision criteria are unchanged. Its continuation runs
the full blind execution before reference scoring. An interrupted execution has
an `execution-interrupted.json` receipt, not a scoreable complete receipt.

Do not change the bound runtime source files while either run is active. A
necessary implementation correction requires a new lock and new output root.
No manuscript, Overleaf project, published weight, release tag, package version,
or historical result artifact was changed by this implementation.

See [the protocol and runbook](known-period-grid-benchmark.md) for endpoints,
timing boundaries, known-period scope, and acceptance criteria. Existing
negative external-transfer evidence is unchanged; this experiment is not a
substitute for external validation.
