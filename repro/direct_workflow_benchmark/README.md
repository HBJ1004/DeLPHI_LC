# Comparison of complete pole searches (manuscript Section 6.4.2 and Appendix E)

These scripts produced the comparison of a pole search started from the six
signed poles of the three DeLPHI axes with a classical search started from a
fixed list of 12 poles, the cold and loaded timings (Table 3, Figure 13), and
the exploratory comparison with 18 and 96 classical starts (Tables 4, 10 and
11). Both searches fit the lightcurves with `convexinv` at the published DAMIT
period and keep the fit of lowest RMS residual.

The results are in release
[`paper-v1`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/paper-v1),
archive `delphi-paper-v1-evidence.tar.gz`, folder
`part-b-analyses/workflow-benchmark/`. That folder holds the locks, the
execution summaries, the scored files and the three run notes kept here. The
raw `convexinv` fits of every start (2.6 GB) are not in the release and are
available from the authors on request.

## Prerequisites

- The DAMIT lightcurves and model table of June 2025 (see
  [docs/reproduction.md](../../docs/reproduction.md)).
- `convexinv` from DAMIT's `version_0.2.1.tar.gz`, compiled after raising three
  limits in `convexinv/constants.h`: `MAX_LC` to 210 lightcurves, `MAX_N_OBS`
  to 26,612 observations and `POINTS_MAX` to 1,549 observations per
  lightcurve. The fitting algorithm is unchanged.
- The trained networks of release `v1.0.0` (`k3-oof-fold-0.tar.gz` to
  `k3-oof-fold-4.tar.gz`) and a CUDA GPU for the DeLPHI arm.
- The folder layout of [docs/reproduction.md](../../docs/reproduction.md), with
  the released `workflow-benchmark/` folder copied to
  `DeLPHI-followup/data/direct-workflow-benchmark/`.

All runs of the paper used one workstation with an Intel Core i5-13600KF CPU
under WSL2 and an NVIDIA RTX 4070 GPU, six single-threaded solver workers, and
the solver settings of Appendix E. The measured times are those of the paper
and of the scored files. They apply to that computer only.

## Scripts and order

| Step | Script | Writes | Role |
|---|---|---|---|
| 1 | `run_direct_benchmark.py development` | `development-report.json` | Chooses the classical budget, the smallest of ten budgets (6 to 146 starts) that met the quality limits on the 30 development asteroids, from saved fits |
| 2 | `run_direct_benchmark.py freeze` | `evaluation-lock.json`, `scoring-lock.json` | Writes the lock files, which fix the 140 evaluation asteroids, the starts, the solver, the networks and the computing environment, with their checksums, before any search is run |
| 3 | `run_direct_benchmark.py execute` | `evaluation/` | Cold timing: a fresh program for every search, three repeats per asteroid and arm (840 searches), without access to the DAMIT poles |
| 4 | `run_direct_benchmark.py score` | `score.json` | Opens the DAMIT poles and computes agreement, residuals, time ratios and bootstrap intervals |
| 5 | `run_loaded_benchmark.py` | `evaluation-loaded/` | Loaded timing: the networks are loaded once per cross-validation run (840 searches) |
| 6 | `run_direct_benchmark.py score` | `score-loaded.json` | Step 4 for the loaded run |
| 7 | `explore_classical_ladder.py` | `classical-ladder-exploratory.json` | Best fit among the first 6 to 146 starts of the fixed classical order, rebuilt from the saved fits of the full 146-start search |
| 8 | `score_classical_ladder.py` | `classical-ladder-intervals.json` | Paired bootstrap intervals for every classical budget |
| 9 | `time_exploratory_budgets.py` | `exploratory-loaded-ladder/` | Loaded timing of DeLPHI and of 18 and 96 classical starts, one repeat |
| 10 | `score_exploratory_timing.py` | `exploratory-loaded-ladder/score.json` | Checks that the timed fits equal the saved fits and summarizes the times |
| 11 | `budget_equivalence.py` | `budget-equivalence.json` | Table 10 |
| 12 | `candidate_accuracy_cost.py` | `candidate-accuracy-cost.json` | Table 11, the 140 asteroids grouped by the oracle error of their DeLPHI candidates |
| 13 | `plot_direct_workflow.py` | `inversion-benchmark.pdf` | Figure 13 |

Steps 7 to 12 were decided after the scores of step 4 were known and are
exploratory, as stated in Appendix E.

## What can be rerun from the release

- Steps 8, 11 and 13 need only released files and run on a CPU.
  `budget_equivalence.py` and `plot_direct_workflow.py` run from this folder of
  the repository and read `DeLPHI-followup/data/direct-workflow-benchmark/`.
  `plot_direct_workflow.py` writes `paper/figures/inversion-benchmark.pdf` in
  the folder that contains `DeLPHI-followup/`, which must exist.
  `score_classical_ladder.py` reads and writes next to itself, so copy it into
  `DeLPHI-followup/data/direct-workflow-benchmark/` and run it there.
- Steps 10 and 12 also read the per-search records under `cases/`, which are
  part of the raw fits.
- Steps 1 to 7 and 9 cannot be rerun as they stand. The scripts read and write
  next to themselves and were run as copies inside
  `DeLPHI-followup/data/direct-workflow-benchmark/`. The lock files store
  absolute paths on the authors' computer and the checksums of the Python
  interpreter, hardware description, solver, lightcurves and networks, and
  every run checks them. Step 1 also reads an earlier lock and the saved fits of the
  full 146-start search, which are not in the release. A new run on another
  computer needs new locks made with steps 1 and 2 and gives new times.

Every script refuses to overwrite an existing output.

## Run notes

- `direct-benchmark-results-20260923.md`: results of the 12-start comparison.
- `timing-interruption-20260923.md`: the 32 cold searches set aside after
  another job ran on the computer, and their reruns.
- `exploratory-ladder-results-20260925.md`: results of the 18- and 96-start
  comparison.

These are records written at the time of the runs. Their numbers are those of
the scored files and of the paper, apart from the rounding noted at the top of
the exploratory note.

## Provenance

Every execution record stores the SHA-256 of the script that wrote it. The
scripts of steps 1 to 10 are byte-identical to the copies that were run. The
copies of `budget_equivalence.py`, `candidate_accuracy_cost.py` and
`plot_direct_workflow.py` differ from the run copies only in reading the
records from `DeLPHI-followup/data/direct-workflow-benchmark/` instead of their
own folder, and in the later restyling of Figure 13.
