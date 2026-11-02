# Comparison of complete pole searches (manuscript Section 6.4.2 and Appendix E)

These scripts produced the comparison of a pole search started from the six
signed poles of the three DeLPHI axes with a classical search started from a
fixed list of 12 poles, the cold and loaded timings (Table 4, Figures 13 and
19), and the exploratory comparison with 18 and 96 classical starts
(Tables 5, 12 and 13). Both searches fit the lightcurves with `convexinv`
and keep the fit of lowest RMS residual. The paper uses our estimated
period, with the DAMIT period as a comparison.

The original-period results are in
[`delphi-paper-v1-evidence.tar.gz`](https://zenodo.org/records/23857881/files/delphi-paper-v1-evidence.tar.gz)
in the [v1.1.0 data record, DOI 10.5281/zenodo.23857881](https://doi.org/10.5281/zenodo.23857881),
folder `delphi-paper-v1-evidence/part-b-analyses/workflow-benchmark/`. It holds
the locks, execution summaries, scores and the three historical run notes kept
here. The dated notes describe original-period comparisons, rather than the
own-period medians in the paper.

Use this repository at tag `v1.1.0` for the paper. The retrained
benchmark results, locks, same-PC timing receipts and runner scripts are in
`delphi-retrain-results-v1.1.0.tar.gz`. Direct scores are under
`retrain-20261030/results/delphi-inversions/timing-retime/{own,damit}/`;
the same tree contains ladder, fixed-period and guided-grid retimes
([run labels](../../docs/reproduction.md#run-labels)). Newly written
per-start fits are in `delphi-retrain-benchmark-fits-v1.1.0.tar.gz`.
The unchanged `delphi-own-period-results-v1.0.0.tar.gz` and
`delphi-benchmark-fits-v1.0.0.tar.gz` retain classical and historical evidence;
their older network scores are not the revised results. Set `DELPHI_DATA`
to the extracted `own-period-20261021/` folder and select the retrained networks:

```bash
export DELPHI_NETWORK_ROOT=/path/to/data/retrain-20261030/artifact-root
export DELPHI_MODEL_BINDING="$DELPHI_NETWORK_ROOT/model-binding.json"
export DELPHI_BUNDLE_ROOT="$DELPHI_NETWORK_ROOT/bundles"
```

The [results map](../../docs/reproduction.md#3-where-each-result-comes-from)
describes the corresponding scripts and frozen sources. The direct comparison
runner retimes six DeLPHI starts against 12 classical starts. The
exploratory comparison runners retime the 18- and 96-start ladder,
fixed-period and guided-grid fits. Their scripts and receipts are under
`retrain-20261030/runners/`, with the archive names explained in the
[run labels](../../docs/reproduction.md#run-labels).

## Prerequisites

- The DAMIT lightcurves and model table of June 2025 (see
  [docs/reproduction.md](../../docs/reproduction.md)).
- `convexinv` from DAMIT's `version_0.2.1.tar.gz`, compiled after raising three
  limits in `convexinv/constants.h`: `MAX_LC` to 210 lightcurves, `MAX_N_OBS`
  to 26,612 observations and `POINTS_MAX` to 1,549 observations per
  lightcurve. The fitting algorithm is unchanged.
- The retrained networks in
  [`delphi-k3-artifacts-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/delphi-k3-artifacts-v1.1.0.tar.gz),
  including the binding and fold bundles selected above, and a CUDA GPU for
  the DeLPHI arm.
- The folder layout of [docs/reproduction.md](../../docs/reproduction.md), with
  the released `workflow-benchmark/` folder copied to
  `DeLPHI-followup/data/direct-workflow-benchmark/`.

All timings printed in the paper were measured on one PC with an
Intel Core Ultra 7 270K Plus CPU and an NVIDIA RTX 5070 Ti GPU. The direct
comparison uses six single-threaded solver workers and the settings of
Appendix E. The dated run notes retain old-PC timings from the Core
i5-13600KF and RTX 4070. Timing intervals apply to the measured asteroids on
the stated computer only.

## Historical scripts and order

The sequence below describes the original DAMIT-period comparison. The
retrained comparisons use `rerun.py`, the direct and exploratory timing
runners and their receipts ([run labels](../../docs/reproduction.md#run-labels)).
Retimed ladder scoring uses `score_exploratory_timing_rt5.py` to record
DeLPHI inference drift while retaining the strict classical-fit check.
DeLPHI timing receipts record GPU numerics settings. See
[Run-to-run GPU differences](../../docs/reproduction.md#run-to-run-gpu-differences).
Table and figure numbers below refer to the submitted manuscript (v1.1.0).

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
| 11 | `budget_equivalence.py` | `budget-equivalence.json` | Table 12 |
| 12 | `candidate_accuracy_cost.py` | `candidate-accuracy-cost.json` | Table 13, the 140 asteroids grouped by the oracle error of their DeLPHI candidates |
| 13 | `plot_direct_workflow.py` | `inversion-benchmark.pdf` | Figure 13 |

Steps 7 to 12 were decided after the scores of step 4 were known and are
exploratory, as stated in Appendix E.

## Rerunning the original-period comparison

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
  full 146-start search, which are not in the unchanged evidence archive. A new run on another
  computer needs new locks made with steps 1 and 2 and gives new times.

Every script refuses to overwrite an existing output.

## Run notes

- `direct-benchmark-results-20260923.md`: results of the 12-start comparison.
- `timing-interruption-20260923.md`: the 32 cold searches set aside after
  another job ran on the computer, and their reruns.
- `exploratory-ladder-results-20260925.md`: results of the 18- and 96-start
  comparison.

These are records written at the time of the runs. Their numbers are those of
the original-period scored files. The paper's own-period results and medians
are mapped separately in the reproduction guide.

## Provenance

Every execution record stores the SHA-256 of the script that wrote it. The
scripts of steps 1 to 10 are byte-identical to the copies that were run. The
copies of `budget_equivalence.py`, `candidate_accuracy_cost.py` and
`plot_direct_workflow.py` differ from the run copies only in reading the
records from `DeLPHI-followup/data/direct-workflow-benchmark/` instead of their
own folder, and in the later restyling of Figure 13.
