# Checking the results of the paper

This guide is for readers who want to check the numbers, tables and figures of
Jo, Ishiguro and Lee, "DeLPHI: Neural-Network Pole Candidates for Asteroid
Lightcurve Inversion" (submitted to PSJ). It explains what to download, how to
recompute the main results from the released predictions without rerunning any analysis,
which file and script lie behind each result, and how to arrange the files to
rerun the analyses.

**Shortest route.** Use this repository at tag `v1.1.0` and the
[Zenodo data record v1.1.0, DOI 10.5281/zenodo.23857881](https://doi.org/10.5281/zenodo.23857881).
Download the retrained results and `SHA256SUMS` using the
[minimal download](#minimal-download-for-check-a), then follow
[Install](#install) and [Check A](#check-a-paper-i-own-period-results).
The other archives support the detailed comparisons and reruns below.

A few terms are used throughout.

- **Oracle error**: for one asteroid, the smallest angle between any of the
  three DeLPHI candidate axes and any of its DAMIT reference poles, with the
  two directions along an axis treated as equal (paper Section 5.1).
- **Test asteroid**: each of the 170 asteroids is analyzed only with the
  networks of the one cross-validation run that held it out of training. File
  names call these results `oof` (out-of-fold).
- **K3** is the name of the DeLPHI model in the code and in file names (`k3-`,
  `delphi-k3`); it refers to the three candidate axes.

## Paper I results

The code at tag `v1.1.0` provides the revised Paper I analyses with our own
rotation periods and the 25 networks retrained on corrected simulations.
Data, predictions and fits are in the
[Zenodo data record v1.1.0](https://doi.org/10.5281/zenodo.23857881). The corrected simulations
use independent random streams for the pole, period, initial phase and noise.
`retrain-20261030/PROTOCOL.md` and `RUNLOG.md` record the retraining, reruns and
same-PC retimes. The unchanged `own-period-20261021/PROTOCOL.md` and protocols in `census7/`,
`refine-chain/`, `period-cost/` and `spacecraft/` record the fixed choices,
validation gates and later amendments. Use the amended network policy where
specified: the lower-quality census uses the five networks of run 0; ALCDEF
uses all 25. The main 170 and ZTF use the five networks of their test fold.

The Python runners and shell drivers under `repro/own_period/` use
`DELPHI_DATA` for **the extracted `own-period-20261021/` folder**, containing
`benchmark/`, `lowq/`, `reliability/` and `own-periods-170.json`. This replaces
the earlier shell setting that named its parent `data/` folder. Settings:

| Setting | Meaning and default |
|---|---|
| `DELPHI_REPO` | Code folder; defaults to the repository root found from the shell script's location. |
| `DELPHI_DATA` | The extracted `own-period-20261021/` folder, with `benchmark/` directly underneath. Python data paths read this setting lazily when used; imports and `--help` need no setting. An unset or empty value stops a shell driver or Python data access. |
| `DELPHI_NETWORK_ROOT` | The extracted `retrain-20261030/artifact-root/`, containing `models/` and `evaluations/`. Its sibling `results/` contains the new network results. Set this for the revised paper; leaving the network variables unset selects historical inputs. |
| `DELPHI_MODEL_BINDING` | The supplied `model-binding.json` in `DELPHI_NETWORK_ROOT`; records the new model hashes. |
| `DELPHI_BUNDLE_ROOT` | The `bundles/` folder in `DELPHI_NETWORK_ROOT`, containing one safe inference bundle per fold. |
| `PYTHON` | Python executable; default `python`. |
| `CONVEXINV_PYTHON` | Interpreter for `v67_queue.sh` and `v67_finalize.sh`; defaults to `PYTHON`. Set it to the pinned interpreter recorded by the inversion locks. |
| `GPU_LOCK` | Optional lock-file path. The shell GPU steps use `flock "$GPU_LOCK"` when set. |
| `GPU_WAIT` | Optional Bash script, called before each shell GPU step. |
| `LOCK` | Optional lock-file path for the Python runners `refine_chain_post.py` and `temporal.py`. When set, they require a held lock inherited on file descriptor 9; when unset or empty, they skip that check. This is separate from the shell drivers' `GPU_LOCK`. |

For example, with the layout below:

```bash
export DELPHI_REPO=/path/to/DeLPHI-followup/source
export DELPHI_DATA=/path/to/DeLPHI-followup/data/own-period-20261021
export DELPHI_NETWORK_ROOT=/path/to/DeLPHI-followup/data/retrain-20261030/artifact-root
export DELPHI_MODEL_BINDING="$DELPHI_NETWORK_ROOT/model-binding.json"
export DELPHI_BUNDLE_ROOT="$DELPHI_NETWORK_ROOT/bundles"
export PYTHON=/path/to/venv/bin/python
# For locked inversion reruns, also set CONVEXINV_PYTHON to the recorded interpreter.
```

Python data paths are resolved from the current `DELPHI_DATA` at use time,
including paths imported from another runner and command-line defaults.
For historical reruns, put the checkout at the follow-up root's `source/`
folder before extracting the supporting and historical sources archives.
The quick checks of current results also work with a separate checkout.
Keep the data record's relative layout: earlier data folders are siblings of
the own-period folder;
`inputs/` and `experiments/` are two levels above it, and the DAMIT snapshot
is three levels above it. Source-file paths used to import or hash this
checkout still come from the source location.

Frozen evidence also refers to historical code folders (`source-op-maps`,
`source-op-lowq`, `source-op-reliability`, `source-op-nettests`, `source-op-v67`
and `source-bench-8a67c52`) alongside `inputs/` and `data/`.
`delphi-historical-sources-v1.1.0.tar.gz` supplies the selected files from
these folders; the supporting archive supplies the historical maps renderer.
The [historical input list](#historical-inputs-and-sources) describes the files
and commands that read them. Preserve the source files and hashes named in
the record. Inversion reruns verify their runtime sources and environment,
and median tools hash or extract functions
from the historical renderers. Setting `DELPHI_DATA` to the recorded
`data/own-period-20261021/` folder preserves that input layout.

In the table, **O/** means the `DELPHI_DATA` folder itself. Other paths are
also relative to `DELPHI_DATA`; `../` reaches its parent `data/` folder.
Script names without a prefix are in `repro/own_period/`.
CLI-selected output paths below describe the recorded runs.
The table retains the historical input and output layout. With the network
variables above set, network-dependent inputs and outputs select the retrained
tree through `repro/network_root.py` and `repro/own_period/reporting_inputs.py`.
The current paper-to-file map is in [Section 3](#3-where-each-result-comes-from);
retiming receipts and runner scripts are in `retrain-20261030/runners/`.

| Result family | Scripts | Reads | Writes |
|---|---|---|---|
| Own periods, main 170 | `manifest.py`; search implementation `repro/period_search/search.py` | `../../experiments/period-refinement-20260921/end-to-end/candidates-v1.jsonl` (top Fourier GPU period, alias P) | **O/**`own-periods-170.json` |
| Out-of-fold poles, 170 | `predict_oof.py`, `score.py`; `predict_oof_full.py` for the per-seed and score-map files | **O/**`own-periods-170.json`; `../../inputs/training-run/repro/data/damit-20250610T000301Z/`; frozen models and evaluations in `../../inputs/frozen-artifacts/k3-definitive-7874092/` | **O/**`own-period-oof.npz`, `validation-damit-period-oof.npz`, `own-period-scores-main.json` or `own-period-scores.json`; full predictions under **O/**`maps/` |
| ZTF | `search_external.py --cohort ztf`, `predict_external.py --cohort ztf`, `score.py` | `../generalization-20260910/mapped-ztf/prepared-current-endpoint/`; `../external-models-final.json`; frozen models; scoring references in `../publication-revision-20260917-r2/error-channel-sensitivity.json` | **O/**`own-periods-ztf.json`, `validation-ztf-published.jsonl`, `own-period-ztf.jsonl`, ZTF block of `own-period-scores.json` |
| ALCDEF | `search_external.py --cohort alcdef`, `predict_external.py --cohort alcdef`, `score.py` | `../generalization-20260910/locked-cohort/alcdef-prepared-30-20260911/` and `alcdef-gaia-transfer-analysis-30-20260911.json`; `../external-models-final.json`; frozen models | **O/**`own-periods-alcdef.json`, `validation-alcdef-published.jsonl`, `own-period-alcdef.jsonl`, ALCDEF block of `own-period-scores.json` |
| Inversion benchmark, including full grid and loaded runs | `benchmark_lock.py`, `benchmark_relock.py`, `full_grid.py`; frozen copies of `repro/direct_workflow_benchmark/run_direct_benchmark.py`, `run_loaded_benchmark.py`, `time_exploratory_budgets.py`, `score_exploratory_timing.py`, `score_classical_ladder.py`, `budget_equivalence.py`, `candidate_accuracy_cost.py` beside the locks | `../direct-workflow-benchmark/` scripts and locks; **O/**`own-periods-170.json`; frozen models, solver and hash-bound benchmark code | **O/**`benchmark/evaluation-lock.json`, `scoring-lock.json`, `evaluation/`, `evaluation-loaded/`, `full-grid/`, `exploratory-loaded-ladder/`, `score.json`, `score-loaded.json`, `classical-ladder-*.json`, `budget-equivalence.json`, `candidate-accuracy-cost.json` |
| Maps and candidate count | `maps/run_predictions.sh`, `maps/run_analyses.sh`; `predict_oof_full.py`, `maps/validate_damit_run.py`, `maps/analyses.py`, `maps/render_figures.py`, `maps/build_results.py` | **O/**`own-periods-170.json`; frozen evaluations; `../reliability-20260915/run-1/`; published interpretation records | **O/**`maps/damit-period/` and `maps/own-period/` (`predictions/`, `evaluations/`, `analyses/`, including `candidate-count.json`); `maps/published-check/`, `maps/figures/`; result table on standard output |
| Lower-quality census | `run_lowq_chain.sh`; `lowq_search.py`, `lowq_predict.py`, `lowq_validate.py`, `lowq_score.py`, `lowq_check_published.py`, `lowq_results.py`, `lowq_figures/render_main_figures.py` | `../../source/experiments/lowq_damit_census_20260917/`; DAMIT snapshot; frozen models; `../publication-revision-20260917-r2/`; published reference baselines | **O/**`lowq/search.jsonl`, `own-periods-lowq.json`, `own-period-lowq-predictions.jsonl`, validation records, `scores-own/`, `scores-published-check/`, `scores-damit-period-500/`, `figures/`, `RESULTS.md` |
| Census blind test (seven post-snapshot asteroids) | `census7.py` (`prepare`, `validate`, `search`, `predict --periods own|damit --networks fold0|all25`, `score --stage own|final`, all with `--out-dir`) | `census7/PROTOCOL.md`, `census7/inputs-sha256.txt`; `../damit-postsnapshot-check/raw/`; `../../source/experiments/lowq_damit_census_20260917/` (validation); frozen models | **O/**`census7/` (`prepared-manifest.json`, `exposure.json`, `validation.json`, `periods.json`, `predictions*.jsonl`, `scores-own.json`, `scores.json`, `RESULTS.md`) |
| Reduced-data and noise reliability | `reliability_chunks.sh`; `reliability.py`, `reliability_score.py`, `reliability_report.py` | `../reliability-20260915/run-1/study.json`, `schedule.json`, scored rows and fits; **O/**`own-periods-170.json`; frozen models | **O/**`reliability/search/`, `predictions/`, validation files, `score/scored-rows-own.json`, `score/reliability-own-period.json`, `RESULTS.md` |
| Network tests and external variants | `network_controls.py`, `summarize_controls.py`, `train_vs_test.py`, `shapley_op.py`, `shapley_check.py`, `external_variants.py`, `nettests_results.py` | **O/**`own-periods-170.json`; frozen models and evaluations; `../internal-validation-20260916/run-1/`; `../interpretation-20260926/shapley/`; external prepared inputs above | **O/**`nettests/controls-{damit,own}/`, `train-vs-test-{damit,own}/`, `shapley-{damit,own}/`, summary and validation JSON; `ztf-variants/`, `no-uncertainty/`, `RESULTS.md` |
| V6 fixed-period and V7 guided search | `v67_periods.py`, `v6_fixed_period.py`, `v7_grid.py`, `v67_queue.sh`, `v67_finalize.sh`, `v67_results.py` | **O/**`own-periods-170.json`, `own-period-oof.npz`, `benchmark/evaluation-lock.json`; `../pole-grid-benchmark/20260912-environment-relocked-v3/`; earlier paper summaries | **O/**`v67/own-periods-170-rounded6.json`, validation runs, `v6-own-period/`, `v7-own-period/`, score JSON, `v7-own-period/crosscheck-own-period-full-grid.json`, `RESULTS.md` |
| Withheld-lightcurve test | `reliability.py search --scope withheld`, `withheld-candidates`, `withheld-fits`; corresponding stages in `reliability_chunks.sh`; `reliability_score.py`, `reliability_report.py` | Retained lightcurves and schedule from `../reliability-20260915/run-1/`; searches use only retained observations; fixed 15.8-degree radius | **O/**`reliability/search/periods-withheld.jsonl`, `reliability/withheld/{damit,own}/` candidates and fits; withheld block of `reliability/score/reliability-own-period.json` |
| Period-search cost | `period_cost.py` | Own-period manifest, `period-cost/PROTOCOL.md`, lightcurves and period-search records | **O/**`period-cost/` (`trial-counts.json`, `period-cost-summary.json`) |
| Period and pole refinement | `refine_chain.py`, `refine_chain_post.py` | `refine-chain/PROTOCOL.md`, frozen sources, solver and manifests | **O/**`refine-chain/` (`refine-medians.json`, selected fits and scores) |
| Spacecraft reference checks | `spacecraft/fetch.py`, `convert.py`, `run.py`, `classical.py`, `classical_freep.py`, `posthoc.py`, `eros_loo.py`, `eros_context.py`, `reference_poles.py` | `spacecraft/` protocols, acquired photometry, geometry and independent reference poles | **O/**`spacecraft/` (acquisition, predictions, fits and scores) |
| Median statistics | `median_stats.py` | Explicit `--input-list` macro inventory and optional `--manuscript`; per-object evidence under **O/**`maps/`, `lowq/`, `reliability/`, `nettests/`, `benchmark/`, `v67/`, `failed-excluded-derived/`, plus published reference records | **O/**`median-stats.json` by default (`--out` overrides); required `--report` path |
| Median figures | `median_figures.py` | **O/**`median-stats.json` and the same frozen evidence; historical renderers; manuscript and original figures in `../../../paper-rev-20261021/` | Required `--out-dir`: PDFs, PNGs and `values.json`; `f1-report.md` and `f2-report.md` in its parent folder |

The main protocol fixes the top Fourier period without choosing P/2 or 2P
after scoring. Reduced, noisy, external-variant and retained-lightcurve inputs
receive their own searches. Failures are reported at 90 degrees and also
excluded. Validation at published periods precedes the corresponding own-period
run. Shell drivers are resumable queues: maps, lower-quality and reliability
wait for `O/benchmark/explore-timed.log`; V67 waits for
`O/reliability/phaseB2.log`, and its finalizer waits for `O/v67/queue.log`.
Keep these logs with the data record when using those drivers.

Median commands require explicit scratch destinations, for example:

```bash
"$PYTHON" -m repro.own_period.median_stats \
  --input-list /path/to/mean-macros.txt --report /path/to/median-report.md
"$PYTHON" -m repro.own_period.median_figures \
  --figures candidate-count inversion-benchmark inversion-timing --out-dir /path/to/median-figures
```

## 1. What to download

Use the software release
[`v1.1.0`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.1.0),
software DOI `10.5281/zenodo.23787947`, with the
[Zenodo data record v1.1.0, DOI 10.5281/zenodo.23857881](https://doi.org/10.5281/zenodo.23857881).
Download URLs have the form
`https://zenodo.org/records/23857881/files/<name>`.

The record has 16 uploads: five new archives, five small network sets,
four retained archives, `README.md` and `SHA256SUMS`. The checksum file covers
the other 15 uploads.

| File | Contents and extraction layout |
|---|---|
| [`k3-oof-fold-0-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/k3-oof-fold-0-v1.1.0.tar.gz) | Five retrained networks for run 0, about 12 MB. Extract into `networks/`; creates `k3-oof-fold-0/` for the Astraea example or a new asteroid. |
| [`k3-oof-fold-1-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/k3-oof-fold-1-v1.1.0.tar.gz) | Five retrained networks for run 1, about 12 MB. Extract into `networks/`; creates `k3-oof-fold-1/`. |
| [`k3-oof-fold-2-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/k3-oof-fold-2-v1.1.0.tar.gz) | Five retrained networks for run 2, about 12 MB. Extract into `networks/`; creates `k3-oof-fold-2/`. |
| [`k3-oof-fold-3-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/k3-oof-fold-3-v1.1.0.tar.gz) | Five retrained networks for run 3, about 12 MB. Extract into `networks/`; creates `k3-oof-fold-3/`. |
| [`k3-oof-fold-4-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/k3-oof-fold-4-v1.1.0.tar.gz) | Five retrained networks for run 4, about 12 MB. Extract into `networks/`; creates `k3-oof-fold-4/`. |
| [`delphi-k3-artifacts-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/delphi-k3-artifacts-v1.1.0.tar.gz) | Full archive for reproduction (648 MB): retrained networks, training histories, fold bundles, model binding and evaluation arrays. Extract to the data directory; creates `retrain-20261030/artifact-root/`. `DELPHI_NETWORK_ROOT` points there, `DELPHI_MODEL_BINDING` to its `model-binding.json`, and `DELPHI_BUNDLE_ROOT` to its `bundles/`. |
| [`delphi-k3-synthetic-data-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/delphi-k3-synthetic-data-v1.1.0.tar.gz) | Corrected simulated training, validation and test sets, recipe metadata and generation checks. Extract to the data directory; creates `retrain-20261030/synthetic/`. |
| [`delphi-retrain-results-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/delphi-retrain-results-v1.1.0.tar.gz) | Rerun predictions, score maps, figures, summaries, locks, protocols, run logs, training times, runners and RT4–RT7 receipts. Extract to the data directory; creates `retrain-20261030/`, with results in `results/`, beside `artifact-root/`. Includes the failed V6 attempt's audit evidence. |
| [`delphi-retrain-benchmark-fits-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/delphi-retrain-benchmark-fits-v1.1.0.tar.gz) | Newly written per-start solver parameters, solutions, modelled lightcurves and output records for direct, withheld, refinement, reporting-validation, RT4/RT5 and RT7 fits. Extract to the data directory; paths start with `retrain-20261030/results/`. |
| [`delphi-historical-sources-v1.1.0.tar.gz`](https://zenodo.org/records/23857881/files/delphi-historical-sources-v1.1.0.tar.gz) | Historical sources and input locks (6.4 MB, 128 files), including census validation inputs and the historical model binding. Extract at the follow-up root, one level above `data/`, after placing the checkout at `source/`. Paths begin with `source-op-.../`, `source-bench-8a67c52/`, `source-own-period/`, `source/`, `data/` and `inputs/`. The source folders contain selected files only. |
| [`delphi-own-period-results-v1.0.0.tar.gz`](https://zenodo.org/records/23857881/files/delphi-own-period-results-v1.0.0.tar.gz) | Retained unchanged: periods, protocols, classical baselines, sample definitions and historical summaries. Extract to the data directory; creates `own-period-20261021/`, selected by `DELPHI_DATA`. Its old network scores are historical. |
| [`delphi-benchmark-fits-v1.0.0.tar.gz`](https://zenodo.org/records/23857881/files/delphi-benchmark-fits-v1.0.0.tar.gz) | Retained unchanged: classical and historical comparison fits. Extract to the data directory; paths start with `own-period-20261021/benchmark/`, `direct-workflow-benchmark/` or `pole-grid-benchmark/20260912-environment-relocked-v3/full/`. |
| [`delphi-supporting-results-v1.0.0.tar.gz`](https://zenodo.org/records/23857881/files/delphi-supporting-results-v1.0.0.tar.gz) | Retained unchanged: cited supporting tables, derived DAMIT catalogue tables and historical scripts. Extract to the follow-up root, one level above the data directory; paths start with `data/`, `experiments/`, `inputs/`, `source/`, `source-own-period/` or `source-op-maps/`. |
| [`delphi-paper-v1-evidence.tar.gz`](https://zenodo.org/records/23857881/files/delphi-paper-v1-evidence.tar.gz) | Retained unchanged: historical evidence and classical analyses. Extract in a separate `legacy/` directory; creates `delphi-paper-v1-evidence/part-a-validation/` and `part-b-analyses/`. |
| [`README.md`](https://zenodo.org/records/23857881/files/README.md) | Data record inventory, extraction instructions, paper map, input sources and licence. |
| [`SHA256SUMS`](https://zenodo.org/records/23857881/files/SHA256SUMS) | Checksums for the other 15 deposited files. |

For prediction, download only the appropriate small bundle and `SHA256SUMS`;
see the [prediction guide](usage.md#2-download-and-check-one-network-set).
The full artifact archive also contains all five bundles for reproduction.

The superseded v1.0.0 network-artifact and simulated-data archives, five
separate v1.0.0 fold bundles and old release manifests/checks remain in the
[v1.0.0 data record, DOI 10.5281/zenodo.23757942](https://doi.org/10.5281/zenodo.23757942).
They are not carried into v1.1.0. The four retained archives keep their old
names and bytes; their historical network scores are not the revised results.
Use this version's `SHA256SUMS` to check downloads.

For the complete set, start in an empty folder:

```bash
mkdir delphi-check && cd delphi-check
mkdir -p downloads extracted
git clone --branch v1.1.0 https://github.com/HBJ1004/DeLPHI_LC.git extracted/source
cd downloads
URL=https://zenodo.org/records/23857881/files
for f in delphi-k3-artifacts-v1.1.0.tar.gz delphi-k3-synthetic-data-v1.1.0.tar.gz \
    k3-oof-fold-0-v1.1.0.tar.gz k3-oof-fold-1-v1.1.0.tar.gz \
    k3-oof-fold-2-v1.1.0.tar.gz k3-oof-fold-3-v1.1.0.tar.gz k3-oof-fold-4-v1.1.0.tar.gz \
    delphi-retrain-results-v1.1.0.tar.gz delphi-retrain-benchmark-fits-v1.1.0.tar.gz \
    delphi-historical-sources-v1.1.0.tar.gz \
    delphi-own-period-results-v1.0.0.tar.gz delphi-benchmark-fits-v1.0.0.tar.gz \
    delphi-supporting-results-v1.0.0.tar.gz delphi-paper-v1-evidence.tar.gz \
    README.md SHA256SUMS; do
  curl --fail -L -O "$URL/$f"
done
sha256sum --check SHA256SUMS
cd ..
```

Leave out files you do not need, and add `--ignore-missing` to check only the
files downloaded. On macOS use `shasum -a 256 --check SHA256SUMS --ignore-missing`.
Each checked file should end in `OK`. Extract with the roots listed above:

```bash
mkdir -p extracted/data extracted/legacy
for f in delphi-own-period-results-v1.0.0.tar.gz delphi-benchmark-fits-v1.0.0.tar.gz \
    delphi-k3-artifacts-v1.1.0.tar.gz delphi-k3-synthetic-data-v1.1.0.tar.gz \
    delphi-retrain-results-v1.1.0.tar.gz delphi-retrain-benchmark-fits-v1.1.0.tar.gz; do
  tar xzf "downloads/$f" -C extracted/data
done
tar xzf downloads/delphi-supporting-results-v1.0.0.tar.gz -C extracted
tar xzf downloads/delphi-historical-sources-v1.1.0.tar.gz -C extracted
tar xzf downloads/delphi-paper-v1-evidence.tar.gz -C extracted/legacy
export DELPHI_DATA="$PWD/extracted/data/own-period-20261021"
export DELPHI_NETWORK_ROOT="$PWD/extracted/data/retrain-20261030/artifact-root"
export DELPHI_MODEL_BINDING="$DELPHI_NETWORK_ROOT/model-binding.json"
export DELPHI_BUNDLE_ROOT="$DELPHI_NETWORK_ROOT/bundles"
```

Saved fits are optional for checking score and timing tables. The new fits
archive contains newly written regular files, omitting absolute symlinks to
reused historical fits; the retained fits archive preserves the old selection.
The documented failed V6 attempt stays together in the new results archive.

### Minimal download for Check A

Check A needs only the new results archive. Its paths below name the same
layout as the complete extraction; the network files themselves are not needed
for this check.

```bash
mkdir delphi-check && cd delphi-check
git clone --branch v1.1.0 https://github.com/HBJ1004/DeLPHI_LC.git extracted/source
mkdir -p downloads extracted/data
cd downloads
curl --fail -LO https://zenodo.org/records/23857881/files/delphi-retrain-results-v1.1.0.tar.gz
curl --fail -LO https://zenodo.org/records/23857881/files/SHA256SUMS
sha256sum --check SHA256SUMS --ignore-missing
cd ..
tar xzf downloads/delphi-retrain-results-v1.1.0.tar.gz -C extracted/data
export DELPHI_NETWORK_ROOT="$PWD/extracted/data/retrain-20261030/artifact-root"
```

### Install

Python 3.11 or 3.12 is needed. From `delphi-check/`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # or python3.11
python -m pip install -e './extracted/source[test,plot]'
delphi-k3 validate-protocol
```

The last command checks that the fixed configuration file of the model,
`repro/k3_redesign_spec.yaml`, is the one used for the paper. It prints

```text
{"schema": "delphi.k3-redesign-spec.v1", "sha256": "84dee816d08a60b7780a6d1d400add16136fa96c0677e84929700cee8bf22a50"}
```

## 2. Quick checks without rerunning anything

Run the checks from `delphi-check/` inside the environment created under
[Install](#install). Checks A and B use the retrained results and networks; Check C uses the
unchanged historical evidence archive.

### Check A: Paper I own-period results

Recompute the main sample's median from the per-object own-period errors,
then read the standard-pole and random-axis medians from `median-stats.json`.
That file records sources, bootstrap intervals and the statistic of each
entry; the random-axis value is the mean of the draw-wise medians.

```bash
python - <<'EOF'
import json
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["DELPHI_NETWORK_ROOT"]).parent / "results"
main = json.loads((root / "headline/own-period-scores-main.json").read_text())["main_sample"]
errors = np.array([row["own_error_deg"] for row in main["objects"]])
entries = {row["mean_macro"]: row for row in
           json.loads((root / "median-stats.json").read_text())["entries"]}
assert np.isclose(np.median(errors), entries["KThreeOofMean"]["median"])
print("asteroids %d: median %.2f deg, within 20 deg %.1f%%" % (
    len(errors), np.median(errors), 100 * (errors <= 20).mean()))
for macro in ("KThreeStandardMean", "KThreeRandomMean"):
    print("%s: %.2f deg (%s)" % (macro, entries[macro]["median"], entries[macro]["statistic"]))
EOF
```

Expected output:

```text
asteroids 170: median 10.70 deg, within 20 deg 77.1%
KThreeStandardMean: 24.13 deg (median)
KThreeRandomMean: 32.33 deg (mean_of_randomization_draw_medians)
```

The `Mean` strings in the macro names are historical identifiers. The
`statistic` fields describe the revised median calculation. Inversion
medians are in the `WorkflowDelphiError` and `WorkflowClassicalError` entries
(23.55° and 45.58°). Their per-object records and same-PC timing ratios are
in `retrain-20261030/results/delphi-inversions/timing-retime/own/score.json`
and `score-loaded.json`. Do not substitute the original-period evidence
for these own-period results.

### Check B: supplied-period oracle errors from archived candidate axes

This recomputes the supplied-DAMIT-period comparison using the retrained
networks' stored axes and the reference poles. Check A uses our estimated
periods instead. Check B needs `delphi-k3-artifacts-v1.1.0.tar.gz` extracted
and `DELPHI_NETWORK_ROOT` set as above.

```bash
python - <<'EOF'
import json
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["DELPHI_NETWORK_ROOT"])
ensemble = np.load(root / "evaluations/real-oof-ensemble.npz")
catalog = "extracted/source/repro/data/damit-20250610T000301Z/catalog.jsonl"
poles = {row["object_id"]: np.array([s["vector"] for s in row["solutions"]])
         for row in map(json.loads, open(catalog))}

errors = []
for name, axes in zip(ensemble["object_ids"], ensemble["refined_axes"]):
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    ref = poles[str(name)] / np.linalg.norm(poles[str(name)], axis=1, keepdims=True)
    errors.append(np.degrees(np.arccos(min(1.0, np.abs(axes @ ref.T).max()))))
errors = np.array(errors)

print("asteroids %d: median %.2f deg, within 20 deg %.1f%%" % (
    len(errors), np.median(errors), 100 * (errors <= 20).mean()))
print("largest difference from the stored errors: %.0e deg" % np.abs(errors - ensemble["oracle_errors_deg"]).max())
for run in range(5):
    e = errors[ensemble["folds"] == run]
    print("run %d: n %d, median %.2f" % (run, len(e), np.median(e)))
for seed in (17, 42, 137, 777, 2027):
    e = np.concatenate([np.load(root / ("evaluations/real-fold-%d-seed-%d.npz" % (run, seed)))["oracle_errors_deg"]
                        for run in range(5)])
    print("seed %d alone: median %.2f" % (seed, np.median(e)))
EOF
```

Expected output:

```text
asteroids 170: median 10.48 deg, within 20 deg 77.6%
largest difference from the stored errors: 2e-06 deg
run 0: n 34, median 10.45
run 1: n 34, median 10.52
run 2: n 34, median 11.21
run 3: n 34, median 10.65
run 4: n 34, median 8.75
seed 17 alone: median 13.44
seed 42 alone: median 13.02
seed 137 alone: median 13.32
seed 777 alone: median 11.47
seed 2027 alone: median 16.00
```

These supplied-period diagnostics differ from the estimated-period results
in Check A. Appendix A's grid-to-refined medians, 10.47° and 10.70°, use
our estimated periods.

### Check C: supporting validation tables

`part-a-validation/` carries its own checks (its `VERIFY.md`). From
`delphi-check/extracted/legacy/delphi-paper-v1-evidence/part-a-validation/`:

```bash
python manuscript-derived/import_followup.py --verify --output k3-followup
python manuscript-derived/derive_followup.py --verify
```

They print `{"files": 64, "passed": true}` and `{"passed": true, "files": 7}`.
The second command re-derives, from the Part A records, the DeLPHI values of
the original-period reduced-data tests, internal checks, lower-quality models,
ZTF, ALCDEF and withheld-lightcurve test, and checks
them against `k3-followup-*.tex`. Those files also list a train-only fixed-axis
baseline (labelled "atlas"); the standard-pole and random-axes values of those comparisons are in
`part-b-analyses/baselines/reference-baselines.json`.

## 3. Where each result comes from

**Asteroids outside the sample.** The ALCDEF and spacecraft comparisons use
all 25 retrained networks. The main 170 and ZTF use the five networks of the
asteroid's test run; the lower-quality census uses run 0. The supported
`delphi-k3-predict` command uses one set of five networks for a new asteroid.
Use the supplied [model binding file](#model-binding-file) for analysis runners.

In this map, **R/** is `extracted/data/retrain-20261030/`, **O/** is
`DELPHI_DATA` (`extracted/data/own-period-20261021/`), and **E/** is
`extracted/legacy/delphi-paper-v1-evidence/`. **K/** is
`DELPHI_NETWORK_ROOT` (`R/artifact-root/`). The new results archive supplies
`R/results/`, except for the per-start solver files in the new fits archive;
the failed V6 attempt stays with its audit evidence in the results archive.
Historical network outputs in **O/** and **E/** describe the old networks.
The numbers below refer to the current manuscript build.

| Paper | Result files | Code or protocol |
|---|---|---|
| Section 2, Table 1, Figure 1 (sample, coverage) | Repository `repro/data/damit-20250610T000301Z/{catalog.jsonl,publication-splits-v2.3.json}`; `E/part-a-validation/k3-followup/supplement/` | `repro/export_k3_reliability_supplement.py`; DAMIT snapshot |
| Sections 3 and 4; Appendix A, Table 6 (model and training) | `R/synthetic/`, `K/models/`, `K/bundles/`, `K/evaluations/`, `K/model-binding.json`, `R/train-times.tsv` | `repro/k3_redesign_spec.yaml`, `R/PROTOCOL.md`, `R/RUNLOG.md`, `R/runners/`; [configuration.md](configuration.md) |
| Sections 5.1 and 5.2, Table 2, Figures 3–6 (oracle error, examples, test sample) | `R/results/headline/`, including `maps/own-period/{evaluations,analyses}/`; `R/results/{median-stats,paired-gains}.json` | `repro/own_period/predict_oof_full.py`, `repro/own_period/maps/`; Check A. Classical references and sample definitions remain in `O/` |
| Section 5.3 (computational cost) | `R/results/headline/maps/own-period/analyses/summary.json`; `R/results/delphi-inversions/timing-retime/{own,damit}/`; `R/results/period-retime/` | `R/runners/receipts-rt{4,5,6,7}/`; historical period-scan definitions in `O/period-cost/` |
| Section 6.1.1 (controls) | `R/results/{input-removals,controls-input-swap,controls-label-shuffle}/` | `repro/own_period/network_controls.py`, `repro/own_period/summarize_controls.py` |
| Section 6.1.2, Appendix D, Tables 10 and 11, Figure 7 (other photometry and lower-quality models) | `R/results/{lower-quality-census,ztf,alcdef,external-variants,derived-comparators}/` | `repro/own_period/{lowq_predict,predict_external,external_variants}.py`; DAMIT, Fink/ZTF, ALCDEF, Gaia DR3 and Horizons inputs |
| Section 6.1.3, Table 3 (spacecraft targets) | `R/results/spacecraft/` | `repro/own_period/spacecraft/`; recorded acquisition receipts and independent reference poles |
| Section 6.1.4, Figure 8; Appendix B, Figure 14 (reduced data and noise) | `R/results/reliability/`, `R/results/derived-comparators/reliability-paired.json`, `R/results/median-stats.json` | `repro/own_period/{reliability,reliability_score,reliability_report}.py` |
| Sections 6.2.1 and 6.3.1, Figures 9 and 11 (mirror ambiguity and pole direction) | `R/results/headline/maps/own-period/analyses/`, `R/results/mirror/` | `repro/own_period/maps/analyses.py`; historical classical products in `E/part-b-analyses/interpretation/` |
| Section 6.2.2, Figure 9 (amplitude–aspect relation) | `R/results/headline/maps/own-period/analyses/` | `repro/own_period/maps/analyses.py` |
| Section 6.2.3, Figure 10; Appendix B (input contributions) | `R/results/shapley/`, `R/results/input-removals/`, `R/results/median-stats.json` | `repro/own_period/{shapley_op,shapley_check}.py` |
| Appendix B, Tables 8 and 9, Figure 15 (folds, seeds and internal checks) | `K/evaluations/`, `R/results/internal-validation/`, `R/results/median-stats.json` | Check B; `repro/own_period/train_vs_test.py` |
| Appendix B, Figure 16 (input removals) | `R/results/input-removals/` | `repro/own_period/network_controls.py` |
| Section 6.3.3, Appendix C, Figures 17 and 18 (period sensitivity and search) | `R/results/{period-sensitivity,period-peak-pilot,period-retime}/`; unchanged `O/own-periods-170.json`, `O/period-cost/` and `E/part-b-analyses/period-search/` | `repro/period_search/` ([README](../repro/period_search/README.md)); `R/runners/check_rt6_fourier.py`, `R/runners/check_rt7_period_bench.py` |
| Section 6.4.1, Figure 12 (candidate count) | `R/results/headline/maps/own-period/analyses/candidate-count.json` | `repro/own_period/maps/analyses.py` |
| Section 6.4.2, Tables 4 and 5, Figure 13; Appendix E, Tables 12 and 13, Figure 19 (inversion comparisons) | `R/results/delphi-inversions/`, including `timing-retime/`; `R/results/refine-chain/`, `R/results/reporting/v7-validation.json`; unchanged classical inputs in `O/benchmark/` and `E/part-b-analyses/workflow-benchmark/` | `repro/direct_workflow_benchmark/`, `repro/own_period/{v6_fixed_period,v7_grid,refine_chain}.py`; RT4/RT5 receipts |
| Appendix E.1, Table 14 (withheld-lightcurve test) | `R/results/reliability/{score,withheld}/`; unchanged classical results in `O/reliability/score/` | `repro/own_period/reliability.py`; new per-start fits in the new fits archive |
| Section 6.5 (future work and census) | `E/part-b-analyses/damit-census/q3_current_classification.csv`, `R/results/census7/` | `repro/damit_census/`, `repro/own_period/census7.py` |
| Section 7 (summary) | The same files as Sections 5 and 6 | |

## 4. Folder layout for rerunning the analyses

Some analysis scripts still require the historical input layout below.
Use the new archives under `data/retrain-20261030/` and keep all three network
variables set to select the retrained networks and results. The old frozen
artifacts and simulated data shown under `inputs/` are historical inputs from
the v1.0.0 data record; they are not the revised networks. Several scripts
require the top folder name `DeLPHI-followup`. Here **A/** and **B/** are the
legacy evidence archive's `part-a-validation/` and `part-b-analyses/` folders.

```text
<parent>/
  damit-20250610T000301Z/                  DAMIT snapshot (tables/, files/)
  paper/figures/                           output folder of Figure 13 (create it)
  DeLPHI-followup/
    source/                                this repository at tag v1.1.0
      experiments/lowq_damit_census_20260917/
        input-index.json                   from historical sources archive
        predictions-fold0-five-seed/predictions.jsonl
                                           original census validation predictions
    source-op-maps/, source-op-lowq/, source-op-reliability/
    source-op-nettests/, source-op-v67/     selected historical files (below)
    source-own-period/                     selected sources from the supporting and historical archives
    source-bench-8a67c52/                   92 Python files and licence, not a full checkout
    inputs/
      frozen-artifacts/k3-definitive-7874092/   Zenodo artifact-root/
      frozen-artifacts/k3-2408c563/synthetic/   Zenodo synthetic-data/ (census only)
      frozen-artifacts/k3-2408c563/comparators/v1-comparators.npz
                                           from historical sources archive
      training-run/repro/data/damit-20250610T000301Z/
                                           copy of source/repro/data/damit-20250610T000301Z
    data/
      retrain-20261030/
        artifact-root/                     DELPHI_NETWORK_ROOT (models, evaluations, bundles)
        synthetic/                         corrected simulations
        results/                           revised results and same-PC retimes
        PROTOCOL.md, RUNLOG.md, runners/    retraining and rerun records
      own-period-20261021/                own-period results and benchmark-fits archives
      reliability-20260915/run-1/         original study/schedule and scoring inputs
      internal-validation-20260916/run-1/study.json
                                           from historical sources archive
      ...                                sibling folders from supporting-results archive
      interpretation-20260926/             from B/interpretation/
      interpretation-20260930/             from B/interpretation/
      direct-workflow-benchmark/           B/workflow-benchmark/
      damit-postsnapshot-check/            B/damit-census/
      publication-revision-20260917-r2/    A/k3-followup/revision/
      generalization-20260910/             ALCDEF result and Gaia DR3 spin table
      external-models-final.json           historical binding from historical sources archive
      solver-internal-capacity-corrected/20260911/expanded-solver/convexinv/
                                           patched convexinv (below)
    experiments/period-refinement-20260921/
                                           repro/period_search/*.py and B/period-search/results/
```

### Arrange the archives

For current results, use the extraction instructions in Section 1, with
`extracted/` serving as the follow-up root. For the retained historical
comparisons, work in an empty folder `<parent>` and separately extract the
v1.0.0 artifacts and synthetic-data archives there, creating `artifact-root/`
and `synthetic-data/`. Set `E` to the legacy evidence folder and `D` to the
download folder. Put the current retraining archives under
`DeLPHI-followup/data/`. Clone the checkout first, then extract the supporting
results and historical sources archives at `DeLPHI-followup/`, as below.
Copy directory contents into the existing destinations so that the archives
do not create nested copies. Keep the new network variables set for current results.

```bash
E=/path/to/delphi-paper-v1-evidence
D=/path/to/downloads
T=DeLPHI-followup
git clone --branch v1.1.0 https://github.com/HBJ1004/DeLPHI_LC.git $T/source
tar xzf "$D/delphi-supporting-results-v1.0.0.tar.gz" -C "$T"
tar xzf "$D/delphi-historical-sources-v1.1.0.tar.gz" -C "$T"
mkdir -p $T/inputs/frozen-artifacts/k3-2408c563 $T/inputs/training-run/repro/data \
         $T/data/interpretation-20260926 $T/data/interpretation-20260930 \
         $T/data/generalization-20260910/locked-cohort \
         $T/data/generalization-20260910/sources/gaia-dr3-spins-20260911 \
         $T/data/direct-workflow-benchmark $T/data/damit-postsnapshot-check \
         $T/data/publication-revision-20260917-r2 \
         $T/experiments/period-refinement-20260921/end-to-end paper/figures
mv artifact-root $T/inputs/frozen-artifacts/k3-definitive-7874092
mv synthetic-data $T/inputs/frozen-artifacts/k3-2408c563/synthetic
cp -r $T/source/repro/data/damit-20250610T000301Z $T/inputs/training-run/repro/data/

I=$E/part-b-analyses/interpretation
cp -r $I/primary-intervals.json $I/random-axes-cdf.npz $I/mirror-symmetry.json \
      $I/candidate-count.json $I/shapley $I/shapley-summary.json $T/data/interpretation-20260926/
cp $I/amplitude-aspect.json $I/amplitude-aspect.npz $I/amplitude-aspect-paired.json \
   $I/label-shuffle-spread.json $I/sampling-baselines.json $T/data/interpretation-20260930/
cp -r "$E/part-b-analyses/workflow-benchmark/." "$T/data/direct-workflow-benchmark/"
cp -r "$E/part-b-analyses/damit-census/." "$T/data/damit-postsnapshot-check/"
cp -r "$E/part-a-validation/k3-followup/revision/." "$T/data/publication-revision-20260917-r2/"
cp $E/part-a-validation/evidence/source-records/alcdef-gaia-analysis.json \
   $T/data/generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json
curl -L -o $T/data/generalization-20260910/sources/gaia-dr3-spins-20260911/table3.dat \
   https://cdsarc.cds.unistra.fr/ftp/J/A+A/675/A24/table3.dat

P=$T/experiments/period-refinement-20260921
cp $T/source/repro/period_search/*.py $P/
cp $E/part-b-analyses/period-search/results/* $P/end-to-end/
mv $P/end-to-end/warm170-first.* $P/
```

The Gaia DR3 spin table is that of Ďurech and Hanuš (2023, A&A 675, A24) at
CDS. Its SHA-256 on 2026-10-01 was
`e7f9743ba094860e553342eff83fe310f6dfcf7e6f740e0794f798a5e81ef865`, the same as
the copy used for the paper.

The arranged results in `data/` serve as inputs of later scripts. Keep the
downloaded archives as comparison evidence before running scripts that write
into that tree. Install the repository (`python -m pip install -e "$T/source[test,plot]"`)
and run the scripts from `$T/source`, for example
`python repro/primary_intervals.py`. When we tested this layout, rerunning
`primary_intervals.py`, `candidate_count.py`, `label_shuffle_spread.py`,
`compute_reference_baselines.py`, `sampling_baselines.py`, `mirror_symmetry.py`,
`amplitude_aspect_comparison.py`, `amplitude_aspect_paired.py`,
`group_shapley_summary.py`, `damit_census/lc_density.py`,
`direct_workflow_benchmark/budget_equivalence.py` and
`direct_workflow_benchmark/score_classical_ladder.py` in this layout
reproduced the released files exactly or to the last digits of floating-point
round-off.

### Historical inputs and sources

Extract `delphi-historical-sources-v1.1.0.tar.gz` at the follow-up root,
alongside `data/` and `inputs/`. For the complete download above that root is
`extracted/`; for the historical layout it is `DeLPHI-followup/`.
The archive adds the two original census files directly under
`source/experiments/lowq_damit_census_20260917/`:

- `input-index.json`, read by `lowq_search.py`, `lowq_predict.py` and
  `census7.py prepare`.
- `predictions-fold0-five-seed/predictions.jsonl`, read by the historical
  `lowq_validate.py` and `census7.py` validation steps.

These are the paths the runners read. Copying the legacy census export to
`data/damit-postsnapshot-check/` alone does not put them there. The new archive
places both files correctly, without a copy or rename. Their SHA-256 values are:

| Census file | SHA-256 |
|---|---|
| `input-index.json` | `392d2dda3336bee1c4639b93f90fc1db4225aff57bb30c78d5435bbb0625a4be` |
| `predictions-fold0-five-seed/predictions.jsonl` | `b3adff514191f4f72f325587c64342859224a75333aa9b7ade18bf2e6ae8f861` |

The archive also puts the original `study.json` and `schedule.json` in
`data/reliability-20260915/run-1/`, with `scored-rows.json`,
`sampling-table.csv` and `report.json`. `reliability.py` uses these locks
for reduced-data, noise and withheld-lightcurve sampling;
`reliability_score.py` uses the original scored rows for validation. Keep
their sealed contents unchanged. Their recorded file paths still describe
the original installation; see the [rerun limits](#scripts-that-cannot-run-from-the-released-files-alone).
The archive supplies `data/internal-validation-20260916/run-1/study.json`
for `train_vs_test.py` as well.

`median_stats.py` and `median_figures.py` read or hash selected historical
producer files. The archive supplies the following files, with their licences.
Paths in the second column are relative to the folder in the first column.

| Historical folder | Selected files |
|---|---|
| `source-op-maps/` | `repro/own_period/maps/analyses.py`. The matching `repro/own_period/maps/render_figures.py` comes from the supporting results archive. |
| `source-op-lowq/` | `repro/own_period/lowq_score.py` and `repro/own_period/lowq_figures/render_main_figures.py`. |
| `source-op-reliability/` | `repro/own_period/reliability_score.py` and `repro/baselines_out/reference-baselines-per-object.npz`, also read by reliability scoring and `derived_comparators.py`. |
| `source-op-nettests/` | `repro/own_period/external_variants.py`, `repro/own_period/summarize_controls.py` and `repro/own_period/train_vs_test.py`. |
| `source-op-v67/` | `repro/own_period/v7_grid.py`. |
| `source-own-period/` | `repro/amplitude_aspect_paired.py` and `repro/compute_reference_baselines.py`, in addition to the files in the supporting results archive. |
| `source-bench-8a67c52/` | All 91 `lc_pipeline` Python files and `repro/run_delphi_workflow_benchmark.py`, the 92 Python files checked by the benchmark runtime locks. This is a subset of the original tree. |

The benchmark subset is needed by `benchmark_lock.py` and the historical
`run_direct_benchmark.py` and `run_loaded_benchmark.py` commands. The source
folders above contain selected files only. They are not complete Git checkouts.
Other small inputs in the archive include the original benchmark locks and
reporting references, plus
`inputs/frozen-artifacts/k3-2408c563/comparators/v1-comparators.npz` for the
median and comparison tools. Current networks and results remain under
`data/retrain-20261030/`.

### DAMIT data

The DAMIT database (Ďurech et al. 2010, A&A 513, A46;
<https://damit.cuni.cz/projects/damit/>) is not redistributed here. The paper
used a copy retrieved in June 2025 and stored as `damit-20250610T000301Z/`.
DAMIT publishes monthly complete exports, and its export of 1 June 2025 is
identical to that copy for all tables and all lightcurve files and for the
spin files of the 170 asteroids:

```bash
curl -L -O https://damit.cuni.cz/projects/damit/exports/complete/damit-20250601T000301Z.tar.gz
tar xzf damit-20250601T000301Z.tar.gz
mv damit-20250601T000301Z damit-20250610T000301Z
```

The archive is about 1.4 GB. The repository lists the SHA-256 of the 442 files
of the main sample (model table, lightcurves and spin files) in
`repro/data/damit-20250610T000301Z/source_files.jsonl`. From `<parent>`:

```bash
python - <<'EOF'
import hashlib, json
bad = 0
for line in open("DeLPHI-followup/source/repro/data/damit-20250610T000301Z/source_files.jsonl"):
    row = json.loads(line)
    data = open("damit-20250610T000301Z/" + row["path"], "rb").read()
    bad += hashlib.sha256(data).hexdigest() != row["sha256"]
print("files differing from the June 2025 snapshot:", bad)
EOF
```

It should print 0. Single tables and lightcurves can also be exported from
DAMIT (`https://damit.cuni.cz/projects/damit/exports/table/asteroids`,
`.../exports/table/asteroid_models`,
`.../light_curves/exportAllForAsteroid/<DAMIT asteroid id>/plaintext` and
`.../generated_files/open/AsteroidModel/<model id>/spin.txt`), as
`lc_pipeline/k3/temporal_external.py` does. A current download differs from the
June 2025 snapshot: DAMIT has added models and lightcurves since then (Section
6.5 of the paper), and the current table exports also differ in format, so
current files will not match these checksums.

### Patched `convexinv`

The inversion comparison, the withheld-lightcurve test and the physical pilot
of the period search use `convexinv` from the DAMIT software, with three storage
limits raised so that the largest data sets of the sample fit (Appendix E). The
fitting algorithm is unchanged.

```bash
curl -L -O https://damit.cuni.cz/projects/damit/files/version_0.2.1.tar.gz
sha256sum version_0.2.1.tar.gz   # b1bc2402732e8fd4d1bfe56973a3693bf7ceb6f3bb31e8b5f03926fac59c410c
tar xzf version_0.2.1.tar.gz
cd version_0.2.1/convexinv
sed -i -e 's/^\(#define MAX_LC *\)[0-9]*/\1210/' \
       -e 's/^\(#define MAX_N_OBS *\)[0-9]*/\126612/' \
       -e 's/^\(#define POINTS_MAX *\)[0-9]*/\11549/' constants.h
make convexinv
```

This allows 210 lightcurves, 26,612 observations and 1,549 observations per
lightcurve. Copy the resulting `convexinv/` folder to
`DeLPHI-followup/data/solver-internal-capacity-corrected/20260911/expanded-solver/convexinv/`.

### Model binding file

The current archive supplies the binding file for all 25 retrained networks.
Use it directly, together with the safe inference bundles:

```bash
export DELPHI_NETWORK_ROOT=/path/to/DeLPHI-followup/data/retrain-20261030/artifact-root
export DELPHI_MODEL_BINDING="$DELPHI_NETWORK_ROOT/model-binding.json"
export DELPHI_BUNDLE_ROOT="$DELPHI_NETWORK_ROOT/bundles"
```

Keep the supplied binding and receipts: they verify the selected models and
results by hash. The historical sources archive supplies
`data/external-models-final.json` for the v1.0.0 networks. Its SHA-256 is
`6d041bcbedd9e7ecacc9c4768d0eeb81827065f016dd4db9e982373db3ebfefa`.
Historical `predict_external.py` and period-search `end_to_end.py score-periods`
use it when the network variables are unset. Those runs also need the old
model checkpoints from the superseded v1.0.0 archives. The historical binding
does not select the retrained models.

### Run-to-run GPU differences

Score grids can differ between GPU runs by about `1e-3`, even with the same
code and settings. In our check, a nearly tied candidate changed for 3 of
170 asteroids (about 2%). This does not happen on every repeat; the exact
source of the run-to-run difference is unknown. Saved starts were used for
the retimed fits. Every DeLPHI timing receipt now records the GPU numerics
settings (`gpu_numerics`); keep these receipts when comparing runs.

The paper's reruns set `DELPHI_SCORE_CHUNK_MAX=256`, as recorded in
`retrain-20261030/runners/rerun_all.sh` and Protocol Addendum 3 (revised).
This positive integer caps the number of grid points scored at once, lowering
peak GPU memory. It changes scores only by float32 rounding. The default
prediction commands do not set it. Each receipt records `score_chunk_max`
as an integer, or null when the variable is unset; GPU fields are null on CPU runs.

### Scripts that cannot run from the released files alone

- The run scripts of the inversion comparison (steps 1 to 7 and 9 in its
  [README](../repro/direct_workflow_benchmark/README.md)) check lock files that
  store the authors' paths and computer. The selected runtime sources and
  input locks are now supplied, but these commands still need locks prepared
  for the local installation and the corresponding saved fits. The archives
  do not include every historical original-period workspace or the targets of
  omitted absolute fit symlinks. Historical reliability and withheld-fit
  commands also need the DAMIT observations and a locally built `convexinv`.
  `candidate_accuracy_cost.py` and `score_exploratory_timing.py` also read the raw per-search records.
- The reruns of the lower-quality, ZTF and ALCDEF predictions need the
  DAMIT or external photometry (Fink, ALCDEF, Gaia DR3) and Horizons geometry.
  The full observations and prepared external photometry are not redistributed.
  External-variant reruns also need the original prepared variant inputs and indices.
- Historical predictions with the network variables unset need the old
  checkpoints and, for the census, the full old synthetic data. Download the
  superseded v1.0.0 archives separately; the new historical sources archive
  contains only the small input and source subset described above.
- `median_stats.py` and `median_figures.py` still need the manuscript and macro
  input list; figure generation also needs the original reference figure PDFs.
  The manuscript and reference PDFs are not supplied by the data record.
- The seven-method period search cannot be repeated from these files alone.
  Its `legacy4` method imports `fourier4.py` and `period_pilot.py`, which are
  not supplied. The documented Fourier and phase-dispersion searches use
  the included code.

## 5. Retraining the networks

The training sequence is run with the `delphi-k3` command (each subcommand has
`--help`) on a clean Git checkout, whose commit it records. It needs
`retrain-20261030/synthetic/` from the v1.1.0 data record, the DAMIT snapshot, the catalog
and the cross-validation assignment of `repro/data/damit-20250610T000301Z/`
(`catalog.jsonl`, `publication-splits-v2.3.json`), and a CUDA GPU.

1. `delphi-k3 train-publication-synthetic --train-directory retrain-20261030/synthetic/train
   --validation-directory retrain-20261030/synthetic/validation --seed <seed> ...` for the
   seeds 17, 42, 137, 777 and 2027.
2. `delphi-k3 train-publication-real-oof --pretrained-checkpoint <network of step 1>
   --synthetic-train-directory retrain-20261030/synthetic/train --catalog ... --dump-root
   damit-20250610T000301Z --splits ... --fold <run> --seed <seed> ...` for each of
   the five cross-validation runs and five seeds.
3. `evaluate-publication-real-oof` for each network, then
   `evaluate-publication-real-oof-ensemble --seeds 17 42 137 777 2027` for the
   combined five networks of each run.
4. The controls: `evaluate-publication-real-oof-input-swap`,
   `train-publication-synthetic-label-shuffle` with
   `evaluate-publication-synthetic-label-shuffle`, and
   `evaluate-publication-real-oof-diagnostic --control <name>`.

`python -m repro.export_k3_bundles` turns trained checkpoints into network
archives like `k3-oof-fold-*.tar.gz`. `delphi-k3 train-smoke --objects 8
--output <dir> --allow-dirty` exercises the code path on a few asteroids; its
numbers mean nothing.

Follow `retrain-20261030/PROTOCOL.md` and its addenda for the corrected
simulations and the recorded training sequence. The networks were trained
once on the NVIDIA RTX 5070 Ti with Python 3.12.15 and PyTorch 2.12.1
(CUDA 13.0). Training wall times are recorded in `retrain-20261030/train-times.tsv`
and the attempts in `RUNLOG.md`; per-epoch histories are in
`retrain-20261030/artifact-root/models/*.jsonl`.

To regenerate the corrected simulations, first run the input checks:

```bash
python -B -m repro.generate_k3_synthetic --check \
  --frozen-dir /path/to/original-v1/synthetic \
  --dump-root /path/to/damit-20250610T000301Z \
  --noise-bank /path/to/noise_bank_seed0_n250.npz
```

All three paths are required. Use the original frozen simulations and residual
noise bank: the recipe hashes, seeds and donor pairs are checked before generation.
For generation, use the same arguments with `--output-dir <new-directory>`
in place of `--check`. The synthetic evaluation Python entry points accept
`allow_legacy=True` to read v1 shards explicitly; training requires v2 shards.

## 6. What cannot be reproduced exactly

- **Retraining.** The released networks and per-asteroid records reproduce
  the reported summaries. The corrected networks were trained once; Appendix A
  does not establish repeatability of retraining. Results may differ on other
  hardware or software versions. Table 9 reports variation across the five seeds.
- **Run times** depend on the computer. All timings printed in the revised
  paper were measured on one PC: Intel Core Ultra 7 270K Plus with an NVIDIA
  RTX 5070 Ti (Appendix E). Bootstrap intervals of times describe the asteroids
  on that computer only.
- **Saved convex-inversion fits.** Newly written fits are in
  `delphi-retrain-benchmark-fits-v1.1.0.tar.gz`; revised scores, locks and
  timing receipts are in `delphi-retrain-results-v1.1.0.tar.gz`. The retained
  `delphi-benchmark-fits-v1.0.0.tar.gz` and
  `delphi-own-period-results-v1.0.0.tar.gz` preserve classical and historical
  evidence. They do not replace the retrained results. Some historical
  workspaces and reused fit symlink targets are absent; the selected historical
  sources and input locks do not make every old fit rerunnable. See the
  [commands with extra prerequisites](#scripts-that-cannot-run-from-the-released-files-alone).
- **DAMIT data** are not redistributed. The DAMIT export of 1 June 2025 matches
  the files used (Section 4), but later DAMIT versions differ.
- **External photometry** (ZTF through Fink, ALCDEF, Gaia DR3) and Horizons
  geometry are not redistributed. The data record holds the prepared inputs'
  checksums, the predictions and the scores. Prepared external variant inputs
  and their indices must also be obtained separately for those reruns.
- **Figures.** The current figures are in the new results archive, with
  runners and receipts. `repro/own_period/median_figures.py` and the rerun
  runners draw the revised network-dependent figures. The median tools still
  require the manuscript and macro input list, and figure generation needs
  the original reference figure PDFs. The manuscript and reference PDFs are
  not included in the data record. Historical versions
  of Figures 13 and 18 were drawn by
  `repro/direct_workflow_benchmark/plot_direct_workflow.py` and
  `repro/period_search/benchmark_end_to_end.py`. The other historical figures were drawn
  by the manuscript's plotting script, which is kept with the manuscript source
  and included unchanged in the Zenodo evidence archive as
  `E/part-b-analyses/figures-code/render_main_figures.py`. It reads the result files listed in
  Section 3 from the folder layout of Section 4 and writes PDF files; it is
  provided as a record of how each figure was drawn rather than as a supported
  tool.
- Several period-search scripts were edited after their runs, and their
  checksums differ from those stored with the results; the
  [period-search README](../repro/period_search/README.md) describes the
  reruns that show the same results.
