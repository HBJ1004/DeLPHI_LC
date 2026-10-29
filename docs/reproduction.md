# Checking the results of the paper

This guide is for readers who want to check the numbers, tables and figures of
Jo, Ishiguro and Lee, "DeLPHI: Pole-Axis Candidates for Asteroid Lightcurve
Inversion" (submitted to PSJ). It explains what to download, how to recompute
the main results from the released predictions without rerunning any analysis,
which file and script lie behind each result, and how to arrange the files to
rerun the analyses.

**Shortest route.** Use this repository at tag `v1.0.0` and the
[Zenodo data record, DOI 10.5281/zenodo.23757942](https://doi.org/10.5281/zenodo.23757942).
Download the own-period results and `SHA256SUMS` using the
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

The code at tag `v1.0.0` provides the Paper I analyses with our own rotation
periods, using the frozen networks. Data, predictions and fits are in the
single [Zenodo data record](https://doi.org/10.5281/zenodo.23757942). Its
`own-period-20261021/PROTOCOL.md` and the protocols in `census7/`,
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
| `PYTHON` | Python executable; default `python`. |
| `CONVEXINV_PYTHON` | Interpreter for `v67_queue.sh` and `v67_finalize.sh`; defaults to `PYTHON`. Set it to the pinned interpreter recorded by the inversion locks. |
| `GPU_LOCK` | Optional lock-file path. The shell GPU steps use `flock "$GPU_LOCK"` when set. |
| `GPU_WAIT` | Optional Bash script, called before each shell GPU step. |
| `LOCK` | Optional lock-file path for the Python runners `refine_chain_post.py` and `temporal.py`. When set, they require a held lock inherited on file descriptor 9; when unset or empty, they skip that check. This is separate from the shell drivers' `GPU_LOCK`. |

For example, with the layout below:

```bash
export DELPHI_REPO=/path/to/DeLPHI-followup/source
export DELPHI_DATA=/path/to/DeLPHI-followup/data/own-period-20261021
export PYTHON=/path/to/venv/bin/python
# For locked inversion reruns, also set CONVEXINV_PYTHON to the recorded interpreter.
```

Python data paths are resolved from the current `DELPHI_DATA` at use time,
including paths imported from another runner and command-line defaults.
The checkout can live independently of the data tree. Keep the data record's
relative layout: earlier data folders are siblings of the own-period folder;
`inputs/` and `experiments/` are two levels above it, and the DAMIT snapshot
is three levels above it. Source-file paths used to import or hash this
checkout still come from the source location.

Frozen evidence also refers to historical code folders (`source-op-maps`,
`source-op-lowq`, `source-op-reliability`, `source-op-nettests`, `source-op-v67`
and `source-bench-8a67c52`) alongside `inputs/` and `data/`. Preserve the
source files and hashes named in the record: inversion reruns verify their
runtime sources and environment, and median tools hash or extract functions
from the historical renderers. Setting `DELPHI_DATA` to the recorded
`data/own-period-20261021/` folder preserves that input layout.

In the table, **O/** means the `DELPHI_DATA` folder itself. Other paths are
also relative to `DELPHI_DATA`; `../` reaches its parent `data/` folder.
Script names without a prefix are in `repro/own_period/`.
CLI-selected output paths below describe the recorded runs.

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
  --figures candidate-count inversion-benchmark --out-dir /path/to/median-figures
```

## 1. What to download

Use the code from the sole software release
[`v1.0.0`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.0.0).
All files below are in the single
[Zenodo data record](https://doi.org/10.5281/zenodo.23757942), under
`https://zenodo.org/records/23757942/files/<name>`.

| File | Contents and extraction layout |
|---|---|
| [`delphi-own-period-results-v1.0.0.tar.gz`](https://zenodo.org/records/23757942/files/delphi-own-period-results-v1.0.0.tar.gz) | Paper I revision predictions, scores, protocols, median statistics and figures; extracts to `own-period-20261021/`, the folder `DELPHI_DATA` points to |
| [`delphi-benchmark-fits-v1.0.0.tar.gz`](https://zenodo.org/records/23757942/files/delphi-benchmark-fits-v1.0.0.tar.gz) | Saved `convexinv` fits of the inversion benchmark; extracts into the same `own-period-20261021/` tree |
| [`delphi-supporting-results-v1.0.0.tar.gz`](https://zenodo.org/records/23757942/files/delphi-supporting-results-v1.0.0.tar.gz) | Other result folders cited by the paper; extracts to sibling folders of `own-period-20261021/` |
| [`delphi-paper-v1-evidence.tar.gz`](https://zenodo.org/records/23757942/files/delphi-paper-v1-evidence.tar.gz) | Supporting validation and original-period analyses, unchanged bytes; extracts to `delphi-paper-v1-evidence/`, containing `part-a-validation/` and `part-b-analyses/` (abbreviated **A/** and **B/** below) |
| [`delphi-k3-artifacts-v1.0.0.tar.gz`](https://zenodo.org/records/23757942/files/delphi-k3-artifacts-v1.0.0.tar.gz) | Extracts to `artifact-root/`: predictions and score maps (`evaluations/`), 25 trained checkpoints and logs (`models/`), controls and original summary (`release/`) |
| [`delphi-k3-synthetic-data-v1.0.0.tar.gz`](https://zenodo.org/records/23757942/files/delphi-k3-synthetic-data-v1.0.0.tar.gz) | Extracts to `synthetic-data/`: 20,000 training, 2,000 validation and 2,000 test simulated asteroids |
| [`k3-oof-fold-0.tar.gz`](https://zenodo.org/records/23757942/files/k3-oof-fold-0.tar.gz), [`k3-oof-fold-1.tar.gz`](https://zenodo.org/records/23757942/files/k3-oof-fold-1.tar.gz), [`k3-oof-fold-2.tar.gz`](https://zenodo.org/records/23757942/files/k3-oof-fold-2.tar.gz), [`k3-oof-fold-3.tar.gz`](https://zenodo.org/records/23757942/files/k3-oof-fold-3.tar.gz), [`k3-oof-fold-4.tar.gz`](https://zenodo.org/records/23757942/files/k3-oof-fold-4.tar.gz) | One five-network safe inference bundle per test fold; each extracts to `k3-oof-fold-<fold>/`, with `bundle.json` and five `seed-*.safetensors` files |
| [`parity.json`](https://zenodo.org/records/23757942/files/parity.json) | Stored inference-bundle parity verification |
| [`publication-archive-manifest.json`](https://zenodo.org/records/23757942/files/publication-archive-manifest.json) | Unchanged original archive manifest, used for model provenance |
| [`release-manifest.json`](https://zenodo.org/records/23757942/files/release-manifest.json) | Original release provenance |
| [`DeLPHI-v1.0-publication-archive-verification.json`](https://zenodo.org/records/23757942/files/DeLPHI-v1.0-publication-archive-verification.json) | Stored archive verification |
| [`DeLPHI-v1.0-publication-package-determinism.json`](https://zenodo.org/records/23757942/files/DeLPHI-v1.0-publication-package-determinism.json) | Stored package determinism verification |
| [`README.md`](https://zenodo.org/records/23757942/files/README.md) | Data record inventory and layout |
| [`SHA256SUMS`](https://zenodo.org/records/23757942/files/SHA256SUMS) | Checksums for the files distributed in this data record |

The original artifacts, synthetic data, fold bundles and small verification
JSON files retain their recorded bytes and hashes. The original manifest and
verification reports describe the historical package, including files that
are no longer distributed. Use the data record's `SHA256SUMS` to verify the
current downloads; do not use the historical four-archive package verifier as
a check of this data record.

For the complete set, start in an empty folder:

```bash
mkdir delphi-check && cd delphi-check
git clone --branch v1.0.0 https://github.com/HBJ1004/DeLPHI_LC.git DeLPHI_LC
mkdir downloads
cd downloads
URL=https://zenodo.org/records/23757942/files
for f in delphi-own-period-results-v1.0.0.tar.gz delphi-benchmark-fits-v1.0.0.tar.gz \
    delphi-supporting-results-v1.0.0.tar.gz delphi-paper-v1-evidence.tar.gz \
    delphi-k3-artifacts-v1.0.0.tar.gz delphi-k3-synthetic-data-v1.0.0.tar.gz \
    k3-oof-fold-0.tar.gz k3-oof-fold-1.tar.gz k3-oof-fold-2.tar.gz \
    k3-oof-fold-3.tar.gz k3-oof-fold-4.tar.gz parity.json \
    publication-archive-manifest.json release-manifest.json \
    DeLPHI-v1.0-publication-archive-verification.json \
    DeLPHI-v1.0-publication-package-determinism.json README.md SHA256SUMS; do
  curl --fail -L -O "$URL/$f"
done
sha256sum --check SHA256SUMS
cd ..
```

Leave out files you do not need, and add `--ignore-missing` to check only the
files downloaded. On macOS use `shasum -a 256 --check SHA256SUMS --ignore-missing`.
Each checked file should end in `OK`. Extract the three results archives to
one parent folder so the two own-period archives merge and supporting folders
remain alongside them:

```bash
mkdir -p data
for f in delphi-own-period-results-v1.0.0.tar.gz delphi-benchmark-fits-v1.0.0.tar.gz \
    delphi-supporting-results-v1.0.0.tar.gz; do
  tar xzf "downloads/$f" -C data
done
export DELPHI_DATA="$PWD/data/own-period-20261021"
tar xzf downloads/delphi-paper-v1-evidence.tar.gz
(cd delphi-paper-v1-evidence && sha256sum -c --quiet SHA256SUMS)
tar xzf downloads/delphi-k3-artifacts-v1.0.0.tar.gz
# Retraining and the census also need:
tar xzf downloads/delphi-k3-synthetic-data-v1.0.0.tar.gz
```

### Minimal download for Check A

```bash
mkdir delphi-check && cd delphi-check
git clone --branch v1.0.0 https://github.com/HBJ1004/DeLPHI_LC.git DeLPHI_LC
mkdir downloads data
cd downloads
curl --fail -LO https://zenodo.org/records/23757942/files/delphi-own-period-results-v1.0.0.tar.gz
curl --fail -LO https://zenodo.org/records/23757942/files/SHA256SUMS
sha256sum --check SHA256SUMS --ignore-missing
cd ..
tar xzf downloads/delphi-own-period-results-v1.0.0.tar.gz -C data
export DELPHI_DATA="$PWD/data/own-period-20261021"
```

### Install

Python 3.11 or 3.12 is needed. From `delphi-check/`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # or python3.11
python -m pip install -e './DeLPHI_LC[test,plot]'
delphi-k3 validate-protocol
```

The last command checks that the fixed configuration file of the model,
`repro/k3_redesign_spec.yaml`, is the one used for the paper. It prints

```text
{"schema": "delphi.k3-redesign-spec.v1", "sha256": "84dee816d08a60b7780a6d1d400add16136fa96c0677e84929700cee8bf22a50"}
```

## 2. Quick checks without rerunning anything

Run the checks from `delphi-check/` inside the environment created under
[Install](#install). Check A uses the own-period results; Checks B and C
use the unchanged supporting archives as specified.

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

root = Path(os.environ["DELPHI_DATA"])
main = json.loads((root / "own-period-scores.json").read_text())["main_sample"]
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
asteroids 170: median 13.58 deg, within 20 deg 71.8%
KThreeStandardMean: 24.13 deg (median)
KThreeRandomMean: 32.33 deg (mean_of_randomization_draw_medians)
```

The `Mean` strings in the macro names are historical identifiers. The
`statistic` fields describe the revised median calculation. Inversion
medians are in the `WorkflowDelphiError` and `WorkflowClassicalError` entries
(27.44° and 45.58°); their underlying per-object records are in
`benchmark/score.json`. Timing ratios are in that file and
`benchmark/score-loaded.json`. Do not substitute the original-period evidence
for these own-period results.

### Check B: original-period oracle errors from archived candidate axes

This recomputes the original published-period comparison from the three
stored candidate axes and the DAMIT reference poles, with per-fold, per-seed
and network-time diagnostics. These are comparison evidence; Check A is the
revised own-period result. It needs `artifact-root/` from
`delphi-k3-artifacts-v1.0.0.tar.gz`.

```bash
python - <<'EOF'
import json
import numpy as np

ensemble = np.load("artifact-root/evaluations/real-oof-ensemble.npz")
catalog = "DeLPHI_LC/repro/data/damit-20250610T000301Z/catalog.jsonl"
poles = {row["object_id"]: np.array([s["vector"] for s in row["solutions"]])
         for row in map(json.loads, open(catalog))}

errors = []
for name, axes in zip(ensemble["object_ids"], ensemble["refined_axes"]):
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    ref = poles[str(name)] / np.linalg.norm(poles[str(name)], axis=1, keepdims=True)
    errors.append(np.degrees(np.arccos(min(1.0, np.abs(axes @ ref.T).max()))))
errors = np.array(errors)

print("asteroids %d: mean %.2f, median %.2f, within 20 deg %.1f%%" % (
    len(errors), errors.mean(), np.median(errors), 100 * (errors <= 20).mean()))
print("largest difference from the stored errors: %.0e deg" % np.abs(errors - ensemble["oracle_errors_deg"]).max())
for run in range(5):
    e = errors[ensemble["folds"] == run]
    print("run %d: n %d, mean %.2f, median %.2f" % (run, len(e), e.mean(), np.median(e)))
for seed in (17, 42, 137, 777, 2027):
    e = np.concatenate([np.load("artifact-root/evaluations/real-fold-%d-seed-%d.npz" % (run, seed))["oracle_errors_deg"]
                        for run in range(5)])
    print("seed %d alone: mean %.2f" % (seed, e.mean()))
t = ensemble["inference_wall_seconds"]
print("network time per asteroid: median %.2f s, mean %.2f s" % (np.median(t), t.mean()))
EOF
```

Expected output:

```text
asteroids 170: mean 15.77, median 12.09, within 20 deg 74.1%
largest difference from the stored errors: 2e-06 deg
run 0: n 34, mean 16.09, median 13.39
run 1: n 34, mean 19.17, median 15.79
run 2: n 34, mean 17.48, median 11.62
run 3: n 34, mean 11.43, median 10.47
run 4: n 34, mean 14.69, median 11.92
seed 17 alone: mean 20.93
seed 42 alone: mean 16.16
seed 137 alone: mean 19.81
seed 777 alone: mean 19.52
seed 2027 alone: mean 16.57
network time per asteroid: median 0.51 s, mean 0.61 s
```

These are the original-period mean diagnostics. The run-4 value 14.695 in
`delphi-paper-v1-evidence/part-a-validation/k3-followup/supplement/k3_fold_metrics.csv`
is already rounded; the unrounded mean is 14.6946°. The ensemble file,
`artifact-root/evaluations/real-oof-ensemble.npz`, also holds `grid_oracle_errors_deg`, whose mean
(15.32°) is the oracle error before the final small adjustment of the
candidates (Appendix A).

### Check C: supporting validation tables

`part-a-validation/` carries its own checks (its `VERIFY.md`). From
`delphi-check/delphi-paper-v1-evidence/part-a-validation/`:

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

**Asteroids outside the sample.** For the ALCDEF comparison (Appendix D), the
paper scored each asteroid with the average score map of all 25 networks,
because none of them was trained on these asteroids
([repro/predict_k3_alcdef_transfer.py](../repro/predict_k3_alcdef_transfer.py),
policy `all_25_frozen_models_for_new_identity`). That script reads the 25
networks as PyTorch checkpoints from the data record's `artifact-root/models/`,
with the [model binding file](#model-binding-file) of Section 4. The
prediction command `delphi-k3-predict` uses one set of five networks instead,
which is the supported way to run DeLPHI on a new asteroid.

The result-family table under [Paper I results](#paper-i-results) maps the
revised own-period results (**O/**) to scripts and input files. The table below
keeps the section-to-file map for the unchanged supporting and original-period
comparisons; its section, table and figure numbers identify those comparisons.
Use the O/ map and `median-stats.json` for the revised medians.

Here `A/` and `B/` abbreviate `delphi-paper-v1-evidence/part-a-validation/`
and `delphi-paper-v1-evidence/part-b-analyses/`; `K/` is `artifact-root/`,
all from the Zenodo data record. Scripts are in this repository at tag `v1.0.0`.
"DAMIT" means the June 2025 DAMIT lightcurves and tables (Section 4).

| Paper | Result files | Script | Needs |
|---|---|---|---|
| Section 2, Table 1, Figure 1 (sample, coverage) | `repro/data/damit-20250610T000301Z/catalog.jsonl` and `publication-splits-v2.3.json` (repository); `A/k3-followup/supplement/k3_data_summary.csv`, `A/k3-followup/supplement/k3_observing_structure.csv` | `repro/export_k3_reliability_supplement.py` | DAMIT |
| Section 5.2, Table 2 (DeLPHI), Tables 6 and 7, Section 5.3 | `K/evaluations/real-oof-ensemble.npz`, `K/evaluations/real-fold-<run>-seed-<seed>.npz`, `K/release/publication-summary.json` | training sequence (Section 5); stored summary retained as provenance | Check B |
| Section 5.2 intervals, random-axes band of Figure 6 | `B/interpretation/primary-intervals.json`, `B/interpretation/random-axes-cdf.npz` | `repro/primary_intervals.py` | `repro/baselines_out/` (repository); CPU |
| Section 5.2, Table 2 (standard poles, random axes); reference rows of Section 6.1.2, Table 9 and Figure 7 | `B/baselines/reference-baselines.json`, `B/baselines/reference-baselines-per-object.npz` (also in `repro/baselines_out/`) | `repro/compute_reference_baselines.py` | `K/` ensemble, DAMIT model table, `A/k3-followup/revision/`, `A/evidence/source-records/alcdef-gaia-analysis.json`, Gaia DR3 spin table; CPU |
| Figures 3, 4 and 5 (worked example, oracle-error illustration, candidates on the sky) | `K/evaluations/real-oof-ensemble.npz`, catalog | figure script of the manuscript (see Section 6) | |
| Section 6.1.1, lightcurve exchange and label shuffle | `K/evaluations/*-input-swap.npz`, `K/evaluations/synthetic-test-seed-17.npz`, `K/evaluations/synthetic-label-shuffle-test-seed-17.npz`, `K/release/publication-summary.json` | `delphi-k3` control commands (Section 5) | Check B |
| Section 6.1.1, candidate spread, random baseline | `B/interpretation/label-shuffle-spread.json` | `repro/label_shuffle_spread.py` | `K/evaluations/`; CPU |
| Section 6.1.2, Appendix D, Tables 8 and 9, Figure 7 | `A/k3-followup/revision/` (`lowq-*`, `error-channel-*`), `A/k3-followup/transfer/`, `A/evidence/source-records/` (`mapped-ztf-score.json`, `alcdef-gaia-analysis.json`), `A/k3-followup-results.tex` | lower-quality models: `repro/run_k3_lowq_census.py`, `run_k3_lowq_damit.py`, `analyze_k3_lowq_lineage.py`, `audit_k3_lowq_exposure.py`; ZTF: `repro/prepare_k3_followup_inputs.py`, `prepare_k3_mapped_ztf.py`, `delphi-k3 predict-ztf-external` and `score-ztf-external`; ALCDEF: `repro/fetch_k3_alcdef_periods.py` to `score_k3_alcdef_gaia_transfer.py`; uncertainties removed: `repro/run_k3_error_channel_sensitivity.py`; collected by `repro/export_k3_revision_evidence.py` and `export_k3_followup_transfer.py` | DAMIT (all models), ZTF photometry from the Fink broker, JPL Horizons geometry, ALCDEF, Gaia DR3 tables (download addresses in `repro/fetch_k3_generalization_sources.py`); GPU for the predictions. Check C re-derives the tables |
| Section 6.1.3, Figure 8, Appendix B (noise), Figure 14 | `A/k3-followup/reliability/k3_data_degradation.csv`, `A/k3-followup/supplement/k3_supplement_numbers.json`, `A/evidence/source-records/reliability-scored-rows.json`, `A/evidence/source-records/reliability-sampling-table.csv`; `B/interpretation/sampling-baselines.json` | `repro/run_k3_reliability_study.py`, `export_k3_reliability_publication.py`, `export_k3_reliability_supplement.py`; `repro/sampling_baselines.py <reliability-scored-rows.json>` | DAMIT, `K/` networks; `sampling_baselines.py` CPU only |
| Appendix B, training versus test, reference reassignment, Figure 15 | `A/k3-followup/internal/summary.json`, `A/k3-followup/internal/train-oof-pairs.csv` | `repro/run_k3_internal_validation.py`, `export_k3_internal_validation_report.py` | DAMIT, `K/` networks |
| Sections 6.2.1 and 6.3.1 (mirror ambiguity, latitude, longitude), Figures 9 and 11 | `B/interpretation/mirror-symmetry.json`; `A/k3-followup/revision/lowq-analysis.json`; `A/k3-followup/supplement/k3_reference_strata.csv`, `A/k3-followup/supplement/k3_descriptive_spearman_matrix.csv` | `repro/mirror_symmetry.py`; the two supplement files from `export_k3_reliability_supplement.py` | `K/` ensemble, DAMIT; CPU |
| Section 6.2.2, amplitude-aspect relation, Figure 9 | `B/interpretation/amplitude-aspect.json`, `B/interpretation/amplitude-aspect.npz`, `B/interpretation/amplitude-aspect-paired.json` | `repro/amplitude_aspect_comparison.py`, then `amplitude_aspect_paired.py` | `K/` ensemble, DAMIT; CPU |
| Section 6.2.3, Appendix B, Shapley values, Figure 10 | `B/interpretation/shapley/*.npz`, `B/interpretation/shapley-summary.json` | `repro/group_shapley.py --output-dir <dir>`, then `group_shapley_summary.py` | `K/models/*.pt`, DAMIT; GPU by default (`--device`) |
| Appendix B, input removals, Figure 16 | `B/interpretation/input-removal-errors.csv` (the per-asteroid errors of all runs in one table) | `repro/interpretation_ablations.py --output-dir <dir>` | `K/models/*.pt`, DAMIT; GPU by default |
| Section 6.3.3, Appendix C, period errors, Figure 17 | `A/k3-followup/supplement/k3_supplement_numbers.json` | `repro/run_k3_reliability_study.py`, `export_k3_reliability_supplement.py` | DAMIT, `K/` networks |
| Section 6.4.1, Figure 12 | `B/interpretation/candidate-count.json` | `repro/candidate_count.py` | `K/` ensemble; CPU |
| Section 6.4.2, Appendix E, Tables 3, 4, 10, 11, Figure 13 | `B/workflow-benchmark/` | `repro/direct_workflow_benchmark/` (see its [README](../repro/direct_workflow_benchmark/README.md)) | DAMIT, patched `convexinv`, networks, GPU; these original-period summaries differ from the own-period benchmark and fits in `O/benchmark/` |
| Appendix E.1, withheld-lightcurve test, Table 12 | `A/evidence/source-records/withheld-rows.json`, `A/k3-followup/supplement/k3_withheld_rms.csv` | `repro/run_k3_reliability_study.py`, `export_k3_reliability_supplement.py` | DAMIT, `K/` networks, patched `convexinv` |
| Appendix C, experimental period search, Figure 18 | `B/period-search/results/` | `repro/period_search/` (see its [README](../repro/period_search/README.md)) | DAMIT; GPU; patched `convexinv` for the physical pilot |
| Section 6.5, DAMIT census of 2026-09-30 | `B/damit-census/` (`raw/` holds the DAMIT tables and lightcurves exported that day, `q3_current_classification.csv`, `candidate_lc_density.json`, and the input list of the lower-quality comparison, `lowq-input-index.json`) | `repro/damit_census/classify.py`, `lc_density.py` | DAMIT snapshot tables and lightcurves, `synthetic-data/` |
| Appendix A, Table 5 | `repro/k3_redesign_spec.yaml` | `delphi-k3 validate-protocol`; see [configuration.md](configuration.md) | |

## 4. Folder layout for rerunning the analyses

The analysis scripts under `repro/` locate their inputs and outputs from their
own position, so they need this layout. The name `DeLPHI-followup` of the top
folder is required, because several scripts contain it.

```text
<parent>/
  damit-20250610T000301Z/                  DAMIT snapshot (tables/, files/)
  paper/figures/                           output folder of Figure 13 (create it)
  DeLPHI-followup/
    source/                                this repository at tag v1.0.0
    inputs/
      frozen-artifacts/k3-definitive-7874092/   Zenodo artifact-root/
      frozen-artifacts/k3-2408c563/synthetic/   Zenodo synthetic-data/ (census only)
      training-run/repro/data/damit-20250610T000301Z/
                                           copy of source/repro/data/damit-20250610T000301Z
    data/
      own-period-20261021/                own-period results and benchmark-fits archives
      ...                                sibling folders from supporting-results archive
      interpretation-20260926/             from B/interpretation/
      interpretation-20260930/             from B/interpretation/
      direct-workflow-benchmark/           B/workflow-benchmark/
      damit-postsnapshot-check/            B/damit-census/
      publication-revision-20260917-r2/    A/k3-followup/revision/
      generalization-20260910/             ALCDEF result and Gaia DR3 spin table
      external-models-final.json           written by bind-models (below)
      solver-internal-capacity-corrected/20260911/expanded-solver/convexinv/
                                           patched convexinv (below)
    experiments/period-refinement-20260921/
                                           repro/period_search/*.py and B/period-search/results/
```

### Arrange the archives

Work in an empty folder `<parent>`. Extract the artifacts and synthetic-data
archives there (creating `artifact-root/` and `synthetic-data/`). Extract the
unchanged evidence archive anywhere and set `E` to its folder. Put the
own-period results and fits under `DeLPHI-followup/data/`, with the supporting
results as siblings, as in the download instructions. The commands below
arrange inputs for the retained original-period comparisons; if supporting
results already supply a destination, use that copy rather than nesting a
second copy inside it.

```bash
E=/path/to/delphi-paper-v1-evidence
T=DeLPHI-followup
git clone --branch v1.0.0 https://github.com/HBJ1004/DeLPHI_LC.git $T/source
mkdir -p $T/inputs/frozen-artifacts/k3-2408c563 $T/inputs/training-run/repro/data \
         $T/data/interpretation-20260926 $T/data/interpretation-20260930 \
         $T/data/generalization-20260910/locked-cohort \
         $T/data/generalization-20260910/sources/gaia-dr3-spins-20260911 \
         $T/experiments/period-refinement-20260921/end-to-end paper/figures
mv artifact-root $T/inputs/frozen-artifacts/k3-definitive-7874092
mv synthetic-data $T/inputs/frozen-artifacts/k3-2408c563/synthetic
cp -r $T/source/repro/data/damit-20250610T000301Z $T/inputs/training-run/repro/data/

I=$E/part-b-analyses/interpretation
cp -r $I/primary-intervals.json $I/random-axes-cdf.npz $I/mirror-symmetry.json \
      $I/candidate-count.json $I/shapley $I/shapley-summary.json $T/data/interpretation-20260926/
cp $I/amplitude-aspect.json $I/amplitude-aspect.npz $I/amplitude-aspect-paired.json \
   $I/label-shuffle-spread.json $I/sampling-baselines.json $T/data/interpretation-20260930/
cp -r $E/part-b-analyses/workflow-benchmark $T/data/direct-workflow-benchmark
cp -r $E/part-b-analyses/damit-census $T/data/damit-postsnapshot-check
cp -r $E/part-a-validation/k3-followup/revision $T/data/publication-revision-20260917-r2
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

The period-search score check, the ZTF predictions and the ALCDEF predictions load the 25 networks
through a binding file that records their checksums. Write it from
`DeLPHI-followup/source`:

```bash
python -m repro.prepare_k3_followup_inputs bind-models \
  --artifact-root ../inputs/frozen-artifacts/k3-definitive-7874092 \
  --archive-index ../inputs/frozen-artifacts/k3-definitive-7874092/release/archive-index.json \
  --publication-package-manifest /path/to/downloads/publication-archive-manifest.json \
  --output ../data/external-models-final.json
```

The binding checks model hashes against the unchanged archive manifest;
keep that manifest even though some of its historical package members are no
longer distributed.

### Scripts that cannot run from the released files alone

- The run scripts of the inversion comparison (steps 1 to 7 and 9 in its
  [README](../repro/direct_workflow_benchmark/README.md)) check lock files that
  store the authors' paths and computer, and need the corresponding saved
  fits and frozen source files. The new fits archive supplies the own-period
  benchmark tree, rather than every historical original-period run.
  `candidate_accuracy_cost.py` and `score_exploratory_timing.py` also read the raw per-search records.
- The reruns of the lower-quality, ZTF and ALCDEF predictions need the
  external photometry (Fink, ALCDEF, Gaia DR3) and Horizons geometry, which are
  not redistributed.

## 5. Retraining the networks

The training sequence is run with the `delphi-k3` command (each subcommand has
`--help`) on a clean Git checkout, whose commit it records. It needs
`synthetic-data/` from the Zenodo data record, the DAMIT snapshot, the catalog
and the cross-validation assignment of `repro/data/damit-20250610T000301Z/`
(`catalog.jsonl`, `publication-splits-v2.3.json`), and a CUDA GPU.

1. `delphi-k3 train-publication-synthetic --train-directory synthetic-data/train
   --validation-directory synthetic-data/validation --seed <seed> ...` for the
   seeds 17, 42, 137, 777 and 2027.
2. `delphi-k3 train-publication-real-oof --pretrained-checkpoint <network of step 1>
   --synthetic-train-directory synthetic-data/train --catalog ... --dump-root
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

The computing environment recorded with the trained networks
(`artifact-root/release/environment/environment.json` and `pip-freeze.txt`) is
an Intel Core i5-13600KF CPU under WSL2, an NVIDIA RTX 4070 GPU (12 GB),
Python 3.12 and PyTorch 2.5.1 with CUDA 12.4. The per-epoch training losses are in
`artifact-root/models/*.jsonl`. The total training time was not recorded.

## 6. What cannot be reproduced exactly

- **Training on a GPU is not bit-for-bit repeatable.** Retraining with the same
  seeds gives slightly different networks and results (paper Appendix A).
  The paper therefore reports the spread over the five seeds (Table 7) and
  releases the trained networks and their predictions, which reproduce every
  reported value.
- **Run times** depend on the computer. The paper's times were measured on the
  workstation described in Appendix E (Intel Core i5-13600KF under WSL2, RTX
  4070). Bootstrap intervals of times describe the asteroids on that computer
  only.
- **Saved convex-inversion fits** of the own-period benchmark are distributed
  in `delphi-benchmark-fits-v1.0.0.tar.gz`, alongside the scores and times in
  `delphi-own-period-results-v1.0.0.tar.gz`. Historical original-period notes
  describe a different set of fits; the benchmark-fits archive covers the
  own-period tree.
- **DAMIT data** are not redistributed. The DAMIT export of 1 June 2025 matches
  the files used (Section 4), but later DAMIT versions differ.
- **External photometry** (ZTF through Fink, ALCDEF, Gaia DR3) and Horizons
  geometry are not redistributed. The data record holds the prepared inputs'
  checksums, the predictions and the scores.
- **Figures.** Figures 13 and 18 are drawn by
  `repro/direct_workflow_benchmark/plot_direct_workflow.py` and
  `repro/period_search/benchmark_end_to_end.py`. The other figures were drawn
  by the manuscript's plotting script, which is kept with the manuscript source
  and included unchanged in the Zenodo evidence archive as
  `B/figures-code/render_main_figures.py`. It reads the result files listed in
  Section 3 from the folder layout of Section 4 and writes PDF files; it is
  provided as a record of how each figure was drawn rather than as a supported
  tool.
- Several period-search scripts were edited after their runs, and their
  checksums differ from those stored with the results; the
  [period-search README](../repro/period_search/README.md) describes the
  reruns that show the same results.
