# Experimental period search (manuscript Appendix C)

These scripts produced the experimental period search of Appendix C: the
Fourier-series and phase-dispersion searches on the 170 asteroids, the
candidate lists with P/2, P and 2P, the 20-asteroid pilot of a short physical
fit, the check with the DeLPHI scores, and the Delta-chi^2 period intervals.
None of these periods was used in the pole results of the paper, which use the
published DAMIT period.

The results are in release
[`paper-v1`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/paper-v1),
archive `delphi-paper-v1-evidence.tar.gz`, folder
`part-b-analyses/period-search/results/`. The same archive keeps a copy of
these scripts in `part-b-analyses/period-search/code/`.

## Scripts

| Script | Role | Result file |
|---|---|---|
| `search.py` | Period-search methods, including the GPU Fourier search (`fourier_gpu`) and phase-dispersion minimization (`pdm`) | (library) |
| `run_search.py` | Runs the searches on the 170 asteroids and times them | `warm170-first.jsonl` |
| `end_to_end.py candidates` | Takes the best Fourier and phase-dispersion periods, adds P/2 and 2P, keeps 2 to 100 h and merges values within 1% | `candidates-v1.jsonl` |
| `end_to_end.py screen` | Short physical fit (50 `convexinv` iterations from the six standard poles) for each candidate period of the 20 pilot asteroids | `pilot-screen-v1.jsonl` |
| `end_to_end.py resummarize` | Selects the period of lowest chi-squared and marks ambiguous cases, from the saved fits | `pilot-screen-v2.jsonl` |
| `end_to_end.py score-periods` | Scores each candidate period of the pilot with the DeLPHI networks | `pilot-period-scores-v1.jsonl` |
| `physical.py` | `convexinv` interface used by the physical fit | (library) |
| `benchmark_end_to_end.py` | Summary numbers and Figure 18 (`accuracy-cost.pdf`) | `benchmark-summary.json` |
| `period_uncertainty.py` | Delta-chi^2 interval of every candidate period | `period-uncertainty-v1.jsonl` |
| `period_uncertainty_summary.py` | Comparison of these intervals with the DAMIT periods | `period-uncertainty-v1.summary.json` |

## Where the scripts must be placed

The scripts locate their inputs relative to their own folder. They were run
from `DeLPHI-followup/experiments/period-refinement-20260921/` in the folder
layout described in [docs/reproduction.md](../../docs/reproduction.md), and
they must be copied there to run again:

```text
<parent>/
  damit-20250610T000301Z/              DAMIT snapshot (files/asteroid_<id>/lc.txt)
  DeLPHI-followup/                     this name is required
    source/                            this repository
    inputs/training-run/repro/data/damit-20250610T000301Z/
                                       copy of source/repro/data/damit-20250610T000301Z
    inputs/frozen-artifacts/k3-definitive-7874092/
                                       release v1.0.0 artifact-root/ (score-periods only)
    data/external-models-final.json    model binding file (score-periods only)
    data/solver-internal-capacity-corrected/20260911/expanded-solver/convexinv/
                                       convexinv with enlarged storage limits (screen only)
    experiments/period-refinement-20260921/
      *.py                             these scripts
      warm170-first.jsonl              from results/
      end-to-end/                      the other files of results/
```

Requirements by step:

- `run_search.py` needs the DAMIT lightcurves and the catalog. The GPU
  Fourier search needs CUDA.
- `end_to_end.py screen` needs `convexinv` compiled from DAMIT's
  `version_0.2.1.tar.gz` with the three storage limits raised as described in
  [docs/reproduction.md](../../docs/reproduction.md).
- `end_to_end.py score-periods` needs CUDA, the trained networks of release
  `v1.0.0` (`artifact-root/models/`), and `data/external-models-final.json`,
  which `python -m repro.prepare_k3_followup_inputs bind-models` writes (see
  [docs/reproduction.md](../../docs/reproduction.md)).
- `benchmark_end_to_end.py`, `period_uncertainty.py` and
  `period_uncertainty_summary.py` need only the result files, the catalog and
  the DAMIT lightcurves.

Every script refuses to overwrite an existing output, except
`period_uncertainty_summary.py`, which always reads
`end-to-end/period-uncertainty-v1.jsonl` and rewrites its summary.

## Commands

Run from `DeLPHI-followup/experiments/period-refinement-20260921/` in the Python
environment where the repository is installed (the scripts add
`DeLPHI-followup/source` to the import path themselves where they need it). New
output names are used so that the released files stay untouched for comparison.

```bash
python run_search.py --limit 170 --repeats 1 --mode warm --methods fourier_gpu,pdm --output warm170-check.jsonl
python end_to_end.py candidates --search-results warm170-first.jsonl --output end-to-end/candidates-check.jsonl
python end_to_end.py screen --candidates end-to-end/candidates-v1.jsonl --output end-to-end/pilot-screen-check.jsonl
python end_to_end.py resummarize --screen end-to-end/pilot-screen-check.jsonl --output end-to-end/pilot-screen-check-v2.jsonl
python end_to_end.py score-periods --candidates end-to-end/candidates-v1.jsonl --output end-to-end/pilot-period-scores-check.jsonl
python benchmark_end_to_end.py --search warm170-first.jsonl \
  --candidates end-to-end/candidates-v1.jsonl --screen end-to-end/pilot-screen-v2.jsonl \
  --scores end-to-end/pilot-period-scores-v1.jsonl --output-directory end-to-end/benchmark-check
python period_uncertainty.py --output end-to-end/period-uncertainty-check.jsonl
```

The original search run used seven methods (`--methods` default). One of them,
`legacy4`, imports two older scripts that are not part of this repository, and
only `fourier_gpu` and `pdm` enter the candidate lists and the paper, so the
command above runs those two. Run times depend on the computer, and the paper's
times were measured on the computer described in Appendix E.

## Provenance

`run_search.py` is the version that was run. `search.py`, `end_to_end.py` and
`benchmark_end_to_end.py` were edited on 2026-09-21 after the runs, and no
record of that edit survives, so their checksums differ from those stored in
the result metadata. On 2026-09-29 we reran the Fourier and phase-dispersion
searches of the current `search.py` on all 170 asteroids and rebuilt the
candidate lists with the current `end_to_end.py`. Every candidate list has the
same number of periods as `candidates-v1.jsonl`, and the periods agree to within
a relative difference of 1.6e-7. On 2026-09-30 we also reran the physical pilot
with the current `end_to_end.py` and the same settings
(`pilot-screen-rerun-20260930.jsonl` and `pilot-screen-rerun-20260930-v2.jsonl`
in the release). All 20 asteroids selected the same period with the same
chi-squared as `pilot-screen-v2.jsonl`, giving the same 7 of 20 correct and 10
ambiguous results. Only the run times differ.

`period_uncertainty.py` and `period_uncertainty_summary.py` are the versions
that produced `period-uncertainty-v1.jsonl` and its summary.
