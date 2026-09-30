# Experimental period search (manuscript Appendix C)

These scripts produced the experimental period search of Appendix C and its
period uncertainties. They are byte-identical copies of the scripts in
`DeLPHI-followup/experiments/period-refinement-20260921/`, where they were run,
and they read and write paths relative to that folder. The result files are in
the release archive `publication-evidence-2026-09-28`, folder `period-search/`.

- `search.py`: Fourier-series and phase-dispersion searches (`estimate`)
- `run_search.py`: 170-asteroid search run (`warm170-first.jsonl`)
- `end_to_end.py`: candidate lists with P/2, P and 2P (`candidates-v1.jsonl`),
  the physical pilot (`pilot-screen-v1.jsonl`, re-summarized as
  `pilot-screen-v2.jsonl`) and the DeLPHI score check (`pilot-period-scores-v1.jsonl`)
- `physical.py`: quick `convexinv` fits used by the physical pilot
- `benchmark_end_to_end.py`: summary of the four result files (`benchmark-summary.json`)
- `period_uncertainty.py`, `period_uncertainty_summary.py`: Delta-chi^2 period
  intervals for every candidate and their comparison with the DAMIT periods

## Provenance

`run_search.py` is the version that was run. `search.py`, `end_to_end.py` and
`benchmark_end_to_end.py` were edited on 2026-09-21 after the runs, and no record
of that edit survives, so their checksums differ from those stored in the result
metadata. On 2026-09-29 we reran the Fourier and phase-dispersion searches of
the current `search.py` on all 170 asteroids and rebuilt the candidate lists with
the current `end_to_end.py`. Every candidate list has the same number of periods
as `candidates-v1.jsonl`, and the periods agree to within a relative difference
of 1.6e-7. On 2026-09-30 we also reran the physical pilot with the current
`end_to_end.py` and the same settings. All 20 asteroids selected the same
period with the same chi-squared as `pilot-screen-v2.jsonl`, giving the same
7 of 20 correct and 10 ambiguous results. Only the run times differ.

`period_uncertainty.py` and `period_uncertainty_summary.py` are the versions
that produced `period-uncertainty-v1.jsonl` and its summary.
