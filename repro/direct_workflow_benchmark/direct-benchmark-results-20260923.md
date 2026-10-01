> **Note added 2026-10-01.** This is a record written at the time of the run
> and is kept unchanged except for the corrected last sentence. Its results are
> in release `paper-v1`, folder `part-b-analyses/workflow-benchmark/` of
> `delphi-paper-v1-evidence.tar.gz`, and are reported in Section 6.4.2 and
> Appendix E of the paper. The per-search case directories mentioned below
> (`evaluation/cases/`, `evaluation-contended/`) are part of the raw fits,
> which are available from the authors on request.

# Fixed-start DeLPHI versus classical inversion, 2026-09-23

The benchmark compares six signed poles from DeLPHI's three axes with 12
fixed classical poles. Neither arm falls back to the full grid. The classical
budget was the smallest of ten fixed budgets to meet the point criteria on
30 development asteroids. The remaining 140 asteroids, 28 from each fold,
were timed three times in both arms before their DAMIT model poles were
opened for scoring. Both arms use the same published periods, enlarged
`convexinv` fitting code, six CPU workers, and fitting settings. DeLPHI uses
the RTX 4070 for neural inference.

| Timing condition | Classical mean | DeLPHI mean | Classical/DeLPHI ratio (95% interval) |
| --- | ---: | ---: | ---: |
| Fresh process and model load per asteroid | 4.07 s | 13.61 s | 0.30 (0.27–0.33) |
| Models kept loaded across a batch | 3.57 s | 2.78 s | 1.29 (1.18–1.39) |

The batch figure excludes the initial Python import and the once-per-fold
model loading. The five model loads took 3.13 s in total, or 0.02 s per
asteroid when spread over 140 objects. Adding those loads gives 2.80 s for
DeLPHI and a descriptive ratio of 1.28. The fresh-process figure includes
startup, model loading, inference, fitting, and output.

Both arms completed on all 140 asteroids. Their selected axes and fitted RMS
values were identical between the two timing modes. Mean disagreement with
the nearest DAMIT model pole was 22.52° for classical starts and 17.32° for
DeLPHI starts. The classical-minus-DeLPHI mean difference was 5.21° (95%
interval 0.74–9.63°). Agreement within 20° was 98/140 and 107/140,
respectively. The geometric mean classical/DeLPHI fitted-RMS ratio was 0.998
(0.986–1.010).

The fixed quality-match rule failed on the evaluation set: the upper bound
for the mean reference-error difference exceeded the allowed 3°, and the
lower bound for the difference in the within-20° fraction fell below −5
percentage points. Thus the loaded batch is faster than this particular
12-start search, but this is not a demonstrated speed gain against a
classical search matched for reference-pole agreement. DAMIT poles are
model-derived references, not known physical truth. Both arms use the DAMIT
published period and do not time period determination or human review.

The cold-start result excludes 32 cases that overlapped an unrelated
compute-heavy job. Those case directories remain under
`evaluation-contended/`; the clean reruns are in `evaluation/`. See
`timing-interruption-20260923.md`. The completed scored files are `score.json`
and `score-loaded.json`. Their SHA-256 digests are
`8c2d78d70d5ffe0940dba440154eb09da23ab53d2dd05e2524fbff04d3a9cd8f`
and
`be9218a7b3f800ca54f082101b236881cd9be63cb0fb6459fb73cdc819934002`,
respectively. These records are in release `paper-v1`
(`part-b-analyses/workflow-benchmark/`). [Corrected 2026-10-01. The note first
said that the records were local and had to be put in a versioned public
release before the new manuscript numbers were submitted.]
