# Exploratory classical-budget comparison, 2026-09-25

This follows the frozen 12-start direct benchmark. Its 140-object evaluation
set had already been scored. The budget scan and the choice to time 18 and 96
classical starts are therefore **post hoc**, not a new blinded test.

The existing full-grid archive was used to reconstruct the best fit from each
prefix of a fixed, widely spaced classical starting-pole order. The saved
12-start fits matched the direct 12-start run exactly on all 140 asteroids.
The 18-start budget was the first to meet the original quality limits by
point estimates. The 96-start budget was the first to meet all three limits
using descriptive, fold-stratified 95% bootstrap bounds. Scanning ten budgets
on the same objects means these intervals do not provide independent
confirmation of the chosen budget.

| Search | Mean DAMIT-pole disagreement | Within 20° | Classical/DeLPHI fitted-RMS ratio | Loaded time/asteroid |
| --- | ---: | ---: | ---: | ---: |
| DeLPHI, 6 starts | 17.32° | 107/140 | — | 3.08 s |
| Classical, 18 starts | 20.01° | 103/140 | 0.979 | 5.76 s |
| Classical, 96 starts | 14.90° | 114/140 | 0.960 | 27.27 s |

The mean-time ratios, classical divided by DeLPHI, are 1.87 (95% object
bootstrap interval 1.70–2.01) for 18 starts and 8.85 (7.97–9.58) for 96
starts. DeLPHI was faster on 136/140 asteroids than the 18-start search and
on all 140 than the 96-start search. One timed repeat per asteroid and arm
was used; the intervals resample asteroids, not repeat executions or
hardware. The three arms were interleaved on one workstation with the same
periods, six solver workers, and model-loaded batch setup. Model setup took
11.24 s total across the five folds, or 0.08 s per asteroid. Adding that to
DeLPHI gives 3.16 s per asteroid and descriptive ratios of 1.82 and 8.63.
The initial Python import, period search, and human review are excluded.

The 18-start classical search has a mean reference error only 2.69° above
DeLPHI, but the 95% interval for that difference is −1.66° to 7.09°. Its
within-20° difference also has a lower bound of −10.7 percentage points,
below the preset −5-point limit. Thus its apparently close average quality
is not established to be comparable by the fixed rule.

The 96-start classical search passes the original one-sided quality rule in
this exploratory scan, but it is not equal in every respect: its selected
fits have about 4% lower residual RMS and its mean DAMIT-pole disagreement
is 2.41° lower than DeLPHI's (95% interval for classical minus DeLPHI,
−6.53° to 1.60°). DeLPHI's large speed gain against 96 starts is therefore
a speed-versus-fit trade-off, not proof of equal physical solution quality.
DAMIT poles are model-derived references, not known truth.

The timing run has 420/420 hashed cases. Every newly timed selected axis
agrees with its saved counterpart to within 0.000002°, and every fitted RMS
matches exactly. The machine-readable records are
`classical-ladder-exploratory.json`, `classical-ladder-intervals.json`, and
`exploratory-loaded-ladder/score.json`. The records are local and are not in
the existing public evidence release.
