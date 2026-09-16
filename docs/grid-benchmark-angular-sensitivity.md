# Post-hoc angular sensitivity of the known-period grid benchmark

This is a descriptive analysis of the sealed v3 full-run score. It was defined
after the primary result was known and cannot replace or rescue the
prespecified 20-degree recovery criterion.

Source: `full-score.json`, SHA-256
`e4f0aae5f69871ad9d6748363a66652f71ac46121b70ec963e4f0b22980cc9da`.
The comparison uses the cold `classical20` and `guided20` arms for all 170
objects. An object is counted as recovered when at least two of its three
selected axes fall within the stated antipode-aware angular threshold of an
admissible DAMIT reference axis.

| Threshold | Classical | Guided | Guided minus classical | Guided-only | Classical-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 deg | 106 | 102 | -2.35 pp | 18 | 22 |
| 15 deg | 127 | 119 | -4.71 pp | 14 | 22 |
| 20 deg | 135 | 134 | -0.59 pp | 14 | 15 |
| 25 deg | 140 | 140 | 0.00 pp | 13 | 13 |
| 30 deg | 141 | 141 | 0.00 pp | 13 | 13 |
| 40 deg | 151 | 148 | -1.76 pp | 9 | 12 |

At 20 degrees, the exact paired two-sided McNemar/binomial p-value for 14 versus
15 discordant objects is 1.0. This does not establish equality or
noninferiority; it shows no directional evidence in the observed discordances.

The per-object mean selected-axis errors were 15.25 degrees for the classical
arm and 16.49 degrees for the guided arm. Their paired mean difference was
+1.24 degrees. A 10,000-resample object-clustered, fold-stratified bootstrap
(seed 20260915) gave a descriptive 95% percentile interval of -2.03 to +4.68
degrees. The paired median difference was 0.0 degrees. Guided error was lower
for 83 objects, equal for 22, and higher for 65.

The aggregate similarity hides object-level disagreement. Twenty-nine objects
changed reference-agreement status at the 20-degree threshold; 29 had an
absolute paired difference above 20 degrees, and 19 exceeded 40 degrees. Counts
above 30 degrees were identical (29 in each arm), but they were not always the
same objects. DAMIT reference uncertainty can affect classifications near the
threshold, while larger differences show that the two initialization strategies
can also select different inversion basins. Because DAMIT axes are model-derived
references and lightcurve inversion can admit ambiguous solutions, these cases
do not establish which returned pole is physically correct.

This analysis supports an estimation-focused interpretation: the observed
20-degree reference-agreement rates are nearly equal, their paired direction is
balanced, and the continuous angular-difference estimate is uncertain. It also
shows that guided and full-grid initialization can return different solution
basins for individual objects. The separately measured roughly 1.9% increase in
guided selected-fit RMS remains independent of DAMIT pole-reference uncertainty,
but it likewise does not determine physical pole correctness.

For practical use, DeLPHI candidates should be treated as proposed starting
regions rather than exclusive bounds on the solution. Users should retain and
inspect alternative fitted poles, compare their lightcurve residuals and
stability, and use a broader-grid fallback when solutions disagree or when a
single pole is required. Independent constraints such as occultations, radar,
resolved imaging, or thermophysical evidence should be used when physical-pole
identification matters.
