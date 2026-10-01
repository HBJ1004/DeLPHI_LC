> **Note added 2026-10-01.** This is a record written at the time of the run
> and is kept unchanged. The scored comparison is in release `paper-v1`,
> folder `part-b-analyses/workflow-benchmark/` of
> `delphi-paper-v1-evidence.tar.gz` (`score.json`, `score-loaded.json` and the
> execution summaries `evaluation/execution.json` and
> `evaluation-loaded/execution.json`). The case directories named below,
> including the 32 set-aside cases in `evaluation-contended/`, are part of the
> raw fits, which are available from the authors on request.

The reference-blind direct benchmark was interrupted on 2026-09-23 after 404
of 840 timed cases. Two unrelated compute-heavy processes started on the same
workstation at approximately 11:33 and 11:37 KST. Their CPU and GPU activity
could have affected the benchmark timings. No reference-pole scores had been
opened.

To avoid using mixed-load timings, the 32 completed case directories whose
`timed-result.json` modification time was at or after 11:30 KST were moved,
without deletion, from `evaluation/cases/` to `evaluation-contended/cases/`.
Their matching logs were moved to `evaluation-contended/logs/`. The 11:30
cutoff is three minutes before the first observed competing job. The 372
earlier completed cases remain in `evaluation/cases/` and are bound to the
same frozen evaluation lock. No completed result was edited.

After the competing jobs ended, the benchmark runner reran the 32 set-aside
cases and finished all remaining cases. The final cold-start execution has
840 of 840 cases. Its score was computed only from `evaluation/cases/` after
all cases finished. The 32 set-aside cases remain in `evaluation-contended/`
as evidence of the interruption and did not enter the scored comparison.

A separate 840-case batch timing run kept the neural models loaded once per
fold. It ran after the host was clear. Its selected poles and fitted RMS values
match the cold-start run exactly for every object and arm. The batch run has
its own execution and score files under `evaluation-loaded/` and
`score-loaded.json`.
