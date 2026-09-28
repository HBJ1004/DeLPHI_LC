# Fixed-start pole-search comparison (manuscript Section 6.3)

These scripts produced the fixed-start comparison of six DeLPHI-guided starts
with 12 classical starts, the cold and loaded timings, and the exploratory
18- and 96-start comparison. They are byte-identical copies of the scripts in
`DeLPHI-followup/data/direct-workflow-benchmark/`, where they were run. Every
execution record stores the SHA-256 of the script that wrote it, and these
copies carry the same checksums.

The scripts read their locks and write their records relative to their own
folder (`ROOT = Path(__file__).resolve().parent`), so to rerun them, place them
next to the evidence records of the release archive. They import
`lc_pipeline.grid_benchmark` and `lc_pipeline.workflow_benchmark`.

- `run_direct_benchmark.py`: locked 12-start and DeLPHI runs with cold timing
- `run_loaded_benchmark.py`: the same comparison with the networks kept loaded
- `explore_classical_ladder.py`, `score_classical_ladder.py`: choice of the classical budget
- `time_exploratory_budgets.py`, `score_exploratory_timing.py`: exploratory 18/96-start timing
- `plot_direct_workflow.py`: manuscript figure
- `*-results-*.md`, `timing-interruption-20260923.md`: run notes
