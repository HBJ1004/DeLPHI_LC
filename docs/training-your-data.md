# Train on your own data

Use this path when you have **labelled** asteroids: compatible observations, a
known period, and one or more independently derived reference pole axes for
each object. Training a new model does not validate it. Keep an object-disjoint
test set that is never used for model selection.

## 1. Make train and validation JSONL files

Copy [training.example.jsonl](../examples/training.example.jsonl). JSONL means
one complete JSON object per line. The observation fields are exactly those in
the [data-format guide](data-format.md). Add `target_axes`, a nonempty list of
unit `[x, y, z]` axes in ecliptic J2000.

```json
{"schema":"delphi.k3-training-example.v1","object_id":"object-a", "known_period":{"hours":7.25,"provenance":"catalogue-v1"}, "epochs":[...], "target_axes":[[0.5,0.5,0.70710678]]}
```

Put each asteroid in exactly one partition. Different epochs of the same
asteroid must never be split between train and validation. Reserve a third,
object-disjoint test partition before experimenting. A few lightcurves from one
object are not independent training examples.

## 2. Train a new scorer

```bash
python -m repro.train_k3_custom \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --output-directory outputs/custom-k3-seed17 \
  --seed 17 \
  --device cuda
```

Use `--device cpu` if no CUDA GPU is available. The command validates every
row, rejects duplicate/overlapping object IDs, writes an atomic `.pt`
checkpoint and a hash-bound `training-report.json`. Choose a new output
directory for each run. The five allowed seeds are 17, 42, 137, 777, and 2027.

The custom command starts from a new K3 scorer. It does not fine-tune published
safetensors bundles: those are five fold ensembles, not one general-purpose
training checkpoint. The `.pt` checkpoint is for trusted local use only; it
uses Python pickle and must not be loaded from an untrusted source.

## 3. Evaluate honestly

For every held-out object, use three K3 axes and report antipode-aware oracle@3
error only as candidate coverage. The helper is:

```python
from lc_pipeline.k3.evaluation import oracle_at_k_error_deg, summarize_errors

# predictions: shape (3, 3); targets: shape (number_of_reference_axes, 3)
error = oracle_at_k_error_deg(predictions, targets)
print(summarize_errors(all_held_out_errors))
```

The oracle selects the closest candidate using reference labels. It cannot
choose an axis at deployment, prove unique-pole recovery, or demonstrate
inversion acceleration. Report held-out object count, splitting rule, period
and geometry sources, all seeds, and failures. Do not compare custom data with
the frozen DAMIT metric unless the cohort, endpoint, and protocol are identical.

## Advanced work

The published pipeline contains synthetic pretraining, five object-disjoint
outer folds, controls, calibration, and a matched inversion benchmark.
Reproducing it needs external frozen artifacts and raw DAMIT inputs; follow
[reproduction.md](reproduction.md). It is intentionally separate from this
short custom-data route.
