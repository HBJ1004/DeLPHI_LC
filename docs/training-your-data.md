# Train on your own data

Use this path when you have **labelled** asteroids: compatible observations, a
known period, and one or more independently derived reference pole axes for
each object. Training a new model does not validate it. Keep an object-disjoint
test set that is never used for model selection.

## 1. Make train and validation JSONL files

Copy [training.example.jsonl](../examples/training.example.jsonl), which shows
the format of one row only (one asteroid with a two-point lightcurve); it is not
enough data to train on. JSONL means
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
row, rejects duplicate or overlapping object IDs, and writes a `.pt`
checkpoint, a `training-report.json`, and an `inference-bundle/` directory
with the weights in the safe `safetensors` format. Choose a new output
directory for each run. The allowed seeds are 17, 42, 137, 777, and 2027, the
five seeds of the paper; the configuration accepts only these, which keeps runs
comparable with the published networks.

**What this command does, and how it differs from the paper.** It trains one
network from random initial weights on your data alone, in a single stage, with
the network and optimizer settings of the paper (learning rate, batch size,
negative axes, loss) and at most 100 passes with early stopping after 15 passes
without improvement on your validation file (the limits of the simulated stage,
`synthetic_max_epochs` and `synthetic_patience` in the configuration guide). It does **not** reproduce the
paper's training, which (Section 4 of the paper) first trained on 20,000
simulated asteroids, then adjusted each network on the real asteroids mixed with
simulated ones at a ratio of 3 to 1 with a reduced learning rate for the part
that describes the photometry, and finally averaged the score maps of five
networks trained with different seeds. A custom run therefore gives one network,
not an average of five, and is expected to be less accurate with small data
sets. To change settings, see the [configuration guide](configuration.md).

**How much data.** We have not tested a minimum. For scale, each network of the
paper saw 108 real asteroids after pretraining on 20,000 simulated ones. With a
few dozen asteroids, expect a network that mostly learns the distribution of
poles in your training set; use the controls of Section 4 below to check.

**Run time** depends on the number of asteroids and observations and has not
been benchmarked for custom data. Start with a small validation run on CPU to
check your files, then train on a GPU.

The command does not fine-tune the published networks: those are five sets of
five networks, one set per cross-validation run, not one general-purpose
checkpoint. The `.pt` checkpoint is for trusted local use only; it uses Python
pickle and must not be loaded from an untrusted source.

## 3. Predict with the trained model

Use the same observation JSON format as the published weights:

```bash
delphi-k3-predict \
  --bundle outputs/custom-k3-seed17/inference-bundle \
  --input data/my_asteroid.json \
  --output outputs/my_asteroid_prediction.json \
  --device cuda
```

Custom output has `calibration: null` and `risk_deg: null`. Its three axes are
an unordered proposal set. Internal scores are retained for diagnostics but
have not been validated as a physical-pole ranking. Establish performance on
your untouched test objects before using the model scientifically.

## 4. Evaluate honestly

For every asteroid of your untouched test set, predict its three candidate
axes and compute the oracle error, the smallest angle between any candidate
axis and any of its reference poles, with an axis and its opposite direction
treated as the same (Section 5.1 of the paper):

```python
import json
import numpy as np
from lc_pipeline.k3.evaluation import oracle_at_k_error_deg, summarize_errors

# reference_poles: {object_id: [[x, y, z], ...]} unit vectors, ecliptic J2000
errors = []
for object_id, poles in reference_poles.items():
    prediction = json.load(open(f"outputs/{object_id}_prediction.json"))
    axes = np.array([axis["axis_xyz"] for axis in prediction["axes"]])
    errors.append(oracle_at_k_error_deg(axes, np.array(poles)))
print(summarize_errors(errors))
```

Compare the result with the six standard starting poles and with three random
axes on the same asteroids, as the paper does; the oracle error of a network
that ignores the lightcurves can be moderate if your poles are clustered. The
oracle error uses the reference poles after the prediction to pick the closest
candidate. It is a measure of how well the candidates cover the reference
solutions, not the accuracy of a single pole chosen by the network, and it does
not show that an inversion becomes faster. Report held-out object count, splitting rule, period
and geometry sources, all seeds, and failures. Do not compare custom data with
the frozen DAMIT metric unless the cohort, endpoint, and protocol are identical.

## Advanced work

The paper's full procedure (simulated pretraining, five cross-validation runs,
five seeds, controls, and the inversion comparison) is described in
[reproduction.md](reproduction.md). It needs the released archives and the DAMIT
data and is separate from this short custom-data route.
