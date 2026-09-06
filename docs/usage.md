# Use published K3 weights

This guide starts with one asteroid that has a known rotation period, relative
photometry, and real observing geometry. DeLPHI does not calculate ephemerides
or discover the period in this path. If either is unknown, resolve that first.

## 1. Make an observation file

Copy [observations.example.json](../examples/observations.example.json) to
`my_asteroid.json`, then replace every example value with your data. The full
field definitions, frame convention, and magnitude-to-flux recipe are in the
[data-format guide](data-format.md).

Check before prediction:

- each epoch has at least two observations;
- brightness is positive **linear relative flux**, not magnitude;
- Sun and observer vectors are asteroid-centred ecliptic J2000 vectors in AU;
- all rows use the same frame and time convention; and
- the supplied period is in hours and its source is recorded in `provenance`.

```bash
python -m json.tool my_asteroid.json >/dev/null
```

## 2. Download and check a fold bundle

Download one `k3-oof-fold-N.tar.gz` and `SHA256SUMS` from the project’s GitHub
release assets. Weights are intentionally not in the Git source tree.

```bash
sha256sum --check SHA256SUMS --ignore-missing
tar -xzf k3-oof-fold-0.tar.gz
```

For a new asteroid, select one fold before inspecting any output and record it
in your analysis. For a published benchmark asteroid, use only the bundle whose
`bundle.json` lists the object under `test_ids`; other folds trained on it.

## 3. Predict

```bash
delphi-k3-predict \
  --bundle k3-oof-fold-0 \
  --input my_asteroid.json \
  --output my_asteroid_prediction.json
```

Add `--device cuda` to use a supported NVIDIA GPU. The output file is created
only once; choose a new filename for a rerun.

`axes` contains three unit vectors in ecliptic J2000 Cartesian coordinates.
Each is an **axis**, so `(x, y, z)` and `(-x, -y, -z)` describe the same output.
The `score` values rank compatibility only within the output; they are not
probabilities or calibrated confidence. `risk_deg` is deliberately `null`.

## 4. Hand candidates to physical inversion

Convert each axis to both signs, giving six pole starts. Let the physical
inversion objective and fit diagnostics choose among them. Do not select an
axis by closeness to a reference solution in deployment—that is an
evaluation-only oracle metric.

The published containment radii apply only to archived DAMIT folds. They do not
provide transfer coverage, individual uncertainty, or a guaranteed search-space
reduction for new data.

## Python API

```python
from pathlib import Path
from lc_pipeline.k3.bundle import K3EnsemblePredictor
from lc_pipeline.k3.predict import read_observations

object_id, period, epochs = read_observations(Path("my_asteroid.json"))
model = K3EnsemblePredictor("k3-oof-fold-0", device="cpu")
result = model.predict(epochs, known_period=period, object_id=object_id)
for candidate in result["axes"]:
    print(candidate["axis_xyz"], candidate["score"])
```

For scientific methods and interpretation, cite Jo, Ishiguro and Lee, in prep.
