# DeLPHI

DeLPHI K3 proposes three asteroid spin-pole **axes** from relative lightcurves,
observing geometry, and an externally supplied rotation period. It scores
6,144 trial axes, averages five evaluated models, extracts three peaks, and
refines their directions. Each axis represents both signs; it is not a unique
directed pole. V1 is retained as the benchmark comparator.

For scientific methods and interpretation, cite **Jo, Ishiguro and Lee, in prep.**

## Install and check

Python 3.11 or 3.12 is required. CPU inference is supported. CUDA is optional;
install the PyTorch build appropriate to your driver before installing this package.

```bash
git clone https://github.com/HBJ1004/DeLPHI_LC.git
cd DeLPHI_LC
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,plot]'
delphi-k3 validate-protocol
delphi-k3 renderer-smoke --cases 8
python -m pytest -q tests repro/tests
```

The `constraints/` files record supported dependency constraints. Exact historical
package versions and hardware belong to the benchmark archive, not a promise of
bitwise reproducibility on different hardware.

## Predict

Obtain an evaluated `k3-oof-fold-N.tar.gz` bundle and its `SHA256SUMS` from the
project's release assets when published. Model weights are not stored in the Git
source tree. Check the download before extraction:

```bash
sha256sum --check SHA256SUMS --ignore-missing
tar -xzf k3-oof-fold-0.tar.gz
delphi-k3-predict --bundle k3-oof-fold-0 --input observations.json --output prediction.json
```

See [input and output format](docs/usage.md). Use `--device cuda` for GPU inference.
For one of the 170 benchmark asteroids, use **only its held-out fold** listed in
the bundled `object_roles.test_ids`; other folds are rejected. Do not average
all 25 models for a benchmark asteroid. For a new object, select a fold before
looking at outcomes; transfer accuracy and calibration have not been established.

## Benchmarks

Frozen DAMIT cohort: 170 asteroids; five object-disjoint outer folds; five training
seeds. Angular errors use the closest of three axes to any qualifying reference
solution, with antipodal equivalence. This oracle metric needs reference labels
and does not select a pole at deployment.

| Measure | K3 | V1 comparator |
|---|---:|---:|
| Mean oracle@3 error | 15.77° | 28.44° |
| Median oracle@3 error | 12.09° | 27.55° |
| Objects within 20° | 74.1% | 22.9% |

K3 uses a five-model **score-map ensemble**; V1 numbers summarize each object's
five seed errors. This is a comparison of complete pipelines, not a controlled
test isolating architecture or ensembling.

In the matched, fixed-period convex-inversion benchmark (six starts per arm,
50 iterations per start), baseline/guided wall time was **0.947**: guided execution
was approximately **5.6% slower**, including neural inference. No inversion
speedup or order-of-magnitude acceleration is demonstrated. Archived model
scoring/refinement took a median 0.509 s/object; this excludes data loading,
tokenization, and model loading and is not end-to-end deployment latency.

The 90% fold containment radii span 31.31–61.52° with 94.1% pooled empirical
coverage. All 95% radii are 90°, covering the entire axial domain and providing
no useful search-space reduction. These are not individual risk estimates.

See [reproduction instructions](docs/reproduction.md) for results, controls,
figures, training entry points, and the required external inputs.

## Source layout

- `lc_pipeline/k3/`: tokenizer, scorer, training, ensemble inference and evaluation.
- `lc_pipeline/v2/`: shared data/geometry utilities and V1 benchmark implementation;
  the directory name does not designate a second released model.
- `lc_pipeline/publication/`: benchmark contracts and result aggregation.
- `repro/`: machine-readable frozen specifications, catalog/splits and reproduction tools.
- `tests/`: numerical, integrity, and interface regression tests.

The source license does not grant rights to third-party photometry or inversion
software. Obtain those from their providers under their respective terms.
