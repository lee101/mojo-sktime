# mojo-sktime

`mojo-sktime` is a focused, standalone Mojo implementation of a few
compute-bound operations from [sktime](https://www.sktime.net/), exposed as the
Python package `mojosktime`. It is not a drop-in replacement for upstream
sktime.

## Covered subset

- `transformations.series.PAA`: univariate NumPy-series Piecewise Aggregate
  Approximation, including non-divisible frame counts and trailing short frames.
- `transformations.series.SAX`: univariate NumPy-series z-normalization, PAA,
  and Gaussian-breakpoint symbolization.
- `classification.distance_based.KNeighborsTimeSeriesClassifier`: equal-length
  NumPy panels of shape `(instances, channels, timepoints)` (or univariate 2D
  input), `euclidean` and `dtw` distances, `brute`/`brute_incr` search, and
  `uniform`/`distance` voting.

The port deliberately does not cover sktime's pandas nested/multi-index data,
unequal-length or missing-value panels, callable and non-covered distances,
`ball_tree`, or the rest of sktime's broad forecasting, annotation, clustering,
and transformation API. Unsupported classifier options fail explicitly instead
of silently selecting a different algorithm.

## Install and run

```bash
pixi install
pixi run build
```

The Pixi environment adds `python/` to `PYTHONPATH`. This example runs as-is:

```bash
pixi run python - <<'PY'
import numpy as np
from mojosktime import KNeighborsTimeSeriesClassifier, PAA, SAX

series = np.arange(10.0)
print(PAA(frames=3).fit_transform(series))
print(SAX(word_size=3, alphabet_size=5).fit_transform(series))

X = np.array([[[0., 0., 1., 1.]], [[4., 4., 5., 5.]]])
y = np.array(["low", "high"])
model = KNeighborsTimeSeriesClassifier(distance="dtw").fit(X, y)
print(model.predict(np.array([[[0., 1., 1., 1.]]])))
PY
```

Run checks and the serialized benchmark with:

```bash
pixi run test
pixi run bench
```

## How it works

All kernels live in one Mojo compilation unit, `src/kernels.mojo`, to avoid
repeated compiler startup cost. NumPy arrays are converted to contiguous
`float64` (or SAX `int64` output) arrays and cross the C ABI as integer memory
addresses. Mojo rebuilds non-parametric pointers with
`AnyOrigin[mut=True]`; Python owns every array and no kernel allocates or
retains memory. Panels are row-major `(instance, channel, timepoint)` buffers.
The DTW kernel uses two dynamic-programming rows supplied as Python-owned
scratch memory.

## Benchmark

Measured with `pixi run bench` on `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`,
Python 3.13.14, with the best of two runs. These are real local measurements,
not projections.

| Case | mojo-sktime | sktime | Relative result |
|---|---:|---:|---:|
| PAA frames=128 (100k points) | 0.3 ms | 141.5 ms | 522.73x faster |
| SAX word=128 alphabet=8 (100k points) | 0.6 ms | 148.6 ms | 257.44x faster |
| KNN Euclidean predict (80x16, 48) | 0.5 ms | 52.9 ms | 105.35x faster |
| KNN DTW predict (80x16, 48) | 18.0 ms | 8,746.5 ms | 485.46x faster |

Benchmark scale and hardware matter. In particular, this port does not use
multithreaded BLAS, and different workloads may favor upstream sktime. The
covered kernels are memory-bound reductions or low-arithmetic-intensity dynamic
programming, so no GPU path is included: its transfer and launch costs lose to
the CPU implementation on these workloads.
