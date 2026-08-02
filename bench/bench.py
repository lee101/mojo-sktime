"""Run through `pixi run bench`; the task takes a machine-wide flock."""

from __future__ import annotations

import math
import platform
import time

import numpy as np
from mojosktime import KNeighborsTimeSeriesClassifier, PAA, SAX
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier as UpstreamKNN
from sktime.transformations.series.paa import PAA as UpstreamPAA
from sktime.transformations.series.sax import SAX as UpstreamSAX


def best(fn, repeat=2):
    elapsed = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        elapsed = min(elapsed, time.perf_counter() - start)
    return elapsed


def main():
    rng = np.random.default_rng(0)
    series = np.ascontiguousarray(rng.normal(size=100_000))
    train = np.ascontiguousarray(rng.normal(size=(80, 1, 48)))
    labels = np.arange(80) % 4
    query = np.ascontiguousarray(rng.normal(size=(16, 1, 48)))
    cases = [
        ("PAA frames=128 (100k points)", lambda: PAA(frames=128).fit_transform(series), lambda: UpstreamPAA(frames=128).fit_transform(series)),
        ("SAX word=128 alphabet=8 (100k points)", lambda: SAX(word_size=128, alphabet_size=8).fit_transform(series), lambda: UpstreamSAX(word_size=128, alphabet_size=8).fit_transform(series)),
        ("KNN euclidean predict (80x16, 48)", lambda: KNeighborsTimeSeriesClassifier(n_neighbors=5, distance="euclidean").fit(train, labels).predict(query), lambda: UpstreamKNN(n_neighbors=5, distance="euclidean").fit(train, labels).predict(query)),
        ("KNN DTW predict (80x16, 48)", lambda: KNeighborsTimeSeriesClassifier(n_neighbors=5, distance="dtw").fit(train, labels).predict(query), lambda: UpstreamKNN(n_neighbors=5, distance="dtw").fit(train, labels).predict(query)),
    ]
    print(f"Machine: {platform.platform()} | Python: {platform.python_version()}", flush=True)
    print(f"{'case':<42} {'mojo-sktime':>13} {'sktime':>13} {'ratio':>9}", flush=True)
    print("-" * 83, flush=True)
    for name, ours, upstream in cases:
        ours()
        mojo_time, upstream_time = best(ours), best(upstream)
        verdict = "faster" if mojo_time < upstream_time else "slower"
        print(f"{name:<42} {mojo_time * 1e3:>11.1f}ms {upstream_time * 1e3:>11.1f}ms {upstream_time / mojo_time:>7.2f}x {verdict}", flush=True)


if __name__ == "__main__":
    main()
