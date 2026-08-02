"""Numerical and behavioural parity checks against the installed sktime."""

import numpy as np
import pytest

from mojosktime import KNeighborsTimeSeriesClassifier, PAA, SAX
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier as UpstreamKNN
from sktime.transformations.series.paa import PAA as UpstreamPAA
from sktime.transformations.series.sax import SAX as UpstreamSAX


@pytest.mark.parametrize("length", [8, 10, 13, 29])
@pytest.mark.parametrize("params", [{"frames": 1}, {"frames": 3}, {"frame_size": 4}])
def test_paa_matches_sktime(length, params):
    x = np.random.default_rng(length).normal(size=length)
    if params.get("frames", 0) > length:
        pytest.skip("invalid PAA parameterization")
    expected = UpstreamPAA(**params).fit_transform(x)
    actual = PAA(**params).fit_transform(x)
    assert np.allclose(actual, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("params", [
    {"word_size": 3, "alphabet_size": 5},
    {"word_size": 7, "alphabet_size": 8},
    {"frame_size": 4, "alphabet_size": 6},
])
def test_sax_matches_sktime(params):
    x = np.random.default_rng(44).normal(size=29)
    expected = UpstreamSAX(**params).fit_transform(x)
    actual = SAX(**params).fit_transform(x)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("x", [np.ones(8), np.array([np.nan, 1.0, 2.0, 3.0])])
def test_sax_matches_sktime_for_degenerate_series(x):
    expected = UpstreamSAX(word_size=3, alphabet_size=5).fit_transform(x)
    actual = SAX(word_size=3, alphabet_size=5).fit_transform(x)
    assert np.array_equal(actual, expected)


def test_paa_and_sax_validation_matches_upstream():
    x = np.arange(4.0)
    for cls, params in ((PAA, {"frames": 5}), (SAX, {"word_size": 5})):
        with pytest.raises(ValueError, match="Series length"):
            cls(**params).fit_transform(x)
    with pytest.raises(ValueError, match="alphabet_size"):
        SAX(alphabet_size=1)


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(24, 2, 16))
    y = np.array(["left"] * 12 + ["right"] * 12)
    X[12:] += 0.8
    query = rng.normal(size=(7, 2, 16))
    query[4:] += 0.8
    return X, y, query


@pytest.mark.parametrize("distance", ["euclidean", "dtw"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_matches_sktime_pairwise_distances_predictions_and_probabilities(panels, distance, weights):
    X, y, query = panels
    upstream = UpstreamKNN(n_neighbors=3, distance=distance, weights=weights).fit(X, y)
    ours = KNeighborsTimeSeriesClassifier(n_neighbors=3, distance=distance, weights=weights).fit(X, y)
    assert np.allclose(ours._distances(query), upstream._dist_adapt._distance(query, X))
    assert np.allclose(ours.predict_proba(query), upstream.predict_proba(query))
    assert np.array_equal(ours.predict(query), upstream.predict(query))
    assert ours.score(query, upstream.predict(query)) == 1.0


def test_knn_accepts_univariate_2d_panels_and_rejects_out_of_scope_options(panels):
    X, y, query = panels
    ours = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="euclidean").fit(X[:, 0], y)
    assert ours.predict(query[:, 0]).shape == (len(query),)
    with pytest.raises(NotImplementedError):
        KNeighborsTimeSeriesClassifier(distance="ddtw").fit(X, y)
    with pytest.raises(NotImplementedError):
        KNeighborsTimeSeriesClassifier(algorithm="ball_tree").fit(X, y)
    with pytest.raises(NotImplementedError):
        KNeighborsTimeSeriesClassifier(n_jobs=2).fit(X, y)


def test_rejects_lossy_dtypes_and_empty_ffi_inputs():
    with pytest.raises(TypeError):
        PAA(frames=1).fit_transform(np.array([1 + 2j]))
    with pytest.raises(ValueError):
        PAA(frames=1).fit_transform(np.array([2**53 + 1], dtype=np.int64))
    with pytest.raises(TypeError):
        PAA(frames=1).fit_transform(np.array([1], dtype=np.longdouble))
    with pytest.raises(ValueError):
        KNeighborsTimeSeriesClassifier().fit(np.empty((0, 1, 2)), np.array([]))
