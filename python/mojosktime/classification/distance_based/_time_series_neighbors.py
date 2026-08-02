"""Brute-force nearest-neighbour time-series classification."""

from __future__ import annotations

import numpy as np

from ..._lib import addr, f64, lib


class KNeighborsTimeSeriesClassifier:
    """KNN time-series classifier for equal-length NumPy 3D panels.

    The covered `distance` values are ``"euclidean"`` and ``"dtw"``. Both
    pairwise distance kernels run in Mojo; voting and estimator bookkeeping stay
    in Python to preserve sktime's labels and public attributes.
    """

    def __init__(
        self,
        n_neighbors=1,
        weights="uniform",
        algorithm="brute",
        distance="dtw",
        distance_params=None,
        distance_mtype=None,
        pass_train_distances=False,
        leaf_size=30,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.distance = distance
        self.distance_params = distance_params
        self.distance_mtype = distance_mtype
        self.pass_train_distances = pass_train_distances
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

    @staticmethod
    def _panel(X):
        x = f64(X)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3:
            raise ValueError("covered input is a NumPy panel with shape (n_instances, n_channels, n_timepoints)")
        if x.shape[2] == 0:
            raise ValueError("time series must contain at least one timepoint")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("panels must contain at least one instance and channel")
        return x

    def _validate(self):
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if self.algorithm not in {"brute", "brute_incr"}:
            raise NotImplementedError("covered classifier algorithms are 'brute' and 'brute_incr'")
        if self.distance not in {"euclidean", "dtw"}:
            raise NotImplementedError("covered distances are 'euclidean' and 'dtw'")
        if self.weights not in {"uniform", "distance"}:
            raise NotImplementedError("covered weights are 'uniform' and 'distance'")
        if self.distance_params:
            raise NotImplementedError("distance_params are not covered")
        if self.distance_mtype is not None or self.pass_train_distances:
            raise NotImplementedError("distance_mtype and pass_train_distances are not covered")
        if self.n_jobs is not None:
            raise NotImplementedError("n_jobs is not covered")

    def fit(self, X, y):
        self._validate()
        self._X = self._panel(X)
        self._y = np.asarray(y)
        if self._y.ndim != 1:
            raise ValueError("y must be a one-dimensional label array")
        if self._X.shape[0] != len(self._y):
            raise ValueError("X and y have inconsistent numbers of instances")
        if self.n_neighbors > len(self._y):
            raise ValueError("Expected n_neighbors <= n_samples_fit")
        self.classes_ = np.unique(self._y)
        self.n_features_in_ = self._X.shape[1] * self._X.shape[2]
        return self

    def _distances(self, X):
        if not hasattr(self, "_X"):
            raise ValueError("This KNeighborsTimeSeriesClassifier instance is not fitted yet")
        x = self._panel(X)
        if x.shape[1:] != self._X.shape[1:]:
            raise ValueError("X has incompatible channel or timepoint dimensions")
        result = np.empty((len(x), len(self._X)))
        train_count, channels, timepoints = self._X.shape
        if self.distance == "euclidean":
            lib().mts_pairwise_euclidean(addr(self._X), addr(x), addr(result), train_count, len(x), channels, timepoints)
        else:
            scratch = np.empty(2 * (timepoints + 1))
            lib().mts_pairwise_dtw(addr(self._X), addr(x), addr(result), addr(scratch), train_count, len(x), channels, timepoints)
        return result

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is None:
            X = self._X
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if not isinstance(k, int) or not 1 <= k <= len(self._X):
            raise ValueError("n_neighbors must be a positive integer no greater than n_samples_fit")
        distances = self._distances(X)
        indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
        chosen = np.take_along_axis(distances, indices, axis=1)
        return (chosen, indices) if return_distance else indices

    def predict_proba(self, X):
        distances, indices = self.kneighbors(X)
        labels = self._y[indices]
        probabilities = np.zeros((len(indices), len(self.classes_)))
        for row in range(len(indices)):
            if self.weights == "uniform":
                weights = np.ones(self.n_neighbors)
            elif np.any(distances[row] == 0):
                weights = (distances[row] == 0).astype(float)
            else:
                weights = 1.0 / distances[row]
            for col, klass in enumerate(self.classes_):
                probabilities[row, col] = weights[labels[row] == klass].sum()
            probabilities[row] /= weights.sum()
        return probabilities

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
