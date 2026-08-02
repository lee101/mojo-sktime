"""Piecewise Aggregate Approximation."""

from __future__ import annotations

import numpy as np

from ..._lib import addr, f64, lib


class PAA:
    """Piecewise Aggregate Approximation with sktime's constructor signature."""

    def __init__(self, frames=8, frame_size=0):
        self.frames = frames
        self.frame_size = frame_size
        self._check_params()

    def _check_params(self):
        for name in ("frames", "frame_size"):
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be of type int. Found {type(getattr(self, name)).__name__}.")
        if self.frames < 1 and not self.frame_size:
            raise ValueError("frames must be at least 1.")
        if self.frame_size < 0:
            raise ValueError("frame_size must be at least 0.")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = f64(X)
        if x.ndim != 1:
            raise ValueError("covered PAA input is one univariate numpy series")
        n = len(x)
        if self.frame_size:
            if self.frame_size > n:
                raise ValueError("Series length cannot be shorter than the desired frame size.")
            result = np.empty((n + self.frame_size - 1) // self.frame_size)
            lib().mts_paa_frame_size(addr(x), addr(result), n, self.frame_size)
        else:
            if self.frames > n:
                raise ValueError("Series length cannot be shorter than the desired number of frames.")
            result = np.empty(self.frames)
            lib().mts_paa_frames(addr(x), addr(result), n, self.frames)
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
