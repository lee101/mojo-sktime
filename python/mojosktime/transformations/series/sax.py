"""Symbolic Aggregate approXimation."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from ..._lib import addr, f64, lib


class SAX:
    """SAX with sktime's constructor signature for univariate NumPy series."""

    def __init__(self, word_size=8, alphabet_size=5, frame_size=0):
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.frame_size = frame_size
        self._check_params()

    def _check_params(self):
        for name in ("word_size", "alphabet_size", "frame_size"):
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be of type int. Found {type(getattr(self, name)).__name__}.")
        if self.word_size < 1 and not self.frame_size:
            raise ValueError("word_size must be at least 1.")
        if self.alphabet_size < 2:
            raise ValueError("alphabet_size must be at least 2.")
        if self.frame_size < 0:
            raise ValueError("frame_size must be at least 0.")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = f64(X)
        if x.ndim != 1:
            raise ValueError("covered SAX input is one univariate numpy series")
        n = len(x)
        if self.frame_size and self.frame_size > n:
            raise ValueError("Series length cannot be shorter than the desired frame size.")
        if not self.frame_size and self.word_size > n:
            raise ValueError("Series length cannot be shorter than the desired number of frames.")
        breakpoints = np.array([NormalDist().inv_cdf(i / self.alphabet_size) for i in range(1, self.alphabet_size)])
        frames = (n + self.frame_size - 1) // self.frame_size if self.frame_size else self.word_size
        result = np.empty(frames, dtype=np.int64)
        # scipy.stats.zscore, used by upstream sktime, produces NaNs for a
        # constant or non-finite series; numpy.digitize assigns those to the
        # final alphabet symbol. Avoid dividing by zero/NaN in the kernel.
        if not np.isfinite(x).all() or np.ptp(x) == 0:
            result.fill(len(breakpoints))
            return result
        if self.frame_size:
            lib().mts_sax_frame_size(addr(x), addr(result), addr(breakpoints), n, self.frame_size, len(breakpoints))
        else:
            lib().mts_sax_frames(addr(x), addr(result), addr(breakpoints), n, self.word_size, len(breakpoints))
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
