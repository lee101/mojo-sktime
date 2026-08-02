"""ctypes bridge for the standalone Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-sktime.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mts_paa_frames": ([I, I, I, I], None),
    "mts_paa_frame_size": ([I, I, I, I], None),
    "mts_sax_frames": ([I, I, I, I, I, I], None),
    "mts_sax_frame_size": ([I, I, I, I, I, I], None),
    "mts_pairwise_euclidean": ([I] * 7, None),
    "mts_pairwise_dtw": ([I] * 8, None),
}


def build() -> str:
    if not os.path.exists(LIB):
        subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")], check=True)
    return LIB


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes, fn.restype = argtypes, restype
    return _lib


def f64(a) -> np.ndarray:
    """Return a C-contiguous float64 buffer without lossy coercion.

    The C ABI has no dtype metadata.  Validate before converting so a complex
    array, extended-precision float, or large integer cannot be silently read
    by the Mojo kernels as a different value.
    """
    array = np.asarray(a)
    if array.dtype.kind not in {"b", "u", "i", "f"}:
        raise TypeError(f"covered inputs must have a real numeric dtype, got {array.dtype}")
    if array.dtype.kind == "f" and array.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError(f"{array.dtype} cannot be converted to float64 without loss")
    if array.dtype.kind == "i":
        limit = 2**53
        if array.size and (array.min() < -limit or array.max() > limit):
            raise ValueError("integer inputs must be exactly representable as float64")
    if array.dtype.kind == "u" and array.size and array.max() > 2**53:
        raise ValueError("integer inputs must be exactly representable as float64")
    return np.ascontiguousarray(array, dtype=np.float64)


def i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def addr(a: np.ndarray) -> int:
    if not isinstance(a, np.ndarray) or not a.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous NumPy arrays")
    if a.size == 0:
        raise ValueError("empty buffers must not cross the FFI boundary")
    return int(a.ctypes.data)
