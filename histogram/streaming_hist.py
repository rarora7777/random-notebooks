"""_summary_
"""

from typing import Any

import numpy as np
import numpy.typing as npt

type RealScalar = (
    int | float | np.integer[Any] | np.floating[Any]
)

class StreamingHistogram:
    """Compute histogram on streaming data using a rolling buffer.

    Raises:
        ValueError: If window_size is not a positive integer
        ValueError: If bins is not a strictly increasing sequence

    Returns:
        _type_: _description_
    """
    _window_size: int
    _buffer: npt.NDArray[np.float64]
    _bins: npt.NDArray[np.float64]
    _idx: int
    _is_full: bool
    _hist: npt.NDArray[np.int64]

    def __init__(self, window_size: int = 10, bins: np.ndarray | list[RealScalar] | None = None) -> None:
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(f"window_size must be a positive integer: given {window_size}")

        if bins is None:
            self._bins = np.arange(0, 256, 1, dtype=np.float64)
        else:
            self._bins = np.array(bins, dtype=np.float64)

        if len(self._bins.shape) != 1 or self._bins.size < 2:
            raise ValueError("bins must be a 1-D array of size >= 2")
        if np.any(np.diff(self._bins) <= 0) or not np.all(np.isfinite(self._bins)):
            raise ValueError("bins must be a strictly increasing sequence of finite numbers")

        self._window_size = window_size
        self._buffer = np.zeros(window_size, dtype=self._bins.dtype)
        self._idx = 0
        self._is_full = False
        self._hist = np.zeros(len(self._bins) - 1, dtype=np.int64)

    def add(self, x: RealScalar) -> None:
        """Add a new sample to the buffer. If the buffer is full, an old sample is deleted.

        Args:
            x (RealScalar): New value.
        """
        old = self._buffer[self._idx]
        self._buffer[self._idx] = x
        bin_idx = self._find_bin(self._buffer[self._idx])
        
        if bin_idx >= 0:
            self._hist[bin_idx] += 1
        
        if self._is_full:
            bin_idx = self._find_bin(old)
            if bin_idx >= 0:
                self._hist[bin_idx] -= 1

        self._idx = (self._idx + 1) % self._window_size

        if self._idx == 0:
            self._is_full = True


    def _find_bin(self, value: RealScalar) -> int:
        if not np.isfinite(value):
            return -1
        
        low = self._bins[0]
        high = self._bins[-1]
        num_bins = len(self._bins) - 1
        idx = -1
        if low <= value <= high:
            idx = int(np.searchsorted(self._bins, value, side='right')) - 1
            if value == high:
                idx = num_bins - 1

        return idx

    def get(self) -> npt.NDArray[np.int64]:
        """Get the histogram

        Returns:
            np.ndarray: histogram as a numpy array of type int
        """
        return self._hist.copy()
