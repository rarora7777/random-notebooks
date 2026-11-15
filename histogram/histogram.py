"""Simplified version of numpy's histogram function"""

import math
from bisect import bisect_right


def histogram(
    data: list[float],
    low: float | None = None,
    high: float | None = None,
    num_bins: int = 10,
) -> tuple[list[int], list[float]]:
    """
    Computes a histogram over the input data. low, high determine the range of values counted
    by the histogram. Values outside of [low, high] are excluded.
    For values at a bin edge, numpy behaviour is emulated: if a bin is [l, h], then included 
    values satisfy l <= x < h, unless if this is the rightmost bin, then l <= x <= h. That is, 
    the `high` value is included in the rightmost bin.

    Args:
        data (list[float]): input list of floats
        low (float | None): lower bound of range included in the histogram. If None (default),
          it's set to min(data)
        high (float | None): upper bound of range included in the histogram. If None (default),
          it's set to max(data)
        num_bins (int): Number of histogram bins

    Returns:
        tuple[list[int], list[float]]: histogram, bin edges
    """
    if len(data) == 0:
        if low is None:
            raise ValueError("low can't be None if the input array is empty")
        if high is None:
            raise ValueError("high can't be None if the input array is empty")

    if any(not math.isfinite(x) for x in data):
        raise ValueError("elements of data must be finite: NaN or infinity are unsupported")

    if low is None:
        low = min(data)
    if high is None:
        high = max(data)

    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("low and high must be finite")
    if low >= high:
        raise ValueError(f"high ({high}) must be strictly greater than low ({low})")

    if not isinstance(num_bins, int) or num_bins < 1:
        raise ValueError(f"num_bins must be a positive integer: given {num_bins}")
    
    step = (high - low) / num_bins
    bin_edges = [low + i * step for i in range(num_bins + 1)]
    bin_edges[-1] = high

    hist = [0] * (len(bin_edges) - 1)

    if len(data) == 0:
        return hist, bin_edges

    for value in data:
        if low <= value <= high:
            idx = bisect_right(bin_edges, value) - 1
            if value == high:
                idx = num_bins - 1
            hist[idx] += 1

    return hist, bin_edges
