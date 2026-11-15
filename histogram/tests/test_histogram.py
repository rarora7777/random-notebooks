"""Tests for the simplified histogram, using NumPy's binning convention.

Run from histogram/ with: python -m unittest discover -s tests -v
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

import unittest

import numpy as np

from histogram.histogram import histogram


class HistogramTests(unittest.TestCase):
    def assert_histogram_matches_numpy(self, data, low, high, num_bins):
        resolved_low = min(data) if low is None else low
        resolved_high = max(data) if high is None else high
        expected_counts, expected_edges = np.histogram(
            data, bins=num_bins, range=(resolved_low, resolved_high)
        )

        counts, edges = histogram(data, low, high, num_bins)

        self.assertIsInstance(counts, list)
        self.assertIsInstance(edges, list)
        self.assertEqual(counts, expected_counts.tolist())
        np.testing.assert_allclose(edges, expected_edges, rtol=0, atol=1e-12)

    def test_counts_unsorted_data_in_equal_width_bins(self):
        self.assert_histogram_matches_numpy([3.5, 0.5, 2.5, 1.5, 0.25], 0, 4, 4)

    def test_interior_edges_belong_to_bin_on_right(self):
        self.assert_histogram_matches_numpy([0, 1, 1, 2, 3], 0, 4, 4)

    def test_ignores_values_outside_explicit_range(self):
        self.assert_histogram_matches_numpy([-10, -1, 0.5, 2.5, 5, 10], 0, 4, 4)

    def test_negative_and_fractional_values(self):
        self.assert_histogram_matches_numpy([-1.75, -1, -0.25, 0, 0.75], -2, 1, 6)

    def test_single_bin(self):
        self.assert_histogram_matches_numpy([0, 0.5, 1, 1.5], 0, 2, 1)

    def test_empty_bins_are_zero(self):
        self.assert_histogram_matches_numpy([0.25, 3.75], 0, 4, 4)

    def test_infers_lower_bound(self):
        self.assert_histogram_matches_numpy([2.5, 0, 1.5, 0.5], None, 4, 4)

    def test_infers_upper_bound(self):
        self.assert_histogram_matches_numpy([3, 0.5, 1.5, 2.5], 0, None, 3)

    def test_infers_both_bounds(self):
        self.assert_histogram_matches_numpy([3, 0, 1, 2], None, None, 3)

    def test_upper_endpoint_is_included_in_last_bin(self):
        self.assert_histogram_matches_numpy([0, 1, 2, 3, 4], 0, 4, 4)

    def test_repeated_upper_endpoint_is_counted_in_full(self):
        self.assert_histogram_matches_numpy([0.5, 4, 4, 4], 0, 4, 4)

    def test_empty_input_with_explicit_bounds(self):
        self.assert_histogram_matches_numpy([], 0, 4, 4)

    def test_all_values_below_range(self):
        self.assert_histogram_matches_numpy([-3, -2, -1], 0, 4, 4)

    def test_all_values_above_range(self):
        self.assert_histogram_matches_numpy([5, 6, 7], 0, 4, 4)

    def test_does_not_mutate_input(self):
        data = [3.5, 0.5, 2.5, 1.5]
        original = data.copy()

        histogram(data, 0, 4, 4)

        self.assertEqual(data, original)

    def test_rejects_nonfinite_data_regardless_of_position(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for data in ([value, 0, 1], [0, 1, value]):
                for bounds in ({}, {"low": 0, "high": 1}):
                    with self.subTest(data=data, bounds=bounds):
                        with self.assertRaises(ValueError):
                            histogram(data, **bounds)

    def test_rejects_nonfinite_bounds(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for low, high in ((value, 1), (0, value)):
                with self.subTest(low=low, high=high):
                    with self.assertRaises(ValueError):
                        histogram([0.5], low, high)


if __name__ == "__main__":
    unittest.main()
