"""Streaming histogram tests.

Run from the repository root:
    python -m unittest discover -s histogram/tests -v
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

import unittest

import numpy as np

from histogram.streaming_hist import StreamingHistogram


class StreamingHistogramTests(unittest.TestCase):
    def assert_stream_matches_numpy(self, samples, bins, window_size):
        histogram = StreamingHistogram(window_size, bins)
        window = []
        for step, sample in enumerate(samples):
            window.append(sample)
            window = window[-window_size:]
            expected, _ = np.histogram(window, bins=bins)
            histogram.add(sample)
            with self.subTest(step=step, sample=sample, window=tuple(window)):
                np.testing.assert_array_equal(histogram.get(), expected)

    def test_starts_with_zero_counts(self):
        histogram = StreamingHistogram(3, [0, 1, 2, 4])
        np.testing.assert_array_equal(histogram.get(), [0, 0, 0])

    def test_counts_samples_before_window_fills(self):
        self.assert_stream_matches_numpy([0.25, 1.5, 1.75], [0., 1., 2., 4.], 5)

    def test_counts_samples_when_window_is_exactly_full(self):
        self.assert_stream_matches_numpy([0, 1, 2, 4], [0, 1, 2, 4], 4)

    def test_edges_belong_to_right_bin_except_final_edge(self):
        self.assert_stream_matches_numpy([0, 1, 2, 4, 4], [0, 1, 2, 4], 6)

    def test_ignores_out_of_range_samples(self):
        self.assert_stream_matches_numpy([-10, 0, 2, 4, 10], [0, 1, 2, 4], 6)

    def test_nonuniform_bins_and_negative_values(self):
        self.assert_stream_matches_numpy(
            [-3., -2.5, -1., -0.5, 0., 1.5, 5.], [-3., -1., 0., 5.], 8
        )

    def test_fractional_samples_with_integer_bin_edges(self):
        self.assert_stream_matches_numpy([-0.5, 0.5, 1.5, 2.5], [0, 1, 2], 5)

    def test_single_bin(self):
        self.assert_stream_matches_numpy([0., 0.5, 1., -1., 2.], [0., 1.], 6)

    def test_numpy_scalar_samples(self):
        self.assert_stream_matches_numpy(
            [np.int32(0), np.float32(0.5), np.float64(2)], [0., 1., 2.], 4
        )

    def test_evicts_oldest_sample(self):
        self.assert_stream_matches_numpy([0, 0, 1, 2], [0, 1, 2, 3], 3)

    def test_nonfinite_evicts_oldest_sample(self):
        self.assert_stream_matches_numpy([0, 0, 1, float('nan')], [0, 1, 2, 3], 3)

    def test_multiple_wraps(self):
        self.assert_stream_matches_numpy([0, 1, 2, 3, 2, 1, 0, 0, 3, 1], [0, 1, 2, 3], 3)

    def test_window_size_one(self):
        self.assert_stream_matches_numpy([0, 1, 2, 0, 1], [0, 1, 2], 1)

    def test_out_of_range_sample_evicts_in_range_sample(self):
        self.assert_stream_matches_numpy([0, 1, 10, -10], [0, 1, 2], 2)

    def test_evicting_out_of_range_sample_does_not_decrement_a_bin(self):
        self.assert_stream_matches_numpy([-10, 1, 0], [0, 1, 2], 2)

    def test_default_bins_can_count_a_sample(self):
        histogram = StreamingHistogram(3, None)
        self.assertGreaterEqual(len(histogram.get()), 2)
        histogram.add(0)
        self.assertEqual(histogram.get().sum(), 1)

    def test_rejects_invalid_window_sizes(self):
        for window_size in (0, -1, 1.5, "3", None):
            with self.subTest(window_size=window_size):
                with self.assertRaises(ValueError):
                    StreamingHistogram(window_size, [0, 1, 2]) # pyright: ignore[reportArgumentType]

    def test_rejects_bins_that_are_not_strictly_increasing(self):
        for bins in ([0, 0, 1], [2, 1, 0], [0, 2, 1]):
            with self.subTest(bins=bins):
                with self.assertRaises(ValueError):
                    StreamingHistogram(3, bins) # pyright: ignore[reportArgumentType]

    def test_does_not_modify_supplied_bin_array(self):
        bins = np.array([0., 1., 2.])
        original = bins.copy()
        histogram = StreamingHistogram(3, bins)
        histogram.add(0.5)
        np.testing.assert_array_equal(bins, original)


if __name__ == "__main__":
    unittest.main()
