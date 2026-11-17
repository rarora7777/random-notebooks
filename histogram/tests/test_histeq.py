"""Histogram equalization tests.

Run from the repository root:
    python -m unittest discover -s histogram/tests -v
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

import unittest

import cv2 as cv
import numpy as np

from histogram.histeq import equalize_histogram


class HistogramEqualizationTests(unittest.TestCase):
    def assert_histogram_matches_opencv(self, im_in):
        expected = cv.equalizeHist(im_in)  # pylint: disable=no-member
        im_out = equalize_histogram(im_in)

        self.assertEqual(im_out.dtype, im_in.dtype)
        self.assertEqual(im_out.shape, expected.shape)

        np.testing.assert_allclose(im_out, expected, rtol=0, atol=1e-12)

    def test_empty_image(self):
        with self.assertRaises(ValueError):
            _ = equalize_histogram(np.zeros((0, 0)))

    def test_wrong_shape(self):
        with self.assertRaises(TypeError):
            _ = equalize_histogram(np.zeros(10))

    def test_not_ndarray(self):
        with self.assertRaises(TypeError):
            _ = equalize_histogram([[0] * 10] * 10)  # pyright: ignore[reportArgumentType]

    def test_invalid_range_float(self):
        with self.assertRaises(ValueError):
            _ = equalize_histogram(-np.ones((5, 5)))

    def test_invalid_int(self):
        with self.assertRaises(ValueError):
            _ = equalize_histogram(256 * np.ones((5, 5), dtype=int))

    def test_random_int(self):
        self.assert_histogram_matches_opencv(
            np.random.randint(0, 255, (5, 5), dtype=np.uint8)
        )

    def test_constant_int(self):
        self.assert_histogram_matches_opencv(100 * np.ones((5, 5), dtype=np.uint8))

    def test_two_values_int(self):
        self.assert_histogram_matches_opencv(
            np.vstack(
                (
                    10 * np.ones((5, 5), dtype=np.uint8),
                    100 * np.ones((5, 5), dtype=np.uint8),
                )
            )
        )

    def test_uniform_unchanged_int(self):
        im_in = np.arange(0, 256, 255)
        im_in = np.vstack((im_in, im_in))
        im_out = equalize_histogram(im_in)
        np.testing.assert_allclose(im_in, im_out, rtol=0, atol=1e-12)

    def test_uniform_unchanged_float(self):
        im_in = np.arange(0.0, 1.0, 255)
        im_in = np.vstack((im_in, im_in))
        im_out = equalize_histogram(im_in)
        np.testing.assert_allclose(im_in, im_out, rtol=0, atol=1e-12)
