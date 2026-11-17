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
        rng = np.random.default_rng(seed=42)
        self.assert_histogram_matches_opencv(
            rng.integers(0, 256, (5, 5), dtype=np.uint8)
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
        im_in = np.arange(256, dtype=np.uint8).reshape(16, 16)
        im_out = equalize_histogram(im_in)
        np.testing.assert_array_equal(im_in, im_out)

    def test_float_equalization(self):
        im_in = np.array([[0.2, 0.2], [0.6, 1.0]])
        expected = np.array([[0.0, 0.0], [0.5, 1.0]])

        im_out = equalize_histogram(im_in)

        self.assertEqual(im_out.dtype, im_in.dtype)
        np.testing.assert_allclose(im_out, expected)

    def test_rejects_nonfinite_values(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    equalize_histogram(np.array([[0.0, value]]))

    def test_rejects_float_above_one(self):
        with self.assertRaises(ValueError):
            equalize_histogram(np.array([[0.0, 1.1]]))

    def test_rejects_negative_integer(self):
        with self.assertRaises(ValueError):
            equalize_histogram(np.array([[-1, 100]]))

    def test_rejects_unsupported_dtypes(self):
        for dtype in (np.bool_, np.complex128, np.str_, object):
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    equalize_histogram(np.array([[0, 1]], dtype=dtype))

    def test_preserves_float32_dtype(self):
        im_in = np.array([[0.2, 0.2], [0.6, 1.0]], dtype=np.float32)

        im_out = equalize_histogram(im_in)

        self.assertEqual(im_out.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(im_out, [[0.0, 0.0], [0.5, 1.0]])

    def test_does_not_modify_input(self):
        for dtype in (np.uint8, np.float32):
            for constant in (False, True):
                with self.subTest(dtype=dtype, constant=constant):
                    values = [[50, 50], [50, 50]] if constant else [[50, 50], [150, 250]]
                    im_in = np.array(values, dtype=dtype)
                    if dtype == np.float32:
                        im_in /= 255
                    original = im_in.copy()

                    equalize_histogram(im_in)

                    np.testing.assert_array_equal(im_in, original)
