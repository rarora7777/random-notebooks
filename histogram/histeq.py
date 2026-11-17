"""Histogram equalization for images"""

import argparse

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Equalizes the histogram of a grayscale image. Histogram is computed over 256 bins.

    Args:
        image (np.ndarray):
            integer image with values in {0, ..., 255} or
            float image with values in [0, 1]

    Raises:
        TypeError: If image is not a 2D numpy array
        ValueError: If image is empty (image.shape = (0,0))
        TypeError: If image dtype is not integer or floating
        ValueError: If image element values exceed the expected range
            (0-1 for float or 0-255 for integer)

    Returns:
        np.ndarray: Histogram-equalized image with the same size and data type as the input image.
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 2:
        raise TypeError("Input image must be a 2D numpy array")
    if image.size == 0:
        raise ValueError("Input image must not be empty")

    is_integer = np.issubdtype(image.dtype, np.integer)
    is_float = np.issubdtype(image.dtype, np.floating)

    if not (is_integer or is_float):
        raise TypeError("Input image must have an integer or floating dtype")

    max_value = 255 if is_integer else 1.0
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > max_value:
        raise ValueError(f"Pixel values must be finite and in [0, {max_value}]")

    if is_integer:
        indices = image.astype(np.intp)
    else:
        indices = np.minimum(np.floor(image.astype(np.float64) * 256), 255).astype(
            np.intp
        )

    hist = np.bincount(indices.ravel(), minlength=256)
    cmf = np.cumsum(hist)
    cmf_min = cmf[np.flatnonzero(hist)[0]]

    # A constant image's histogram can't be stretched out any further
    if cmf_min == image.size:
        return image.copy()

    # Map occupied bins into [0, 1].
    lut = np.clip(
        (cmf.astype(np.float64) - cmf_min) / (image.size - cmf_min),
        0,
        1,
    )

    if is_integer:
        lut = np.rint(lut * 255)

    return lut[indices].astype(image.dtype)


def main(impath: str) -> None:
    """Main CLI entry point

    Args:
        impath (str): Absolute or relative (to pwd) path to an image file
    """
    image = cv.imread(impath, cv.IMREAD_GRAYSCALE)  # pylint: disable=no-member
    assert image is not None, f"Unable to open image at {impath}"

    print(image.dtype)
    im_eq = equalize_histogram(image)

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(image, vmin=0, vmax=255, cmap="gray")
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(image.flatten(), 256, range=(0, 256))

    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(im_eq, vmin=0, vmax=255, cmap="gray")
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(im_eq.flatten(), 256, range=(0, 256))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Histogram equalizer")
    parser.add_argument(
        "--impath",
        "-i",
        type=str,
        required=True,
        help="Absolute or relative path to an image",
    )
    args = parser.parse_args()
    main(impath=args.impath)
