# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys

# Allow helpers package to be found when this module is imported standalone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import load_image, get_data_path

SPHINX_IMAGE = "2560px-Great_Sphinx_of_Giza_-_20080716a.jpg"

def read_image():
    """Load and preprocess the Sphinx image: scale by 0.5 and convert to greyscale."""
    image_path = get_data_path(SPHINX_IMAGE)
    return load_image(image_path, scale_factor=2, as_gray=True)

# Skeleton Code will generate a 1D Gaussian Kernel for given sigma.
# You need to implement 2D convolution as efficiently as possible.
class GaussianFilt:
    def __init__(self, sigma):
        self.sigma = sigma

    def gauss_kernel(self):
        """Generate a normalised 1D Gaussian kernel of appropriate size for self.sigma.

        Returns:
            gauss_1d (ndarray): Shape (k_size, 1) — column vector kernel.
        """
        # Work out the necessary kernel size and range to generate the Gaussian over.
        k_size = int(6 * self.sigma + 1)
        offsets = np.arange(k_size) - (k_size // 2)

        gauss_1d = np.exp(-(offsets ** 2) / (2 * self.sigma ** 2))
        gauss_1d /= np.sum(gauss_1d)

        return np.expand_dims(gauss_1d, axis=1)

    def my_conv_method(self, image):
        """Perform separable 2D Gaussian convolution via two 1D matrix-multiply passes.

        Args:
            image (ndarray): 2D greyscale image (H, W), any numeric dtype.

        Returns:
            conv_img (ndarray): Blurred image of same shape, dtype uint8.
        """
        image = np.asarray(image, dtype=np.float64)
        kernel = self.gauss_kernel().ravel()
        k_size = len(kernel)
        pad = k_size // 2

        # Pad the image once so the output keeps the same height and width.
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")

        # Apply the 1D kernel horizontally, then vertically.
        horizontal = cv2.filter2D(
            padded,
            ddepth=cv2.CV_64F,
            kernel=kernel.reshape(1, -1),
            borderType=cv2.BORDER_CONSTANT,
        )
        blurred = cv2.filter2D(
            horizontal,
            ddepth=cv2.CV_64F,
            kernel=kernel.reshape(-1, 1),
            borderType=cv2.BORDER_CONSTANT,
        )

        # Keep the region corresponding to the original image.
        height, width = image.shape
        conv_img = blurred[pad:pad + height, pad:pad + width]

        # Convert to uint8, clipping values to the valid image intensity range.
        conv_img = np.clip(np.rint(conv_img), 0, 255).astype(np.uint8)
        return conv_img