"""
This file contains all the mutilation functions that will be used for the experiments.

To use the mutilation function, define it in `*.config.ini` in the [DEFAULT] section
"""

from PIL import Image
import numpy as np
from weight_uncertainty.util.load_data import normalize
from weight_uncertainty import conf


def rotate_cifar(images, angle):
    """
    Special function to rotate cifar images as they are RGB
    :param images:
    :param angle:
    :return:
    """
    images_out = np.zeros_like(images)

    for n in range(images.shape[0]):
        imarray = normalize(images[n], reverse=True)
        im = Image.fromarray(imarray.astype(np.uint8), mode="RGB")
        im = im.rotate(-1 * angle)
        images_out[n] = normalize(np.array(im))
    return images_out


def rotation(images, angle):
    """
    Rotates the image over <angle> degrees.
    :param images: numpy array
    :param angle:
    :return: numpy array
    """
    if conf.dataset == "cifar":
        return rotate_cifar(images, angle)
    images_out = np.zeros_like(images)
    # Check if there is a dimension of unit size
    try:
        unit_dim = images.shape.index(1)
    except ValueError:
        unit_dim = None

    if unit_dim:
        images = np.squeeze(images, axis=unit_dim)

    num_batch = images.shape[0]
    for n in range(num_batch):
        im = Image.fromarray(images[n])
        im = im.rotate(-1 * angle)
        if unit_dim:
            im = np.expand_dims(np.array(im), axis=unit_dim - 1)
        images_out[n] = im
    return images_out


def noise(images, sigma):
    """
    Adds Gaussian noise to a set of images.

        Args:
            images: The input images as a NumPy array.
            sigma: The standard deviation of the Gaussian noise.

        Returns:
            NumPy array: The images with added noise.
    """
    images += sigma * np.random.randn(*images.shape)  # , *conf.range)
    return images


def noise_clip(images, sigma):
    """
    Adds clipped Gaussian noise to images.

        Args:
            images: The input images as a NumPy array.
            sigma: The standard deviation of the Gaussian noise.

        Returns:
            NumPy array: The images with added noise, clipped to the specified range defined in conf.range.
    """
    images += np.clip(sigma * np.random.randn(*images.shape), *conf.range)
    return images


##########################################################################################################
# BELOW IS SUPER UGLY AND REPEATING CODE
##########################################################################################################


def warp_cifar(images, warp_value):
    """
    Warps CIFAR images using an affine transformation.

        Args:
            images: The input images as a NumPy array.
            warp_value: The amount of warping to apply.

        Returns:
            A NumPy array containing the warped images.
    """
    # TODO merge this function with rotate_cifar()
    images_out = np.zeros_like(images)

    for n in range(images.shape[0]):
        imarray = normalize(images[n], reverse=True)
        im = Image.fromarray(imarray.astype(np.uint8), mode="RGB")
        im = im.transform(
            (32, 32),
            Image.AFFINE,
            data=(1 - warp_value, warp_value, 0, warp_value, 1.0 + warp_value, 0),
        )
        images_out[n] = normalize(np.array(im))
    return images_out


def warp(images, warp_value):
    """
    Applies an affine transformation to warp images.

        This function warps a batch of images using an affine transformation
        defined by the `warp_value`. It handles different dataset configurations
        and optionally squeezes/expands dimensions for compatibility.  For CIFAR datasets,
        it calls a specialized warping function.

        Args:
            images: The input images as a NumPy array.
            warp_value: The amount of warp to apply. This value is used in the affine transformation matrix.

        Returns:
            NumPy array: The warped images with the same shape as the input `images`.
    """
    # TODO merge this function with rotation()
    if conf.dataset == "cifar":
        return warp_cifar(images, warp_value)
    images_out = np.zeros_like(images)
    # Check if there is a dimension of unit size
    try:
        unit_dim = images.shape.index(1)
    except ValueError:
        unit_dim = None

    if unit_dim:
        images = np.squeeze(images, axis=unit_dim)

    num_batch = images.shape[0]
    for n in range(num_batch):
        im = Image.fromarray(images[n])
        im = im.transform(
            (28, 28),
            Image.AFFINE,
            data=(1 - warp_value, warp_value, 0, warp_value, 1.0 + warp_value, 0),
        )
        if unit_dim:
            im = np.expand_dims(np.array(im), axis=unit_dim - 1)
        images_out[n] = im
    return images_out
