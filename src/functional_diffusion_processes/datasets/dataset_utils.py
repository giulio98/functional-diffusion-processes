from typing import Any

import tensorflow as tf


def resize_small(image: Any, resolution: int) -> Any:
    """Resize an image to a specified resolution while preserving its aspect ratio.

    This function scales the image dimensions such that the smaller dimension equals the given resolution,
    and the other dimension is scaled proportionately to maintain the original aspect ratio of the image.
    Antialiasing is applied during the resizing process to reduce aliasing artifacts.

    Args:
        image (Any): A tensor of shape (height, width, channels) representing the image to be resized.
        resolution (int): The target size for the smaller dimension of the image.

    Returns:
        Any: A tensor of shape (new_height, new_width, channels) representing the resized image, where
             new_height and new_width are determined based on the original aspect ratio of the image.
    """
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio)
    w = tf.round(w * ratio)
    image = tf.image.resize(images=image, size=[h, w], antialias=True)
    return image


def central_crop(image: Any, size: int) -> Any:
    """Crop the central region of an image to the specified size.

    This function extracts a square region from the center of the input image with side length equal to
    the specified size. If the specified size is larger than either dimension of the image, a ValueError
    is raised.

    Args:
        image (Any): A tensor of shape (height, width, channels) representing the image to be cropped.
        size (int): The side length of the square region to be cropped from the center of the image.

    Returns:
        Any: A tensor of shape (size, size, channels) representing the central cropped region of the image.

    Raises:
        ValueError: If the specified size is greater than either dimension of the image, indicating that
                    the requested crop is not possible.
    """
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    if top < 0 or left < 0:
        raise ValueError("Size is greater than the dimensions of the image.")
    cropped_image = tf.image.crop_to_bounding_box(
        image=image,
        offset_height=top,
        offset_width=left,
        target_height=size,
        target_width=size,
    )
    return cropped_image
