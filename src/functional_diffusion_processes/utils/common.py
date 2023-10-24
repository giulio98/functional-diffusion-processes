import io
import math
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from flax.core import FrozenDict, unfreeze


def filter_mask(shape, radius):
    device_num, batch_size, rows, cols, n_channels = shape
    crow, ccol = int(rows / 2), int(cols / 2)
    center = [crow, ccol]
    x, y = jnp.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= radius * radius
    mask = jnp.ones_like(mask_area)
    mask = jnp.where(mask_area, 0, mask)
    mask = mask.reshape(1, 1, rows, cols, 1)
    mask = jnp.repeat(mask, device_num, axis=0)
    mask = jnp.repeat(mask, batch_size, axis=1)
    mask = jnp.repeat(mask, n_channels, axis=4)
    return mask


@jax.pmap
def to_grayscale(images):
    weights = np.array([0.2989, 0.5870, 0.1140])[None, None, None, :]  # Extend dimensions
    grayscale_images = np.sum(images * weights, axis=-1)
    return grayscale_images


def save_samples(round_num: int, samples: Any, file_path: str) -> None:
    """Save samples to a file.

    Args:
        round_num: The round number of the evaluation.
        samples: Tensor of samples to save.
        file_path: string of the Path to the file where the samples will be saved.
    """
    for i in range(samples.shape[0]):
        clean_path = os.path.join(file_path, f"clean/samples_{round_num}_{i}.npy")
        np.save(clean_path, samples[i])
    samples_path = os.path.join(file_path, f"samples_{round_num}.npz")
    with tf.io.gfile.GFile(samples_path, "wb") as f_out:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, samples=samples)
        f_out.write(io_buffer.getvalue())


def process_images(images: Any) -> Any:
    """Reshape images to the correct shape.

    Args:
        images: Tensor of images to reshape.

    Returns:
        A tensor of images with the correct shape.
    """
    w = np.sqrt(images.shape[2]).astype(int)
    h = np.sqrt(images.shape[2]).astype(int)
    o = images.shape[3]
    return images.reshape(-1, w, h, o)


def separate_learning_rates(params):
    """Separate the learning rates from the other parameters.

    Args:
        params: A dictionary of parameters.

    Returns:
        A tuple containing the learning rates and the other parameters.
    """
    learning_rates = {}
    other_params = {}
    for layer_name, layer_params in params["params"].items():
        if "lr" in layer_name.lower():
            learning_rates[layer_name] = layer_params
        else:
            other_params[layer_name] = layer_params

    new_lr_params = {}
    for key, value in learning_rates.items():
        layer_name, param_id = key.split("_", 1)  # Split the key into layer_name and param_name
        clean_layer_name = layer_name.replace("LR", "")
        combined_layer_name = f"{clean_layer_name}_{param_id}"

        if combined_layer_name not in new_lr_params:
            new_lr_params[combined_layer_name] = {}
        new_lr_params[combined_layer_name] = value

    return FrozenDict({"params": other_params}), FrozenDict({"params": new_lr_params})


def merge_learning_rates(params, learning_rates):
    """Merge the learning rates with the other parameters.

    Args:
        params: A dictionary of parameters.
        learning_rates: A dictionary of learning rates.

    Returns:
        A dictionary containing the merged parameters.
    """
    new_params = {}

    for key, value in learning_rates["params"].items():
        layer_name, param_id = key.split("_", 1)
        new_layer_name = f"{layer_name}LR_{param_id}"
        if new_layer_name not in new_params:
            new_params[new_layer_name] = value

    new_params.update(params["params"])

    return FrozenDict({"params": new_params})


def clip_learning_rates(params):
    """Clip the learning rates to the range [0, 1].

    Args:
        params: A dictionary of parameters.

    Returns:
        A dictionary containing the clipped learning rates.
    """
    params_true, learning_rates_true = separate_learning_rates(unfreeze(params))
    clipped_learning_rates = jax.tree_map(lambda lr: jnp.clip(lr, 0, 1), learning_rates_true)
    new_params = merge_learning_rates(unfreeze(params_true), unfreeze(clipped_learning_rates))
    return new_params


def batch_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Perform element-wise multiplication of two arrays.

    Args:
        a: First array.
        b: Second array.

    Returns:
        The element-wise multiplication of the two arrays.
    """
    return jax.vmap(lambda x, y: x * y)(a, b)


def normalize_coordinates(coordinates: jnp.ndarray, max_coordinate: Any) -> jnp.ndarray:
    """Normalize coordinates to the range [-1, 1].

    Args:
        coordinates: The coordinates to normalize.
        max_coordinate: The maximum coordinate value.

    Returns:
        The normalized coordinates.
    """
    # Get points in range [-0.5, 0.5]
    # normalized_coordinates = coordinates / (max_coordinate - 1) - 0.5
    normalized_coordinates = coordinates / max_coordinate - 0.5
    # Convert to range [-1, 1]
    normalized_coordinates *= 2
    # normalized_coordinates = coordinates / max_coordinate
    return normalized_coordinates


def make_coordinates(batch_size: int, shape: Any, num_channels: int = None) -> jnp.ndarray:
    """Make coordinates for a given shape.

    Args:
        batch_size: The batch size.
        shape: The shape of the coordinates.
        num_channels: The number of channels.

    Returns:
        A Numpy Array of coordinates.
    """
    x = jnp.stack(jnp.ones(shape).nonzero(), -1) * jnp.ones([batch_size, 1, 1])
    x = normalize_coordinates(x, max_coordinate=jnp.max(x))
    if len(shape) == 2:
        grid_size = shape[0] * shape[1]
    else:
        grid_size = shape[0]
    if num_channels is not None:
        y_aux = jnp.ones((batch_size, grid_size, num_channels))
        x = jnp.concatenate([x, y_aux], axis=-1)

    # initialize t to 1
    t = jnp.ones((batch_size, grid_size, 1))

    x = jnp.concatenate([x, t], axis=-1)
    return x


def make_grid_image(ndarray: Any, inverse_scaler: Callable, padding: int = 2, pad_value: float = 0.0) -> Any:
    """Make a grid image from a Numpy Array.

    Args:
        ndarray: The Numpy Array.
        inverse_scaler: The inverse scaler.
        padding: The padding.
        pad_value: The padding value.

    Returns:
        The grid image.
    """
    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    n_row = int(np.sqrt(ndarray.shape[0]))
    # make the mini-batch of images into a grid
    n_maps = ndarray.shape[0]
    x_maps = min(n_row, n_maps)
    ymaps = int(math.ceil(float(n_maps) / x_maps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = np.full((height * ymaps + padding, width * x_maps + padding, num_channels), pad_value).astype(np.float32)
    k = 0
    for y in range(ymaps):
        for x in range(x_maps):
            if k >= n_maps:
                break
            grid[
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ] = ndarray[k]
            k = k + 1

    ndarr = inverse_scaler(grid)
    ndarr = jnp.clip(ndarr * 255, 0, 255).astype(jnp.uint8)
    return ndarr
