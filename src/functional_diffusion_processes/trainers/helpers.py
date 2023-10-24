from functools import partial
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import optax

from ..models import BaseMAML
from ..samplers import Sampler
from ..utils.common import clip_learning_rates


def construct_train_step(optimizer, loss_fn) -> Callable:
    """Construct a train step function to be used in the training loop.

    This function creates a training step function which, when called, performs
    a single step of training including forward pass, loss computation, and
    backward pass for gradient computation and updates.

    Args:
        optimizer: The optimizer instance used for updating model parameters.
        loss_fn: The loss function used for computing the loss.

    Returns:
        Callable: The constructed train step function.
    """

    @partial(jax.pmap, axis_name="device")
    def train_fn(
        rng,
        params,
        optim_params,
        step,
        batch_input,
        batch,
    ):
        grad_params, (new_rng, loss, loss_inner, batch_reconstructed, batch_corrupted, target) = loss_fn(
            rng, params, step, batch_input, batch
        )

        loss = jax.lax.pmean(loss, axis_name="device")
        grad_params = jax.lax.pmean(grad_params, axis_name="device")

        updates, optim_params = optimizer.update(grad_params, optim_params, params)

        params = optax.apply_updates(params, updates)
        params = clip_learning_rates(params)
        return new_rng, loss, loss_inner, params, optim_params, batch_reconstructed, batch_corrupted, target

    return train_fn


def construct_sampling_fn(model: flax.linen.Module, sampler: Sampler) -> Callable:
    """Construct a sampling function for generating samples from the model.

    Args:
        model (flax.linen.Module): The model instance from which to generate samples.
        sampler (Sampler): The sampler instance used for sampling.

    Returns:
        Callable: The constructed sampling function.
    """
    predict_fn = model.make_predict_fn()
    if isinstance(model, BaseMAML):
        super_resolution_fn = model.make_super_resolution_fn()
        sample_fn = sampler.make_sampler(predict_fn, super_resolution_fn)
    else:
        sample_fn = sampler.make_sampler(predict_fn, None)
    return sample_fn


def sampling_fn(sample_fn: Callable, carry_state: Tuple, batch_input: jnp.ndarray) -> Tuple:
    """Perform sampling task using a given sampling function.

    Args:
        sample_fn (Callable): The sampling function.
        carry_state (Tuple): The current state of the model.
        batch_input (jnp.ndarray): The input data for sampling.

    Returns:
        Tuple: The updated state after performing the sampling.
    """
    (rng, state) = carry_state
    return sample_fn(rng, batch_input, state.ema_params)


def colorizing_fn(
    sample_fn: Callable, carry_state: Tuple, batch_input: jnp.ndarray, gray_scale_img: jnp.ndarray
) -> Tuple:
    """Perform colorizing task on a given grayscale image.

    Args:
        sample_fn (Callable): The sampling function used for colorization.
        carry_state (Tuple): The current state of the model.
        batch_input (jnp.ndarray): The input data for colorization.
        gray_scale_img (jnp.ndarray): The grayscale image to be colorized.

    Returns:
        Tuple: The updated state and the colorized image.
    """
    (rng, state) = carry_state
    return sample_fn(rng, batch_input, state.ema_params, gray_scale_img)


def inpainting_fn(
    sample_fn: Callable, carry_state: Tuple, batch_input: jnp.ndarray, image: jnp.ndarray, mask: jnp.ndarray
) -> Tuple:
    """Perform inpainting task on a given image using a mask.

    Args:
        sample_fn (Callable): The sampling function used for inpainting.
        carry_state (Tuple): The current state of the model.
        batch_input (jnp.ndarray): The input data for inpainting.
        image (jnp.ndarray): The image to be inpainted.
        mask (jnp.ndarray): The mask used for inpainting.

    Returns:
        Tuple: The updated state and the inpainted image.
    """
    (rng, state) = carry_state
    return sample_fn(rng, batch_input, state.ema_params, image, mask)
