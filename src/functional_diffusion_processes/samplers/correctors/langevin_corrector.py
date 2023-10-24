# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.random import PRNGKeyArray

from src.functional_diffusion_processes.utils.common import batch_mul

from ...sdetools.base_sde import SDE
from .base_corrector import Corrector

Params = FrozenDict[str, Any]


class LangevinCorrector(Corrector):
    """Implementation of Langevin Corrector for denoising corrupted data.

    Langevin Corrector is an algorithm that iteratively refines the corrupted data
    by applying corrections derived from the gradients of a score function,
    driven by a stochastic differential equation (SDE).

    Inherits:
        Corrector: The abstract class for a corrector algorithm.

    Methods:
        update_fn(rng: PRNGKeyArray, params: Params, predict_fn: Callable,
                  y_corrupted: jnp.ndarray, batch_input: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Performs one update of the Langevin corrector on the corrupted data.
    """

    def __init__(self, sde: SDE, snr: float, n_steps: int) -> None:
        """Initialize the LangevinCorrector with SDE object, Signal-to-Noise Ratio, and number of steps.

        Args:
            sde (SDE): The Stochastic Differential Equation object which the corrector algorithm will work on.
            snr (float): The Signal-to-Noise Ratio for evaluating the performance of the correction.
            n_steps (int): The number of correction steps to be performed by the algorithm.
        """
        super().__init__(sde, snr, n_steps)

    def update_fn(
        self,
        rng: PRNGKeyArray,
        params: Params,
        predict_fn: Callable,
        y_corrupted: jnp.ndarray,
        batch_input: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs one update of the Langevin corrector on the corrupted data.

        This method applies one step of correction to the corrupted data by calculating
        the gradient of the score function, and then updating the corrupted data in the
        direction of the gradient while adding a controlled amount of stochastic noise.

        Args:
            rng (PRNGKeyArray): A JAX random state.
            params (Params): Model parameters.
            predict_fn (Callable): The model prediction function.
            y_corrupted (jnp.ndarray): A JAX array representing the corrupted data.
            batch_input (jnp.ndarray): A JAX array representing the input of the model.
            t (jnp.ndarray): A JAX array representing the current time step.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - y_corrupted (jnp.ndarray): A JAX array of the updated corrupted data.
                - y_corrupted_mean (jnp.ndarray): A JAX array of the updated corrupted data without random noise.
        """
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        alpha = jnp.ones_like(t)

        def loop_body(_, val):
            rng_, params_, y_corrupted_, y_corrupted_mean_, batch_input_ = val
            psm = sde.get_psm(t)
            b, g, c = y_corrupted_.shape
            shape = self.sde.sde_config.shape
            y_reconstructed = predict_fn(
                params=params_, batch_corrupted=y_corrupted_, batch_input=batch_input_, time=t, psm=psm, shape=shape
            )
            rng_, step_rng = jax.random.split(rng_)
            grad = score_fn(y_corrupted_, y_reconstructed, t, step_rng)
            rng_, step_rng = jax.random.split(rng_)
            noise = self.sde.get_reverse_noise(step_rng, y_corrupted_.shape)
            if self.sde.sde_config.frequency_space:
                noise = self.sde.fourier_transform(state=noise.reshape(b, *shape, c))
            grad_norm = jnp.linalg.norm(grad.reshape((grad.shape[0], -1)), axis=-1).mean()
            grad_norm = jax.lax.pmean(grad_norm, axis_name="device")
            noise_norm = jnp.linalg.norm(noise.reshape((noise.shape[0], -1)), axis=-1).mean()
            noise_norm = jax.lax.pmean(noise_norm, axis_name="device")

            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            y_corrupted_mean_ = y_corrupted_ + batch_mul(step_size, grad)
            if self.sde.sde_config.frequency_space:
                noise_std = jnp.real(
                    self.sde.inverse_fourier_transform(state=batch_mul(noise, jnp.sqrt(step_size * 2)))
                ).reshape(b, g, c)
            else:
                noise_std = batch_mul(noise, jnp.sqrt(step_size * 2))
            y_corrupted_ = y_corrupted_mean_ + noise_std
            return rng_, params_, y_corrupted_, y_corrupted_mean_, batch_input

        *_, y_corrupted, y_corrupted_mean, _ = jax.lax.fori_loop(
            0, n_steps, loop_body, (rng, params, y_corrupted, y_corrupted, batch_input)
        )
        return y_corrupted, y_corrupted_mean
