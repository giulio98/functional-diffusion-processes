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

import abc
from functools import partial
from typing import Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax import Array
from jax.random import PRNGKeyArray
from omegaconf import DictConfig

from ..sdetools import SDE
from . import Sampler
from .correctors.base_corrector import Corrector
from .predictors.base_predictor import Predictor

Params = FrozenDict[str, Any]


class PCInpainter(Sampler, abc.ABC):
    """A class for a predictor-corrector (PC) sampler tailored for inpainting and deblurring tasks.

    The PCInpainter class extends the generic Sampler class to provide a specialized
    implementation for handling image inpainting and deblurring tasks using a stochastic
    differential equation (SDE) based approach.

    Attributes:
        predictor (Predictor): An instance of the Predictor class for estimating the next state.
        corrector (Corrector): An instance of the Corrector class for refining the estimated state.
        sde (SDE): A stochastic differential equation model describing the diffusion process.
        sampler_config (DictConfig): A configuration object containing sampler settings.
    """

    def __init__(self, predictor: Predictor, corrector: Corrector, sde: SDE, sampler_config: DictConfig) -> None:
        """Initializes an instance of the PCInpainter class.

        Args:
            predictor (Predictor): The predictor object for the sampler.
            corrector (Corrector): The corrector object for the sampler.
            sde (SDE): The SDE object for the sampler.
            sampler_config (DictConfig): The configuration for the sampler.
        """
        super().__init__(predictor=predictor, corrector=corrector, sde=sde, sampler_config=sampler_config)

    def make_sampler(self, predict_fn: Callable, _: Union[Any, Callable]) -> Callable:
        """Creates a sampler function for processing images, specifically for inpainting or deblurring tasks.

        Given a prediction function, this method returns a sampling function that operates in a parallel
        manner across multiple devices. The sampling function processes image data, applying prediction
        and correction steps to generate inpainted or deblurred images.

        Args:
            predict_fn (Callable): The model prediction function.
            _ (Callable): The auxiliary function for the model. Not used here.

        Returns:
            Callable: A function that performs sampling to process image data.
        """
        times = (
            jnp.linspace(self.sampler_config.T, self.sampler_config.eps, self.sampler_config.N + 1)
            ** self.sampler_config.k
        )

        def get_inpaint_update_fn(update_fn: Callable) -> Callable:
            """Modifies the update function to include data masking, which is essential for inpainting and deblurring tasks.

            Args:
                update_fn (Callable): The original update function.

            Returns:
                Callable: A modified update function that incorporates data masking.
            """

            def inpaint_update_fn(
                rng: PRNGKeyArray,
                params: Params,
                batch_input: jnp.ndarray,
                image: jnp.ndarray,
                mask: jnp.ndarray,
                x: jnp.ndarray,
                t: jnp.ndarray,
            ) -> Tuple[Array, Array]:
                """Performs an update step for inpainting or deblurring, handling Fourier transforms and masking operations.

                Args:
                    rng (random.PRNGKey): Random number generator key.
                    params (Params): Parameters dictionary.
                    batch_input (Array): Batch input data.
                    image (Array): Input image data.
                    mask (Array): Mask data.
                    x (Array): Current state.
                    t (float): Time value.

                Returns:
                    Tuple[Array, Array]: Updated state and mean state.
                """
                sde = self.sde
                rng, step_rng = jax.random.split(rng)
                dt = (sde.sde_config.T - sde.sde_config.eps) / sde.sde_config.N
                x, x_mean = update_fn(rng, params, predict_fn, x, batch_input, t)
                rng, step_rng = jax.random.split(rng)
                masked_data_mean, masked_data = sde.diffuse(step_rng, image, t - dt)
                if self.sampler_config.task == "deblurring":
                    b, g, c = x.shape
                    shape = sde.sde_config.shape
                    x_freq = self.sde.fourier_transform(state=x.reshape(b, *shape, c))
                    masked_data_freq = self.sde.fourier_transform(state=masked_data.reshape(b, *shape, c))
                    masked_data_mean_freq = self.sde.fourier_transform(state=masked_data_mean.reshape(b, *shape, c))
                    if self.sde.is_unidimensional:
                        x_freq = jnp.fft.fftshift(x_freq, axes=1)
                        masked_data_freq = jnp.fft.fftshift(masked_data_freq, axes=1)
                        masked_data_mean_freq = jnp.fft.fftshift(masked_data_mean_freq, axes=1)
                    else:
                        x_freq = jnp.fft.fftshift(x_freq, axes=(1, 2))
                        masked_data_freq = jnp.fft.fftshift(masked_data_freq, axes=(1, 2))
                        masked_data_mean_freq = jnp.fft.fftshift(masked_data_mean_freq, axes=(1, 2))

                    mask = mask.reshape(b, *shape, c)
                    x_freq = x_freq * (1.0 - mask) + masked_data_freq * mask

                    x_freq_mean = x_freq * (1.0 - mask) + masked_data_mean_freq * mask

                    if self.sde.is_unidimensional:
                        x_freq = jnp.fft.ifftshift(x_freq, axes=1)
                        x_freq_mean = jnp.fft.ifftshift(x_freq_mean, axes=1)
                    else:
                        x_freq = jnp.fft.ifftshift(x_freq, axes=(1, 2))
                        x_freq_mean = jnp.fft.ifftshift(x_freq_mean, axes=(1, 2))

                    x = jnp.real(self.sde.inverse_fourier_transform(state=x_freq)).reshape(b, g, c)

                    x_mean = jnp.real(self.sde.inverse_fourier_transform(state=x_freq_mean)).reshape(b, g, c)
                else:
                    x = x * (1.0 - mask) + masked_data * mask
                    x_mean = x * (1.0 - mask) + masked_data_mean * mask
                return x, x_mean

            return inpaint_update_fn

        corrector_inpaint_update_fn = get_inpaint_update_fn(self.corrector.update_fn)
        predictor_inpaint_update_fn = get_inpaint_update_fn(self.predictor.update_fn)

        def _step_pc_sample_fn(i: int, val: Tuple) -> Tuple:
            """Executes a single step in the PC sampling process, applying the inpainting update functions for both predictor and corrector.

            Args:
                i (int): Index of the current step.
                val (Tuple): Tuple containing the current values.

            Returns:
                Tuple: Updated values for the next step.
            """
            rng, x, x_mean, batch_input, params, image, mask, history = val
            t = times[i - 1]
            vec_t = t * jnp.ones((x.shape[0], 1))

            rng, step_rng = jax.random.split(rng)

            x, x_mean = corrector_inpaint_update_fn(step_rng, params, batch_input, image, mask, x, vec_t)

            rng, step_rng = jax.random.split(rng)

            x, x_mean = predictor_inpaint_update_fn(step_rng, params, batch_input, image, mask, x, vec_t)

            return rng, x, x_mean, batch_input, params, image, mask, history.at[i - 1].set(x)

        @partial(jax.pmap, axis_name="device")
        def sample_fn(
            rng: PRNGKeyArray, batch_input: jnp.ndarray, params: Params, image: jnp.ndarray, mask: jnp.ndarray
        ) -> Tuple[Any, jnp.ndarray | Any, Any]:
            """Parallel sampling function for executing the inpainting or deblurring process across multiple devices.

            Args:
                rng (PRNGKeyArray): Random number generator array.
                batch_input (jnp.ndarray): Batch input data.
                params (Params): Parameters dictionary.
                image (jnp.ndarray): Input image data.
                mask (jnp.ndarray): Mask data.

            Returns:
                Tuple[Any, jnp.ndarray | Any, Any]: Mean state, updated state, and all steps state.
            """
            b, g, _ = batch_input.shape
            c = self.sampler_config.output_size
            rng, step_rng = jax.random.split(rng)
            # image = image * mask + (1 - mask) * predict_fn(params, coordinates, image, self.sde.get_psm(vec_t))
            batch_start = self.sde.prior_sampling(step_rng, (b, g, c))

            history_buffer = jnp.zeros((self.sampler_config.N,) + batch_start.shape)
            _, x, x_mean, _, _, _, _, x_all_steps = jax.lax.fori_loop(
                1,
                self.sampler_config.N,
                _step_pc_sample_fn,
                (rng, batch_start, batch_start, batch_input, params, image, mask, history_buffer),
            )
            t = times[-1]
            vec_t = t * jnp.ones((b, 1))
            psm = self.sde.get_psm(vec_t)
            shape = self.sde.sde_config.shape
            y_reconstructed = predict_fn(params, x_mean, batch_input, vec_t, psm, shape)
            if self.sampler_config.task == "deblurring":
                y_reconstructed_freq = self.sde.fourier_transform(state=y_reconstructed.reshape(b, *shape, c))
                x_freq = self.sde.fourier_transform(state=x.reshape(b, *shape, c))
                if self.sde.is_unidimensional:
                    y_reconstructed_freq = jnp.fft.fftshift(y_reconstructed_freq, axes=1)
                    x_freq = jnp.fft.fftshift(x_freq, axes=1)
                else:
                    y_reconstructed_freq = jnp.fft.fftshift(y_reconstructed_freq, axes=(1, 2))
                    x_freq = jnp.fft.fftshift(x_freq, axes=(1, 2))

                mask = mask.reshape(b, *shape, c)
                x_freq = y_reconstructed_freq * (1.0 - mask) + x_freq * mask

                if self.sde.is_unidimensional:
                    x_freq = jnp.fft.ifftshift(x_freq, axes=1)
                else:
                    x_freq = jnp.fft.ifftshift(x_freq, axes=(1, 2))
                x = jnp.real(self.sde.inverse_fourier_transform(state=x_freq)).reshape(b, g, c)
            else:
                x = image * mask + (1 - mask) * y_reconstructed
            return x_mean, x, x_all_steps

        return sample_fn
