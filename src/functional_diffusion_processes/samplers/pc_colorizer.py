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
from jax.random import PRNGKeyArray
from omegaconf import DictConfig

from ..sdetools import SDE
from .base_sampler import Sampler
from .correctors.base_corrector import Corrector
from .predictors.base_predictor import Predictor

Params = FrozenDict[str, Any]


class PCColorizer(Sampler, abc.ABC):
    """A class for PC-Sampler, specialized for colorizing grayscale images using a predefined stochastic differential equation (SDE) model.

    The PCColorizer extends the abstract Sampler class to provide a concrete implementation
    for colorizing grayscale images based on a specified SDE model. It utilizes a predictor-corrector
    approach in the sampling process to generate colorized images.

    Attributes:
        predictor (Predictor): An instance of Predictor to estimate the next state.
        corrector (Corrector): An instance of Corrector to refine the estimated state.
        sde (SDE): A stochastic differential equation model to describe the colorization process.
        sampler_config (DictConfig): A configuration object containing sampler settings.
    """

    def __init__(self, predictor: Predictor, corrector: Corrector, sde: SDE, sampler_config: DictConfig) -> None:
        """Initialize the PCColorizer instance.

        Args:
            predictor (Predictor): An instance of Predictor to estimate the next state.
            corrector (Corrector): An instance of Corrector to refine the estimated state.
            sde (SDE): A stochastic differential equation model to describe the colorization process.
            sampler_config (DictConfig): A configuration object containing sampler settings.
        """
        super().__init__(predictor=predictor, corrector=corrector, sde=sde, sampler_config=sampler_config)

    def make_sampler(self, predict_fn: Callable, _: Union[Any, Callable]) -> Callable:
        """Create a function to perform sampling over a range of time for colorizing images.

        This method constructs a sampling function based on the predictor, corrector, and
        the specified SDE model to generate colorized images from grayscale images. It defines
        specialized update functions for the colorization task and orchestrates the predictor-corrector
        sampling steps.

        Args:
            predict_fn (Callable): The model prediction function.
            _ (Callable): The auxiliary function for the model. Not used here.

        Returns:
            Callable: A function that performs sampling to colorize grayscale images.
        """
        times = (
            jnp.linspace(self.sampler_config.T, self.sampler_config.eps, self.sampler_config.N + 1)
            ** self.sampler_config.k
        )

        def get_mask(image: jnp.ndarray) -> jnp.ndarray:
            """Generate a mask to separate the color channels.

            Args:
                image (jnp.ndarray): The input image tensor.

            Returns:
                jnp.ndarray: A mask tensor to separate color channels.
            """
            mask = jnp.concatenate([jnp.ones_like(image[..., :1]), jnp.zeros_like(image[..., 1:])], axis=-1)
            return mask

        def decouple(inputs: jnp.ndarray) -> jnp.ndarray:
            """Decouple the input tensor using a predefined matrix.

            Args:
                inputs (jnp.ndarray): Input tensor to be decoupled.

            Returns:
                jnp.ndarray: Decoupled tensor.
            """
            return jnp.einsum("BHi,ij->BHj", inputs, matrix)

        # The inverse function to `decouple`.
        def couple(inputs: jnp.ndarray) -> jnp.ndarray:
            """Couple the input tensor using the inverse of the predefined matrix.

            Args:
                inputs (jnp.ndarray): Input tensor to be coupled.

            Returns:
                jnp.ndarray: Coupled tensor.
            """
            return jnp.einsum("BHi,ij->BHj", inputs, inverse_matrix)

        matrix = jnp.asarray(
            [
                [5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                [5.7735026e-01, 4.0824822e-01, -7.0710683e-01],
            ]
        )
        # matrix = jnp.asarray([[0.28361226, 0.95408977, 0.09631611],
        #                  [0.95408977, -0.29083936, 0.07159032],
        #                  [0.09631611, 0.07159032, -0.9927729]])

        # `invM` is the inverse transformation of `matrix`
        inverse_matrix = jnp.linalg.inv(matrix)

        def get_colorization_update_fn(update_fn: Callable) -> Callable:
            """Create a specialized update function for colorization task.

            Args:
                update_fn (Callable): The generic update function.

            Returns:
                Callable: A specialized update function for colorization.
            """

            def colorization_update_fn(
                rng: PRNGKeyArray,
                params: Params,
                batch_input: jnp.ndarray,
                gray_scale_img: jnp.ndarray,
                x: jnp.ndarray,
                t: jnp.ndarray,
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
                """Perform an update step for colorization.

                Args:
                    rng (PRNGKeyArray): Random number generator for stochastic processes.
                    params (Params): Parameters for the model.
                    batch_input (jnp.ndarray): Input data batch.
                    gray_scale_img (jnp.ndarray): Grayscale image to be colorized.
                    x (jnp.ndarray): Current state of the colorization process.
                    t (jnp.ndarray): Current time step in the diffusion process.

                Returns:
                    Tuple[jnp.ndarray, jnp.ndarray]: The updated state and mean state of the colorization process.
                """
                sde = self.sde
                mask = get_mask(x)
                x, x_mean = update_fn(rng, params, predict_fn, x, batch_input, t)
                rng, step_rng = jax.random.split(rng)
                masked_data_mean, masked_data = sde.diffuse(step_rng, decouple(gray_scale_img), t)

                x = couple(decouple(x) * (1.0 - mask) + masked_data * mask)
                x_mean = couple(decouple(x) * (1.0 - mask) + masked_data_mean * mask)
                return x, x_mean

            return colorization_update_fn

        corrector_colorize_update_fn = get_colorization_update_fn(self.corrector.update_fn)
        predictor_colorize_update_fn = get_colorization_update_fn(self.predictor.update_fn)

        def _step_pc_sample_fn(
            i: int, val: Tuple[PRNGKeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Params, jnp.ndarray, jnp.ndarray]
        ) -> Tuple[PRNGKeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Params, jnp.ndarray, jnp.ndarray]:
            """Execute a single step in the PC sampling process for colorization.

            Args:
                i (int): The current step index.
                val (Tuple): The current values of the sampling process.

            Returns:
                Tuple: The updated values of the sampling process.
            """
            rng, x, x_mean, batch_input, params, gray_scale_img, history = val
            t = times[i - 1]
            vec_t = t * jnp.ones((x.shape[0], 1))

            rng, step_rng = jax.random.split(rng)

            x, x_mean = corrector_colorize_update_fn(step_rng, params, batch_input, gray_scale_img, x, vec_t)

            rng, step_rng = jax.random.split(rng)

            x, x_mean = predictor_colorize_update_fn(step_rng, params, batch_input, gray_scale_img, x, vec_t)

            return rng, x, x_mean, batch_input, params, gray_scale_img, history.at[i - 1].set(x)

        @partial(jax.pmap, axis_name="device")
        def sample_fn(
            rng: PRNGKeyArray, batch_input: jnp.ndarray, params: Params, gray_scale_img: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Perform parallel sampling to colorize grayscale images.

            This function orchestrates the predictor-corrector sampling process to
            generate colorized images from grayscale images. It utilizes JAX's parallel
            mapping (pmap) to perform the sampling in parallel across multiple devices.

            Args:
                rng (PRNGKeyArray): Random number generator for stochastic processes.
                batch_input (jnp.ndarray): Input data batch.
                params (Params): Parameters for the model.
                gray_scale_img (jnp.ndarray): Grayscale image to be colorized.

            Returns:
                Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                    - The final colorized state.
                    - The reconstructed colorized image.
                    - The states at all time steps.
            """
            b, g, _ = batch_input.shape
            c = self.sampler_config.output_size
            mask = get_mask(gray_scale_img)
            rng, step_rng = jax.random.split(rng)
            batch_start = couple(
                decouple(gray_scale_img) * mask + decouple(self.sde.prior_sampling(step_rng, (b, g, c)) * (1.0 - mask))
            )
            history_buffer = jnp.zeros((self.sampler_config.N,) + batch_start.shape)
            _, x, x_mean, _, _, _, x_all_steps = jax.lax.fori_loop(
                1,
                self.sampler_config.N,
                _step_pc_sample_fn,
                (rng, batch_start, batch_start, batch_input, params, gray_scale_img, history_buffer),
            )
            t = times[-1]
            vec_t = t * jnp.ones((b, 1))
            psm = self.sde.get_psm(vec_t)
            shape = self.sde.sde_config.shape
            y_reconstructed = predict_fn(params, x_mean, batch_input, vec_t, psm, shape)

            return x, y_reconstructed, x_all_steps

        return sample_fn
