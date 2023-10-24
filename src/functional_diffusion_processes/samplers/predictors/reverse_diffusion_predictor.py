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

from ...sdetools.base_sde import SDE
from ...utils.common import batch_mul
from .base_predictor import Predictor

Params = FrozenDict[str, Any]


class ReverseDiffusionPredictor(Predictor):
    """A predictor class utilizing the reverse diffusion process for updating states.

    This predictor operates based on a reverse diffusion process that attempts to reverse
    the corruption in the data to make accurate predictions of the original signal. It
    makes use of a given Stochastic Differential Equation (SDE) object to derive the
    necessary dynamics for prediction.

    Methods:
        update_fn(rng: PRNGKeyArray, params: Params, predict_fn: Callable,
                  y_corrupted: jnp.ndarray, batch_input: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Executes one update of the predictor, applying reverse diffusion to approximate
            the original uncorrupted data from the given corrupted data.
    """

    def __init__(self, sde: SDE) -> None:
        """Initializes the ReverseDiffusionPredictor object with a given SDE object.

        Args:
            sde (SDE): The stochastic differential equation governing the system's dynamics.
        """
        super().__init__(sde)

    def update_fn(
        self,
        rng: PRNGKeyArray,
        params: Params,
        predict_fn: Callable,
        y_corrupted: jnp.ndarray,
        batch_input: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs one update of the predictor based on the reverse diffusion process.

        This method applies a step of reverse diffusion to approximate the original
        uncorrupted data from the given corrupted data. It leverages the associated
        Stochastic Differential Equation (SDE) to derive the dynamics necessary for
        the update.

        Args:
            rng (PRNGKeyArray): A JAX random state.
            params (Params): The parameters of the model.
            predict_fn (Callable): The model prediction function.
            y_corrupted (jnp.ndarray): A JAX array representing the corrupted data.
            batch_input (jnp.ndarray): A JAX array representing the input of the model.
            t (jnp.ndarray): A JAX array representing the current time step.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - y_corrupted (jnp.ndarray): The updated state of the system.
                - y_corrupted_mean (jnp.ndarray): The updated state of the system without the stochastic noise.
        """
        sde = self.sde
        psm = sde.get_psm(t)
        shape = sde.sde_config.shape
        y_reconstructed = predict_fn(params, y_corrupted, batch_input, t, psm, shape)
        rng, step_rng = jax.random.split(rng)
        f, g = self.reverse_sde.discretize(y_corrupted, t, step_rng, y_reconstructed)
        noise = self.sde.get_reverse_noise(rng, y_corrupted.shape)
        y_corrupted_mean = y_corrupted + f
        y_corrupted = y_corrupted_mean + batch_mul(g, noise)
        return y_corrupted, y_corrupted_mean
