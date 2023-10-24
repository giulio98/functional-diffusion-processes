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

from ...sdetools import SDE
from .base_predictor import Predictor

Params = FrozenDict[str, Any]


class EulerMaruyamaPredictor(Predictor):
    """A predictor class utilizing the Euler-Maruyama method for approximating solutions of SDEs.

    This predictor is based on the Euler-Maruyama method, a numerical method for solving stochastic
    differential equations (SDEs). It advances the state of the system by approximating the drift
    and diffusion terms of the underlying SDE.

    Methods:
        update_fn(rng: PRNGKeyArray, params: Params, predict_fn: Callable,
                  y_corrupted: jnp.ndarray, batch_input: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Performs one update step of the Euler-Maruyama method, advancing the state of the system.
    """

    def __init__(self, sde: SDE) -> None:
        """Initializes the EulerMaruyamaPredictor object with a given SDE object.

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
        """Performs one update step of the Euler-Maruyama method.

        This method advances the state of the system by approximating the drift and diffusion
        terms of the underlying SDE. It performs one step of the Euler-Maruyama method to
        estimate the next state of the system based on the current state and the stochastic
        nature of the system's dynamics.

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
        dt = (sde.sde_config.T - sde.sde_config.eps) / sde.sde_config.N
        psm = sde.get_psm(t)
        shape = sde.sde_config.shape
        b, g, c = y_corrupted.shape
        y_reconstructed = predict_fn(params, y_corrupted, batch_input, t, psm, shape)

        rng, step_rng = jax.random.split(rng)
        noise = self.sde.get_reverse_noise(step_rng, (b, g, c))
        rng, step_rng = jax.random.split(rng)
        drift, diffusion = self.reverse_sde.sde(y_corrupted, t, step_rng, y_reconstructed)
        y_corrupted_mean = y_corrupted + dt * drift
        y_corrupted = y_corrupted_mean + jnp.sqrt(dt) * batch_mul(diffusion, noise)

        dth = self.sde.sde_config.factor * dt
        tn = t - dt + dth
        psm = sde.get_psm(tn)
        rng, step_rng = jax.random.split(rng)
        y_corrupted_mean, y_corrupted = sde.diffuse(step_rng, y_corrupted, tn, t - dt)

        y_reconstructed = predict_fn(params, y_corrupted, batch_input, tn, psm, shape)

        rng, step_rng = jax.random.split(rng)
        noise = self.sde.get_reverse_noise(step_rng, y_corrupted.shape)
        rng, step_rng = jax.random.split(rng)
        drift, diffusion = self.reverse_sde.sde(y_corrupted, tn, step_rng, y_reconstructed)
        y_corrupted_mean = y_corrupted + dth * drift
        y_corrupted = y_corrupted_mean + jnp.sqrt(dth) * batch_mul(diffusion, noise)

        return y_corrupted, y_corrupted_mean
