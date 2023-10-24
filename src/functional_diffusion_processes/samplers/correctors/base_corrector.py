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
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from flax.core import FrozenDict
from jax.random import PRNGKeyArray

from ...sdetools.base_sde import SDE

Params = FrozenDict[str, Any]


class Corrector(abc.ABC):
    """The abstract base class for a corrector algorithm, designed to work on Stochastic Differential Equations (SDEs).

    A corrector algorithm aims to correct the trajectory of an SDE based on some criteria,
    typically to reduce the error or improve stability in numerical simulations.

    Attributes:
        sde (SDE): An object instance of the Stochastic Differential Equation to work on.
        snr (float): The Signal-to-Noise Ratio, typically used to assess the quality or performance of the corrector.
        n_steps (int): The number of steps to be taken by the algorithm.
        score_fn (Callable): The scoring function derived from the SDE object, used to evaluate or guide the correction.

    Methods:
        update_fn(rng: PRNGKeyArray, params: Params, predict_fn: Callable, y_corrupted: jnp.ndarray,
                  batch_input: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Abstract method to define one update/correction step of the corrector algorithm.
    """

    def __init__(self, sde: SDE, snr: float, n_steps: int) -> None:
        """Initialize the Corrector with an SDE object, Signal-to-Noise Ratio, and number of steps.

        Args:
            sde (SDE): The Stochastic Differential Equation object which the corrector algorithm will work on.
            snr (float): The Signal-to-Noise Ratio for evaluating the performance of the correction.
            n_steps (int): The number of correction steps to be performed by the algorithm.
        """
        super().__init__()
        self.sde = sde
        self.score_fn = self.sde.score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(
        self,
        rng: PRNGKeyArray,
        params: Params,
        predict_fn: Callable,
        y_corrupted: jnp.ndarray,
        batch_input: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """One update step of the corrector, to be implemented by subclasses.

        This abstract method defines the structure of a single update/correction step in the algorithm.
        Subclasses should provide a concrete implementation of this method.

        Args:
            rng (PRNGKeyArray): A JAX random state used for any stochastic operations in the update.
            params (Params): Model parameters.
            predict_fn (Callable): The model prediction function.
            y_corrupted (jnp.ndarray): A JAX array representing the corrupted data at the current step.
            batch_input (jnp.ndarray): A JAX array representing the input data for the current step.
            t (jnp.ndarray): A JAX array representing the current time step in the algorithm.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - y_corrupted (jnp.ndarray): A JAX array representing the updated state after the correction.
                - y_corrupted_mean (jnp.ndarray): A JAX array representing the updated state without random noise,
                  useful for denoising.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the update_fn method.")
