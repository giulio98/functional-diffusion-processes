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

import jax.numpy as jnp
from flax.core import FrozenDict
from jax.random import PRNGKeyArray

from ...sdetools.base_sde import SDE
from .base_corrector import Corrector

Params = FrozenDict[str, Any]


class NoneCorrector(Corrector):
    """A specialized Corrector that performs no correction.

    The NoneCorrector is a subclass of the Corrector abstract base class.

    Inherits:
        Corrector: The abstract base class for corrector algorithms.

    Methods:
        update_fn(rng: PRNGKeyArray, params: Params, predict_fn: Callable, y_corrupted: jnp.ndarray,
                  batch_input: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Overrides the update_fn method from Corrector to provide a no-operation update function.
    """

    def __init__(self, sde: SDE, snr: float, n_steps: int) -> None:
        """Initialize the NoneCorrector with SDE object, Signal-to-Noise Ratio, and number of steps.

        The initialization merely passes the arguments to the base class initializer.

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
        """A no-operation update function that returns the input data unmodified.

        Args:
            rng (PRNGKeyArray): A JAX random state, not used in this implementation.
            params (Params): Model parameters, not used in this implementation.
            predict_fn (Callable): The model prediction function, not used in this implementation.
            y_corrupted (jnp.ndarray): A JAX array representing the corrupted data at the current step.
            batch_input (jnp.ndarray): A JAX array representing the input data for the current step, not used in this implementation.
            t (jnp.ndarray): A JAX array representing the current time step in the algorithm, not used in this implementation.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - y_corrupted (jnp.ndarray): The unmodified input data representing the corrupted state.
                - y_corrupted (jnp.ndarray): The unmodified input data representing the corrupted state.
        """
        return y_corrupted, y_corrupted
