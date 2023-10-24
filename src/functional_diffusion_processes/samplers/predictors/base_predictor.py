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


class Predictor(abc.ABC):
    """Abstract base class for a predictor algorithm working on Stochastic Differential Equations (SDEs).

    A predictor algorithm attempts to predict the next state of a system given its current state. This abstract class defines the interface that such
    algorithms must adhere to.

    Attributes:
        sde (SDE): The stochastic differential equation governing the system's dynamics.
        reverse_sde (SDE): The reverse of the original SDE, computed during initialization.

    Methods:
        update_fn(rng: PRNGKeyArray, params: Params, predict_fn: Callable,
                  y_corrupted: jnp.ndarray, batch_input: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Abstract method for performing one update step of the predictor.
    """

    def __init__(self, sde: SDE) -> None:
        """Initializes the Predictor object with a given stochastic differential equation.

        Computes the reverse of the provided SDE and stores both the original and reversed SDE.

        Args:
            sde (SDE): The stochastic differential equation governing the system's dynamics.
        """
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.reverse_sde = sde.reverse()

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
        """Performs one update step of the predictor algorithm.

        This abstract method defines the interface for a single update step of the predictor.
        Concrete implementations must provide the logic for computing the next state of the system
        based on the current state and possibly other information.

        Args:
            rng (PRNGKeyArray): A JAX random state.
            params (Params): The parameters of the model.
            predict_fn (Callable): The model prediction function.
            y_corrupted (jnp.ndarray): A JAX array representing the corrupted data.
            batch_input (jnp.ndarray): A JAX array representing the input of the model.
            t (jnp.ndarray): A JAX array representing the current time step.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - y_corrupted (jnp.ndarray): A JAX array of the next state.
                - y_corrupted_mean (jnp.ndarray): A JAX array of the next state without random noise, useful for denoising.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the update_fn method.")
