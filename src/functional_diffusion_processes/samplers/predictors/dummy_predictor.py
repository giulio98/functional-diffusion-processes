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
from .base_predictor import Predictor

Params = FrozenDict[str, Any]


class NonePredictor(Predictor):
    """A null Predictor that performs no prediction.

    The NonePredictor is a subclass of the Predictor abstract base class.

    Inherits:
        Predictor: The abstract base class for predictor algorithms.

    Methods:
        update_fn(rng: PRNGKeyArray, params: Params, predict_fn: Callable,
                  y_corrupted: jnp.ndarray, batch_input: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Performs a null update, returning the input corrupted data without any modifications.
    """

    def __init__(self, sde: SDE) -> None:
        """Initializes the NonePredictor object with a given SDE object.

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
        """Performs a null update, returning the input corrupted data without any modifications.

        Args:
            rng (PRNGKeyArray): A JAX random state.
            params (Params): The parameters of the model.
            predict_fn (Callable): The model prediction function.
            y_corrupted (jnp.ndarray): A JAX array representing the corrupted data.
            batch_input (jnp.ndarray): A JAX array representing the input of the model.
            t (jnp.ndarray): A JAX array representing the current time step.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - y_corrupted (jnp.ndarray): The unmodified input corrupted data.
                - y_corrupted (jnp.ndarray): The unmodified input corrupted data.
        """
        return y_corrupted, y_corrupted
