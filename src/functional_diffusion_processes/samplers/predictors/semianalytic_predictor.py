from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.random import PRNGKeyArray

from ...sdetools import SDE
from .base_predictor import Predictor

Params = FrozenDict[str, Any]


class SemiAnalyticPredictor(Predictor):
    """A predictor utilizing a Semi-Analytic approach to perform state updates.

    The SemiAnalyticPredictor class extends the abstract Predictor class,
    implementing a Semi-Analytic method for updating states based on a provided
    Stochastic Differential Equation (SDE) object. This method attempts to approximate
    the drift term analytically to improve the prediction of the next state.

    Methods:
        update_fn(rng: PRNGKeyArray, params: Params, predict_fn: Callable,
                  y_corrupted: jnp.ndarray, batch_input: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Executes one update of the predictor using the Semi-Analytic approach to compute
            the drift term and updates the state accordingly.
    """

    def __init__(self, sde: SDE) -> None:
        """Initializes the SemiAnalyticPredictor object with a given SDE object.

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
        """Performs one update of the predictor using the Semi-Analytic method.

        This method computes the drift term using a Semi-Analytic approach and updates the
        state of the system accordingly.

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
        y_reconstructed = predict_fn(params, y_corrupted, batch_input, t, psm, shape)
        rng, step_rng = jax.random.split(rng)
        drift, _ = self.reverse_sde.semi_analytic(y_corrupted, t, step_rng, y_reconstructed)
        y_corrupted = y_corrupted + dt * drift

        rng, step_rng = jax.random.split(rng)
        y_corrupted_mean, y_corrupted = sde.diffuse(step_rng, y_corrupted, t - dt, t)
        if sde.sde_config.probability_flow:
            y_corrupted = y_corrupted_mean

        dth = sde.sde_config.factor * dt
        tn = t - dt + dth

        rng, step_rng = jax.random.split(rng)
        y_corrupted_mean, y_corrupted = sde.diffuse(step_rng, y_corrupted, tn, t - dt)
        psm = sde.get_psm(tn)
        y_reconstructed = predict_fn(params, y_corrupted, batch_input, tn, psm, shape)
        rng, step_rng = jax.random.split(rng)
        drift, _ = self.reverse_sde.semi_analytic(y_corrupted, tn, step_rng, y_reconstructed)
        y_corrupted = y_corrupted + dth * drift

        rng, step_rng = jax.random.split(rng)
        sde.diffuse(step_rng, y_corrupted, tn - dth, tn)
        if sde.sde_config.probability_flow:
            y_corrupted = y_corrupted_mean

        return y_corrupted, y_corrupted_mean
