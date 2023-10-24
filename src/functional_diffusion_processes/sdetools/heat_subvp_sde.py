import abc
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKeyArray
from omegaconf import DictConfig

from ..sdetools.base_sde import SDE
from ..sdetools.sde_utils import construct_b_and_r, rand_phase
from ..utils.common import batch_mul


class HeatSubVPSDE(SDE, abc.ABC):
    """Heat SubVP SDE class representing a Stochastic Differential Equation (SDE) for a Heat SubVP model.

    Inherits from a base SDE class and implements the specific behavior of the Heat SubVP SDE.
    """

    def __init__(self, sde_config: DictConfig):
        """Initialize a Heat SubVP SDE instance with the provided configuration.

        Args:
            sde_config (DictConfig): The configuration dictionary containing the settings for the SDE.
        """
        super().__init__(sde_config)
        self.beta_max = sde_config.beta_max
        self.beta_min = sde_config.beta_min
        self.shape = sde_config.shape
        self.is_unidimensional = True if len(self.shape) == 1 else False
        self.x_norm = sde_config.x_norm
        self.energy_norm = sde_config.energy_norm
        self.b, self.r = construct_b_and_r(self.x_norm, self.energy_norm, shape=self.shape)

        self.const = sde_config.const
        self.c = jnp.log((self.beta_min + (self.beta_max - self.beta_min) / 2) / self.const + 1)

    def _get_beta(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute the beta function value at a given time t.

        Args:
            t (jnp.ndarray): The time at which to compute the beta function.

        Returns:
            jnp.ndarray: The value of beta at time t.
        """
        return self.const * self.c * jnp.exp(t * self.c)

    def _int_beta(self, t: jnp.ndarray, t0: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute the integrated beta value from 0 to t or from t0 to t.

        Args:
            t (jnp.ndarray): The time until which to integrate beta.
            t0 (Optional[jnp.ndarray], optional): The starting time of integration. Defaults to None.

        Returns:
            jnp.ndarray: The integrated beta value.
        """
        k = self.const * (jnp.exp(t * self.c) - 1)
        if t0 is not None:
            k -= self.const * (jnp.exp(t0 * self.c) - 1)
        return k

    def _k(self, t: jnp.ndarray, t0: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute the k function value at a given time t.

        Args:
            t (jnp.ndarray): The time at which to compute the k function.
            t0 (Optional[jnp.ndarray], optional): The starting time for computing k. Defaults to None.

        Returns:
            jnp.ndarray: The value of k at time t.
        """
        int_beta = self._int_beta(t, t0)
        k = jnp.exp(-int_beta * self.b)
        return k

    def sde(
        self,
        y_corrupted: jnp.ndarray,
        t: jnp.float32,
        rng: Optional[PRNGKeyArray] = None,
        y_reconstructed: Optional[jnp.ndarray] = None,
    ) -> Tuple[Any, Array | Any]:
        """Compute the drift and diffusion terms of the SDE at a given time t.

        Args:
            y_corrupted (jnp.ndarray): The corrupted state of the system.
            t (jnp.float32): The current time.
            rng (Optional[PRNGKeyArray], optional): The random number generator for any stochastic processes. Defaults to None.
            y_reconstructed (Optional[jnp.ndarray], optional): The reconstructed state of the system. Defaults to None.

        Returns:
            Tuple[Any, Array | Any]: The drift and diffusion terms of the SDE.
        """
        t = jnp.expand_dims(t, axis=-1)
        beta_t = self._get_beta(t)
        b, g, c = y_corrupted.shape
        y_corrupted_freq = self.fourier_transform(state=y_corrupted.reshape(b, *self.shape, c))
        y_corrupted_freq = self.b.reshape(1, *self.shape, 1) * y_corrupted_freq
        f = jnp.real(self.inverse_fourier_transform(state=y_corrupted_freq)).reshape(b, g, c)
        drift = -beta_t * f
        diffusion = jnp.sqrt(jnp.abs(2 * beta_t))
        return drift, diffusion

    def marginal_prob(
        self,
        rng: PRNGKeyArray,
        x: jnp.ndarray,
        t: jnp.float32,
        t0: Optional[jnp.ndarray] = None,
    ) -> Tuple[Any, jnp.ndarray | Any]:
        """Compute the marginal probability of the SDE at a given time t.

        Args:
            rng (PRNGKeyArray): The random number generator for any stochastic processes.
            x (jnp.ndarray): The state of the system.
            t (jnp.float32): The current time.
            t0 (Optional[jnp.ndarray], optional): The initial time. Defaults to None.

        Returns:
            Tuple[Any, jnp.ndarray | Any]: The marginal probability of the SDE and the factor to scale the Fourier transform of the noise.
        """
        b, g, c = x.shape
        x_freq = self.fourier_transform(state=x.reshape(b, *self.shape, c))
        t = jnp.expand_dims(t, axis=-1)
        if t0 is not None:
            t0 = jnp.expand_dims(t0, axis=-1)
        k_t = self._k(t, t0).reshape(b, *self.shape, 1)
        x_freq = batch_mul(k_t, x_freq)
        x0t = jnp.real(self.inverse_fourier_transform(state=x_freq)).reshape(b, g, c)
        phase = rand_phase(rng, (b, *self.shape, c))
        fact = self.r / self.b * (1 - self._k(t, t0) ** 2)
        sigma = jnp.sqrt(jnp.abs(fact)).reshape(b, *self.shape, 1)
        noise_std = batch_mul(sigma, phase)
        return x0t, noise_std

    def prior_sampling(
        self, rng: PRNGKeyArray, shape: Tuple[int, ...], t0: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Sample from the prior distribution of the SDE.

        Args:
            rng (PRNGKeyArray): The random number generator for any stochastic processes.
            shape (Tuple[int, ...]): The shape of the sample to be generated.
            t0 (Optional[jnp.ndarray], optional): The initial time. Defaults to None.

        Returns:
            jnp.ndarray: A sample from the prior distribution of the SDE.
        """
        b, g, c = shape
        t = jnp.ones((b, 1))
        t = jnp.expand_dims(t, axis=(-1))
        if t0 is not None:
            t0 = jnp.expand_dims(t0, axis=(-1))
        # Generate the noise as a Gaussian random variable.
        z = jax.random.normal(key=rng, shape=shape)
        z_freq = self.fourier_transform(state=z.reshape(b, *self.shape, c))

        rng, step_rng = jax.random.split(rng)
        phase = rand_phase(step_rng, z_freq.shape)
        fact = self.r / self.b * (1 - self._k(t, t0) ** 2)
        sigma = jnp.sqrt(jnp.abs(fact)).reshape(b, *self.shape, 1)
        z = jnp.real(self.inverse_fourier_transform(state=batch_mul(batch_mul(sigma, z_freq), phase))).reshape(b, g, c)
        return z

    def score_fn(
        self, y_corrupted: jnp.ndarray, y_reconstructed: jnp.ndarray, t: jnp.ndarray, rng: Optional[PRNGKeyArray] = None
    ) -> jnp.ndarray:
        """Compute the score function of the SDE.

        Args:
            y_corrupted (jnp.ndarray): The corrupted state of the system.
            y_reconstructed (jnp.ndarray): The reconstructed state of the system.
            t (jnp.ndarray): The current time.
            rng (Optional[PRNGKeyArray], optional): The random number generator for any stochastic processes. Defaults to None.

        Returns:
            jnp.ndarray: The score function of the SDE.
        """
        b, g, c = y_corrupted.shape
        t = jnp.expand_dims(t, axis=(-1))
        k_t = self._k(t).reshape(b, *self.shape, 1)
        y_corrupted_freq = self.fourier_transform(state=y_corrupted.reshape(b, *self.shape, c))
        y_reconstructed_freq = self.fourier_transform(state=y_reconstructed.reshape(b, *self.shape, c))
        fact = 1 / self.b * (1 - self._k(t) ** 2)
        sigma_squared = fact.reshape(b, *self.shape, 1)
        score_freq = -(y_corrupted_freq - batch_mul(k_t, y_reconstructed_freq)) / sigma_squared
        score = jnp.real(self.inverse_fourier_transform(state=score_freq)).reshape(b, g, c)
        return score

    def get_psm(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute the Pwer-Special-Matrix (PSM) at a given time t to weight the loss function.

        Args:
            t (jnp.ndarray): The current time.

        Returns:
            jnp.ndarray: The state-dependent diffusion matrix.
        """
        if not self.is_unidimensional:
            t = jnp.expand_dims(t, axis=(-1))
        psm = jnp.expand_dims(jnp.ones_like(t) * jnp.sqrt(self.b / self.r).reshape(1, *self.shape), -1)
        return psm

    def get_reverse_noise(self, rng: PRNGKeyArray, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Generate the noise to multiply the diffusion of the reverse SDE.

        Args:
            rng (PRNGKeyArray): The random number generator for any stochastic processes.
            shape (Tuple[int, ...]): The shape of the noise to be generated.

        Returns:
            jnp.ndarray: The noise to multiply the diffusion of the reverse SDE.
        """
        b, g, c = shape
        noise = jax.random.normal(rng, shape)
        noise_freq = self.fourier_transform(state=noise.reshape(b, *self.shape, c))

        rng, step_rng = jax.random.split(rng)
        phase = rand_phase(step_rng, noise_freq.shape)

        fact = jnp.sqrt(jnp.abs(self.r)).reshape(1, *self.shape, 1)
        noise = jnp.real(self.inverse_fourier_transform(state=batch_mul(fact * noise_freq, phase))).reshape(b, g, c)
        return noise

    def diffuse(
        self, rng: PRNGKeyArray, x: jnp.ndarray, t: jnp.ndarray, t0: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Diffuse the input from time t0 to time t.

        This method evolves the state of the system from an initial time t0 to a final time t
        using the dynamics defined by the SDE.

        Args:
            rng (PRNGKeyArray): The random number generator for any stochastic processes.
            x (jnp.ndarray): The initial state of the system.
            t (jnp.ndarray): The final time.
            t0 (Optional[jnp.ndarray], optional): The initial time. Defaults to None.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the mean of the corrupted input and the corrupted input.
        """
        b, g, c = x.shape
        noise = jax.random.normal(rng, (b, g, c))
        noise_freq = self.fourier_transform(state=noise.reshape(b, *self.shape, c))

        rng, step_rng = jax.random.split(rng)
        y_corrupted_mean, std = self.marginal_prob(step_rng, x, t, t0)

        noise_std = jnp.real(self.inverse_fourier_transform(batch_mul(std, noise_freq)).reshape(b, g, c))
        y_corrupted = y_corrupted_mean + noise_std
        return y_corrupted_mean, y_corrupted
