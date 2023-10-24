import abc
from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax.random import PRNGKeyArray
from omegaconf import DictConfig

from ..utils.common import batch_mul


class SDE(abc.ABC):
    """Abstract base class for representing Stochastic Differential Equations (SDEs).

    This class provides a structured way to define and work with SDEs, including computing
    Fourier transforms, discretizing the equations, and defining the drift and diffusion terms.

    Attributes:
        sde_config (DictConfig): Configuration object containing SDE settings.
        T (float): Total time duration.
        N (int): Number of time steps.
        eps (float): Small constant for numerical stability.
        is_unidimensional (bool): Flag indicating if the SDE is unidimensional.
    """

    def __init__(self, sde_config: DictConfig) -> None:
        """Initializes the SDE with the given configuration.

        Args:
            sde_config (DictConfig): Configuration object containing SDE settings.
        """
        super().__init__()
        self.sde_config = sde_config
        self.T = self.sde_config.T
        self.N = self.sde_config.N
        self.eps = self.sde_config.eps
        self.is_unidimensional = True if len(self.sde_config.shape) == 1 else False

    def fourier_transform(self, state: jnp.ndarray) -> jnp.ndarray:
        """Computes the Fourier transform of the given state.

        This method can handle both vectorized and non-vectorized input states.

        Args:
            state (jnp.ndarray): State whose Fourier transform is to be computed.

        Returns:
            jnp.ndarray: Fourier transform of the given state.
        """
        return (
            jnp.fft.fft(state, norm="ortho", axis=1)
            if self.is_unidimensional
            else jnp.fft.fft2(state, norm="ortho", axes=(1, 2))
        )

    def inverse_fourier_transform(self, state: jnp.ndarray) -> jnp.ndarray:
        """Computes the inverse Fourier transform of the given state.

        This method can handle both vectorized and non-vectorized input states.

        Args:
            state (jnp.ndarray): State whose inverse Fourier transform is to be computed.

        Returns:
            jnp.ndarray: Inverse Fourier transform of the given state.
        """
        return (
            jnp.fft.ifft(state, norm="ortho", axis=1)
            if self.is_unidimensional
            else jnp.fft.ifft2(state, norm="ortho", axes=(1, 2))
        )

    @abc.abstractmethod
    def sde(
        self,
        y_corrupted: jnp.ndarray,
        t: jnp.ndarray,
        rng: Optional[PRNGKeyArray] = None,
        y_reconstructed: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Abstract method to compute the drift and diffusion terms of the SDE.

        Args:
            y_corrupted (jnp.ndarray): Corrupted state of the system.
            t (jnp.ndarray): Current time.
            rng (Optional[PRNGKeyArray], optional): Random number generator. Defaults to None.
            y_reconstructed (Optional[jnp.ndarray], optional): Reconstructed state of the system. Defaults to None.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing the drift and diffusion terms of the SDE.

        Raises:
            NotImplementedError: If this method is not overridden by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the sde method.")

    @abc.abstractmethod
    def marginal_prob(
        self,
        rng: PRNGKeyArray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        t0: Optional[jnp.ndarray] = None,
    ) -> Tuple[Any, jnp.ndarray | Any]:
        """Computes the marginal probability density at a given time.

        This is an abstract method that should be overridden by subclasses to
        compute the marginal probability density based on the state and time.

        Args:
            rng (PRNGKeyArray): Random number generator.
            x (jnp.ndarray): State of the system.
            t (jnp.ndarray): Current time.
            t0 (Optional[jnp.ndarray], optional): Initial time. Defaults to None.

        Returns:
            Tuple[Any, jnp.ndarray | Any]: Marginal probability density at the given time.

        Raises:
            NotImplementedError: If this method is not overridden by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the marginal_prob method.")

    @abc.abstractmethod
    def diffuse(
        self, rng: PRNGKeyArray, x: jnp.ndarray, t: jnp.ndarray, t0: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs diffusion of the input from time t0 to time t.

        This is an abstract method that should be overridden by subclasses to
        implement the diffusion process based on the state and time.

        Args:
            rng (PRNGKeyArray): Random number generator.
            x (jnp.ndarray): Input state.
            t (jnp.ndarray): Current time.
            t0 (Optional[jnp.ndarray], optional): Initial time. Defaults to None.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Mean of the corrupted input and the corrupted input.

        Raises:
            NotImplementedError: If this method is not overridden by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the diffuse method.")

    @abc.abstractmethod
    def prior_sampling(
        self, rng: PRNGKeyArray, shape: Tuple[int, ...], t0: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Generates a sample from the prior distribution of the SDE.

        This is an abstract method that should be overridden by subclasses to
        implement the prior sampling process based on the shape and initial time.

        Args:
            rng (PRNGKeyArray): Random number generator.
            shape (Tuple[int, ...]): Shape of the sample to be generated.
            t0 (Optional[jnp.ndarray], optional): Initial time. Defaults to None.

        Returns:
            jnp.ndarray: A sample from the prior distribution of the SDE.

        Raises:
            NotImplementedError: If this method is not overridden by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the prior_sampling method.")

    @abc.abstractmethod
    def score_fn(
        self, y_corrupted: jnp.ndarray, y_reconstructed: jnp.ndarray, t: jnp.ndarray, rng: Optional[PRNGKeyArray] = None
    ) -> jnp.ndarray:
        """Computes the score function based on the corrupted and reconstructed states.

        This is an abstract method that should be overridden by subclasses to
        compute the score function based on the state and time.

        Args:
            y_corrupted (jnp.ndarray): Corrupted state of the system.
            y_reconstructed (jnp.ndarray): Reconstructed state of the system.
            t (jnp.ndarray): Current time.
            rng (Optional[PRNGKeyArray], optional): Random number generator. Defaults to None.

        Returns:
            jnp.ndarray: The score function.

        Raises:
            NotImplementedError: If this method is not overridden by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the score_fn method.")

    @abc.abstractmethod
    def get_psm(self, t: jnp.ndarray) -> jnp.ndarray:
        """Computes the Power-Special-Matrix(PSM) used as a weighting factor for the loss.

        This is an abstract method that should be overridden by subclasses to
        compute the state-dependent diffusion matrix based on the time.

        Args:
            t (jnp.ndarray): Current time.

        Returns:
            jnp.ndarray: The state-dependent diffusion matrix.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the get_psm method.")

    @abc.abstractmethod
    def get_reverse_noise(self, rng: PRNGKeyArray, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Generates noise for the reverse SDE.

        This is an abstract method that should be overridden by subclasses to
        generate reverse noise based on the shape.

        Args:
            rng (PRNGKeyArray): Random number generator.
            shape (Tuple[int, ...]): Shape of the noise to be generated.

        Returns:
            jnp.ndarray: The reverse noise.

        Raises:
            NotImplementedError: If this method is not overridden by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the get_reverse_noise method.")

    def discretize(
        self,
        y_corrupted: jnp.ndarray,
        t: jnp.ndarray,
        y_reconstructed: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Discretizes the SDE into an iterative update rule.

        This method computes the discrete drift and diffusion terms based on the continuous SDE.

        Args:
            y_corrupted (jnp.ndarray): Corrupted state of the system.
            t (jnp.ndarray): Current time.
            y_reconstructed (Optional[jnp.ndarray], optional): Reconstructed state of the system. Defaults to None.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing the discrete drift and diffusion terms.
        """
        dt = (self.T - self.eps) / self.N
        drift, diffusion = self.sde(y_corrupted, t, y_reconstructed)
        f = drift * dt
        g = diffusion * jnp.sqrt(dt)
        return f, g

    def reverse(self):
        """Creates a reverse-time version of the current SDE.

        This method defines a nested class for the reverse-time SDE and returns an instance of it.

        Returns:
            ReverseSDE: An instance of the reverse-time SDE subclass.
        """
        num_time_steps = self.N
        end_t = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize
        score_fn = self.score_fn
        sde_config = self.sde_config

        class ReverseSDE(self.__class__, abc.ABC):
            """Reverse Stochastic Differential Equation abstract base class."""

            def __init__(self) -> None:
                """Initialize the ReverseSDE class.

                Inherits the properties from the original SDE class and overrides the relevant methods for the
                reverse-time SDE.
                """
                super().__init__(sde_config)
                self.N = num_time_steps
                self.T = end_t
                self.score_fn = score_fn

            def sde(
                self,
                y_corrupted: jnp.ndarray,
                t: jnp.ndarray,
                rng: Optional[PRNGKeyArray] = None,
                y_reconstructed: Optional[jnp.ndarray] = None,
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
                """Return the drift and diffusion terms for the reverse-time SDE.

                Args:
                    y_corrupted (jnp.ndarray): Corrupted state of the system.
                    t (jnp.ndarray): Current time.
                    rng (Optional[PRNGKeyArray], optional): Random number generator. Defaults to None.
                    y_reconstructed (Optional[jnp.ndarray], optional): Reconstructed state of the system. Defaults to None.

                Returns:
                    Tuple[jnp.ndarray, jnp.ndarray]: Drift and diffusion terms for the reverse-time SDE.
                """
                drift, diffusion = sde_fn(y_corrupted, t, y_reconstructed)
                score = self.score_fn(y_corrupted, y_reconstructed, t, rng=rng)
                drift = -drift + batch_mul(diffusion**2, score * (0.5 if self.sde_config.probability_flow else 1.0))
                # Set the diffusion function to zero for ODEs.
                diffusion = jnp.zeros_like(diffusion) if self.sde_config.probability_flow else diffusion
                return drift, diffusion

            def discretize(
                self,
                y_corrupted: jnp.ndarray,
                t: jnp.ndarray,
                rng: Optional[PRNGKeyArray] = None,
                y_reconstructed: Optional[jnp.ndarray] = None,
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
                """Discretizes the reverse-time SDE in the form of an iterative update rule.

                Args:
                    y_corrupted (jnp.ndarray): Corrupted state of the system.
                    t (jnp.ndarray): Current time.
                    rng (Optional[PRNGKeyArray], optional): Random number generator. Defaults to None.
                    y_reconstructed (Optional[jnp.ndarray], optional): Reconstructed state of the system. Defaults to None.

                Returns:
                    Tuple[jnp.ndarray, jnp.ndarray]: Drift and diffusion terms for the discretized reverse-time SDE.
                """
                f, g = discretize_fn(y_corrupted, t, y_corrupted)
                rev_f = -f + batch_mul(
                    g**2,
                    self.score_fn(y_corrupted, y_reconstructed, t, rng=rng)
                    * (0.5 if self.sde_config.probability_flow else 1.0),
                )
                rev_g = jnp.zeros_like(g) if self.sde_config.probability_flow else g
                return rev_f, rev_g

            def semi_analytic(
                self,
                y_corrupted: jnp.ndarray,
                t: jnp.ndarray,
                rng: Optional[PRNGKeyArray] = None,
                y_reconstructed: Optional[jnp.ndarray] = None,
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
                """Computes the semi-analytic drift and diffusion terms for the reverse-time SDE.

                Args:
                    y_corrupted (jnp.ndarray): Corrupted state of the system.
                    t (jnp.ndarray): Current time.
                    rng (Optional[PRNGKeyArray], optional): Random number generator. Defaults to None.
                    y_reconstructed (Optional[jnp.ndarray], optional): Reconstructed state of the system. Defaults to None.

                Returns:
                    Tuple[jnp.ndarray, jnp.ndarray]: Drift and diffusion terms for the semi-analytic reverse-time SDE.
                """
                _, diffusion = sde_fn(y_corrupted, t, y_reconstructed)
                score = self.score_fn(y_corrupted, y_reconstructed, t, rng=rng)
                drift = batch_mul(diffusion**2, score * (0.5 if self.sde_config.probability_flow else 1.0))
                diffusion = jnp.zeros_like(diffusion) if self.sde_config.probability_flow else diffusion
                return drift, diffusion

        return ReverseSDE()
