from typing import Tuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray


def construct_b_and_r(x_norm: float, energy_norm: float, shape: Tuple[int, ...]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Construct b and r matrices based on the given shape.

    The function computes two matrices b and r for a specified shape. These matrices
    are often utilized in processes involving Fourier transformations, where b acts
    as a scaling factor for frequency components and r serves as a normalization factor
    to maintain the energy of the Fourier transform.

    Args:
        x_norm (float): The normalization factor for the x frequencies.
        energy_norm (float): The normalization factor for r matrix.
        shape (Tuple[int, ...]): A tuple representing the dimensions of the input matrix.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two JAX arrays representing
        the constructed b and r matrices, respectively.
    """
    n_rows = shape[0]

    n = n_rows // 2

    # Construct an array of frequencies in the x direction
    x_freq = jnp.concatenate([jnp.arange(0, n + 1), jnp.arange(n - 1, 0, -1)])

    x_freq = x_freq / (n_rows / x_norm)
    if len(shape) == 2:
        y_freq = x_freq.reshape(-1, 1)
    else:
        y_freq = 0
    r = 1.0 / (x_freq**2 + y_freq**2 + 2.0)
    r = r * 64**2 / 23.27
    b = 1.0 / ((x_freq**2 + y_freq**2) + 0.3) + ((x_freq**2 + y_freq**2 + 2.0) / 33**2) ** (1 / 4)
    r = r / energy_norm
    return b, r


def rand_phase(rng_p: PRNGKeyArray, shape: Tuple[int, ...]) -> jnp.ndarray:
    """Generate a random phase array for a specified shape.

    This function constructs a random phase array which encapsulates phase information
    of matrix frequency components within Fourier transform operations. The generated phase
    array is complex with unit magnitude, and its angles are uniformly distributed
    in the interval [0, 2π].

    Args:
        rng_p (PRNGKeyArray): A JAX random number generator key to seed the random phase generation.
        shape (Tuple[int, ...]): A tuple representing the desired dimensions of the output phase array.

    Returns:
        jnp.ndarray: A JAX array representing the generated random phase array. The phase array
        is complex-valued, and it's structured to adhere to the Hermitian symmetry, which is a
        characteristic of real-valued signals in the Fourier domain.
    """
    if len(shape) == 4:
        phase_shape = (shape[0], shape[1] // 2 + 1, shape[2] // 2 + 1, shape[3])

        # Generate a random phase using a uniform distribution and the given key
        phase = jnp.exp(1j * 2 * jnp.pi * jax.random.uniform(rng_p, phase_shape))

        # Set the DC and Nyquist frequencies to 1.0
        phase = phase.at[:, :, (0, shape[1] // 2), :].set(1.0)

        # Concatenate the phase array with its conjugate reflection along the 2nd dimension
        conjugate_phase = jnp.conjugate(jnp.flip(phase[:, :, 1:-1, :], axis=2))
        cp_phase = jnp.append(phase, conjugate_phase, axis=2)

        # Concatenate the phase array with its reflection along the 1st dimension
        reflected_phase = jnp.flip(cp_phase[:, 1 : shape[1] // 2, :, :], axis=1)
        full_phase = jnp.append(cp_phase, reflected_phase, axis=1)
    else:  # 1d case
        phase_shape = (shape[0], shape[1] // 2 + 1, shape[2])

        # Generate a random phase using a uniform distribution and the given key
        phase = jnp.exp(1j * 2 * jnp.pi * jax.random.uniform(rng_p, phase_shape))

        # Set the DC and Nyquist frequencies to 1.0
        phase = phase.at[:, (0, shape[1] // 2), :].set(1.0)

        # Concatenate the phase array with its conjugate reflection along the 2nd dimension
        conjugate_phase = jnp.conjugate(jnp.flip(phase[:, 1:-1, :], axis=1))
        full_phase = jnp.append(phase, conjugate_phase, axis=1)

    return full_phase
