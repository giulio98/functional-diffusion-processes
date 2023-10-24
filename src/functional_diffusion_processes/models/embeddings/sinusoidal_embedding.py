import math

import jax.numpy as jnp


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """Generates positional embeddings for given timesteps using sine and cosine functions.

    The function follows the approach outlined in the "Attention is All You Need" paper to
    create positional embeddings. This method of creating positional embeddings is designed
    to be easily learnable by models.

    Args:
        timesteps (jnp.ndarray): A 1D array containing the timesteps for which embeddings
            need to be generated.
        embedding_dim (int): The dimensionality of the embeddings to be generated.
        max_positions (int, optional): A scaling factor used in the calculation of the
            positional embeddings. Defaults to 10000.

    Returns:
        jnp.ndarray: A 2D array of shape `(num_timesteps, embedding_dim)` containing the
            generated positional embeddings.

    Raises:
        AssertionError: If `timesteps` is not a 1D array.

    Note:
        The code for this function has been ported from the DDPM codebase available at:
        https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
