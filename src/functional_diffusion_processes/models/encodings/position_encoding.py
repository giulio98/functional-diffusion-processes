from typing import Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0
) -> np.ndarray:
    """Generates a 2D positional embedding using sine and cosine functions.

    Args:
        embed_dim (int): The dimension of the embedding.
        grid_size (int): The size of the grid.
        cls_token (bool, optional): Whether to include a cls_token. Defaults to False.
        extra_tokens (int, optional): Number of extra tokens to add. Defaults to 0.

    Returns:
        np.ndarray: A 2D array of shape [grid_size*grid_size, embed_dim] or
            [1+grid_size*grid_size, embed_dim] (with or without cls_token).
    """
    grid_h = np.linspace(0, 1, grid_size, dtype=np.float32)
    grid_w = np.linspace(0, 1, grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generates 2D positional embedding from a given grid using sine and cosine functions.

    Args:
        embed_dim (int): Output dimension for each position.
        grid (np.ndarray): A grid to be encoded.

    Returns:
        np.ndarray: 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: Union[np.ndarray, list]) -> np.ndarray:
    """Generates 1D positional embedding from a given grid using sine and cosine functions.

    Args:
        embed_dim (int): Output dimension for each position.
        pos (Union[np.ndarray, list]): A list of positions to be encoded.

    Returns:
        np.ndarray: 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = (1.0 / 10000**omega) * 64  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# noinspection PyAttributeOutsideInit
class AddPositionEncodings(nn.Module):
    """Module to add positional encodings to the inputs.

    Attributes:
        num_hiddens (int): Number of hidden units.
        image_size (int): Size of the image.
        old_image_size (int): Original size of the image before resizing.
        patch_size (int): Size of each patch.

    Methods:
        setup(): Set up the positional encodings based on the module attributes.
        __call__(inputs: jnp.ndarray) -> jnp.ndarray: Applies the AddPositionEncodings module to the inputs.
    """

    num_hiddens: int
    image_size: int
    old_image_size: int
    patch_size: int

    def setup(self):
        """Set up the positional encodings based on the module attributes.

        Calculates the positional encodings to be used based on the attributes of the module.

        Returns:
            np.ndarray: Positional encoding.
        """
        self.pos_encoding = get_2d_sincos_pos_embed(self.num_hiddens, self.old_image_size // self.patch_size)
        self.old_seq_len = self.old_image_size**2 // self.patch_size**2
        self.new_seq_len = self.image_size**2 // self.patch_size**2
        self.root_old_seq_len = int(self.old_seq_len ** (1 / 2))
        self.root_new_seq_len = int(self.new_seq_len ** (1 / 2))
        return self.pos_encoding

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEncodings module to the inputs.

        Args:
            inputs (jnp.ndarray): Input data of shape `(batch_size, seq_length, num_hiddens)`.

        Returns:
            jnp.ndarray: Output data with positional encodings added of shape `(batch_size, seq_length, num_hiddens)`.

        Raises:
            AssertionError: If the number of dimensions in the input is not 3.
        """
        assert inputs.ndim == 3, "Number of dimensions should be 3, but it is: %d" % inputs.ndim

        # Separate the time step embedding
        time_step_embedding = inputs[:, -1, :]
        inputs_without_timestep = inputs[:, :-1, :]

        if self.old_seq_len != self.new_seq_len:
            pos_encoding = self.pos_encoding.reshape(
                (1, self.root_old_seq_len, self.root_old_seq_len, self.num_hiddens)
            )

            new_shape = (1, self.root_new_seq_len, self.root_new_seq_len, self.num_hiddens)

            pos_encoding = jax.image.resize(pos_encoding, new_shape, method="bilinear")

            pos_encoding = pos_encoding.reshape((1, self.new_seq_len, self.num_hiddens))
        else:
            pos_encoding = self.pos_encoding.reshape((1, self.old_seq_len, self.num_hiddens))

        # Add the positional encoding to the inputs (without timestep)
        encoded_inputs = inputs_without_timestep + pos_encoding

        # Reconcatenate the time step embedding
        encoded_inputs_with_timestep = jnp.concatenate([encoded_inputs, time_step_embedding[:, None, :]], axis=1)

        return encoded_inputs_with_timestep
