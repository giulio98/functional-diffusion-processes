from typing import Any

import jax.numpy as jnp
from flax import linen as nn

Dtype = Any


# noinspection PyAttributeOutsideInit
class PatchEmbeddings(nn.Module):
    """Module for embedding image patches.

    This module takes an input tensor representing images and extracts patches from the images,
    which are then embedded into a specified embedding dimension using either 1D or 2D convolution.

    Attributes:
        patch_size (int): Size of each patch to be embedded.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of the embedding space.
        is_unidimensional (bool): Indicator to use 1D convolution, if set to True, otherwise 2D convolution is used.
        dtype (Dtype): Data type of the computation, default is jnp.float32.

    Methods:
        setup(): Configures the convolution settings based on input dimensionality.
        __call__(x: jnp.ndarray) -> jnp.ndarray: Embeds patches of the input tensor using 1D or 2D convolution.
    """

    patch_size: int
    in_chans: int
    embed_dim: int
    is_unidimensional: bool
    dtype: Dtype = jnp.float32

    def setup(self):
        """Configures convolution settings.

        Sets up the convolution configurations including kernel size, strides, and padding,
        based on whether the input is unidimensional or not.
        """
        self.kernel_size = (self.patch_size,) if self.is_unidimensional else (self.patch_size, self.patch_size)
        self.strides = self.kernel_size  # Strides are the same as kernel_size for patch embedding
        self.padding = "VALID"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embeds patches of the input tensor.

        Extracts patches from the input tensor and embeds them into the specified embedding dimension
        using either 1D or 2D convolution based on the `is_unidimensional` flag.

        Args:
            x (jnp.ndarray): Input tensor with images.

        Returns:
            jnp.ndarray: Tensor containing the embedded patches.
        """
        b, *_ = x.shape
        x = nn.Conv(
            features=self.embed_dim,
            dtype=self.dtype,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            name="patch_embeddings",
        )(x)
        x = x.reshape((b, -1, self.embed_dim))
        return x
