from typing import Any, Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


# noinspection PyAttributeOutsideInit
class AddPositionEmbs(nn.Module):
    """Module for adding learned positional embeddings to the inputs.

    This module is designed to apply learned positional embeddings to the input tensor. It is
    useful in Transformer architectures where the addition of positional embeddings is essential for
    capturing the order information among the elements in the input sequence.

    Attributes:
        posemb_init (Callable): Function to initialize the positional embeddings.
        old_image_size (int): The size of the image before resizing.
        patch_size (int): The size of each patch.
        image_size (int): The current size of the image.

    Methods:
        setup(): Computes the sequence lengths based on the image and patch sizes.
        __call__(inputs: jnp.ndarray) -> jnp.ndarray: Applies learned positional embeddings to the inputs.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
    old_image_size: int
    patch_size: int
    image_size: int

    def setup(self):
        """Computes sequence lengths based on image and patch sizes.

        Calculates the sequence lengths based on the original and current image sizes and the patch size.
        These values are used to determine the size of the positional embedding tensor.
        """
        self.old_seq_len = self.old_image_size // self.patch_size + 1
        self.new_seq_len = self.image_size // self.patch_size + 1

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies learned positional embedding to the inputs.

        This method takes an input tensor and adds learned positional embeddings to it. If the image size
        has changed, it resizes the positional embeddings tensor using bilinear interpolation before adding
        it to the input tensor.

        Args:
            inputs (jnp.ndarray): Input tensor of shape `(batch_size, sequence_length + 1, embedding_dimension)`.

        Returns:
            jnp.ndarray: Output tensor of shape `(batch_size, sequence_length + 1, embedding_dimension)`.
        """
        assert inputs.ndim == 3, f"Number of dimensions should be 3, but it is: {inputs.ndim}"
        pos_emb_shape = (1, self.old_seq_len, inputs.shape[2])
        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape)
        if self.old_image_size != self.image_size:
            pe = jax.image.resize(pe, (1, self.new_seq_len, inputs.shape[2]), method="bilinear")
        return inputs + pe
