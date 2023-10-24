from typing import Any, Optional

import einops
import flax.linen as nn
import jax.nn
import jax.numpy as jnp

Dtype = Any


class Attention(nn.Module):
    """Transformer attention block.

    This module implements the multi-head self-attention mechanism, which is a
    core component of Transformer models. It follows the design of attention
    mechanism from "Attention is All You Need", Vaswani et al., 2017.

    Attributes:
        dim (int): Dimensionality of the input and output space.
        dtype (Dtype, optional): The data type of the computation. Defaults to jnp.float32.
        num_heads (int, optional): The number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): If True, include bias terms in the query, key, value
            linear transformations. Defaults to False.
        qk_scale (Optional[float], optional): Scaling factor for the query-key dot product.
            If None, scale by `head_dim ** -0.5`. Defaults to None.
        attn_drop (float, optional): Dropout rate for the attention probabilities. Defaults to 0.0.
        proj_drop (float, optional): Dropout rate for the output. Defaults to 0.0.

    Methods:
        __call__(inputs: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
            Compute the multi-head self-attention for a given input tensor.

    Raises:
        ValueError: If the `dim` attribute is not divisible by `num_heads`.
    """

    dim: int
    dtype: Dtype = jnp.float32
    num_heads: int = 8
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        """Compute the multi-head self-attention for a given input tensor.

        This method computes the multi-head self-attention given an input tensor.
        The computation follows the standard process of splitting the input into queries, keys,
        and values, computing the scaled dot-product attention, and then recombining
        the heads into the final output tensor.

        Args:
            inputs (jnp.ndarray): Input tensor of shape `(batch_size, seq_length, input_dim)`.
                This tensor is transformed into queries, keys, and values for self-attention computation.
            deterministic (bool): Whether to apply dropout in a deterministic manner.
                If `True`, dropout is skipped, which is typically desired at evaluation time.

        Returns:
            jnp.ndarray: The result of the multi-head self-attention, of shape
                `(batch_size, seq_length, output_dim)`.

        Raises:
            ValueError: If the `dim` attribute is not divisible by `num_heads`.
        """
        b, l, c = inputs.shape

        if self.dim % self.num_heads != 0:
            raise ValueError(f"`dim` ({self.dim}) must be divisible by `num_heads` ({self.num_heads})")

        head_dim = self.dim // self.num_heads
        scale = self.qk_scale or head_dim**-0.5

        qkv = nn.Dense(
            features=3 * self.dim,
            kernel_init=nn.initializers.lecun_normal(),
            use_bias=self.qkv_bias,
            dtype=self.dtype,
            name="qkv",
        )(inputs)
        qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D

        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop)(attn, deterministic=deterministic)

        x = jnp.swapaxes((attn @ v), 1, 2).reshape(b, l, c)

        x = nn.Dense(features=self.dim, dtype=self.dtype, kernel_init=nn.initializers.lecun_normal(), name="proj")(x)

        x = nn.Dropout(rate=self.proj_drop)(x, deterministic=deterministic)

        return x
