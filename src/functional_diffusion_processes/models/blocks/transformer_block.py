from typing import Optional

import jax.numpy as jnp
from flax import linen as nn

from ..attention.attention import Attention
from .mlp_block import MlpBlock


class Block(nn.Module):
    """Transformer Block comprising a multi-head self-attention layer and a feed-forward neural network.

    This module encapsulates a typical block within Transformer-based architectures,
    consisting of a multi-head self-attention mechanism followed by a feed-forward
    neural network. Optionally, a skip connection mechanism can be activated.

    Attributes:
        mlp_dim (int): Dimensionality of the feed-forward neural network (MLP) following the attention block.
        num_heads (int): Number of attention heads within the multi-head self-attention mechanism.
        dtype (jnp.float32): The data type used for computation, default is float32.
        mlp_ratio (float): Scaling factor for the dimensionality of the MLP, usually set to 4.0 (default: 4.0).
        dropout_rate (float): Dropout rate applied post-attention and in the output layers (default: 0.1).
        attention_dropout_rate (float): Dropout rate applied to the attention scores (default: 0.1).
        skip (bool): Toggle for enabling a skip connection mechanism (default: False).

    Methods:
        __call__(inputs: jnp.ndarray, skip: Optional[jnp.ndarray], *, deterministic: bool) -> jnp.ndarray:
            Forward pass through the Transformer block.
    """

    mlp_dim: int
    num_heads: int
    dtype: jnp.float32
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    skip: bool = False  # added parameter to control skip mechanism

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, skip: Optional[jnp.ndarray] = None, *, deterministic: bool) -> jnp.ndarray:
        """Applies the Transformer block to the input tensor.

        This method orchestrates the flow of data through the Transformer block, encompassing
        a multi-head self-attention mechanism and a feed-forward neural network. Optionally,
        a skip connection mechanism can be activated, which involves an additional dense layer.

        Args:
            inputs (jnp.ndarray): Input tensor of shape `(batch_size, seq_length, input_dim)`.
            skip (Optional[jnp.ndarray], optional): An optional tensor for the skip connection of shape
                `(batch_size, seq_length, skip_dim)`. If provided and `skip` attribute is True,
                a skip connection mechanism is activated. Defaults to None.
            deterministic (bool): Flag to determine whether to apply dropout in a deterministic manner.

        Returns:
            jnp.ndarray: Output tensor of shape `(batch_size, seq_length, mlp_dim // mlp_ratio)`.
        """
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"

        # if skip is available, apply dense layer on concatenated inputs and skip
        if self.skip and skip is not None:
            inputs = nn.Dense(
                self.mlp_dim // self.mlp_ratio,
                kernel_init=nn.initializers.normal(stddev=0.02),
                bias_init=nn.initializers.constant(0),
                dtype=self.dtype,
            )(jnp.concatenate([inputs, skip], axis=-1))

        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = Attention(
            dim=self.mlp_dim // self.mlp_ratio,
            num_heads=self.num_heads,
            dtype=self.dtype,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=self.attention_dropout_rate,
            proj_drop=self.dropout_rate,
        )(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            out_dim=self.mlp_dim // self.mlp_ratio,
        )(y, deterministic=deterministic)

        return x + y
