from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class MlpBlock(nn.Module):
    """Transformer MLP (Multi-Layer Perceptron) / feed-forward block.

    This module encapsulates the MLP or feed-forward block used in Transformer models.
    It comprises two dense layers interspersed with a GELU activation function and dropout.

    Attributes:
        mlp_dim (int): The dimensionality of the intermediate layer in the MLP.
        dtype (Dtype, optional): The data type of the computation. Defaults to jnp.float32.
        out_dim (Optional[int], optional): The dimensionality of the output space.
            If None, defaults to the dimensionality of the input space. Defaults to None.
        dropout_rate (float, optional): The dropout rate applied to the output of each layer. Defaults to 0.1.
        kernel_init (Callable[[PRNGKey, Shape, Dtype], Array], optional): Initialization function
            for the weight matrices. Defaults to nn.initializers.xavier_uniform().
        bias_init (Callable[[PRNGKey, Shape, Dtype], Array], optional): Initialization function
            for the biases. Defaults to nn.initializers.normal(stddev=1e-6).

    Methods:
        __call__(inputs: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
            Applies the MLP block to a given input tensor, producing an output tensor.

    """

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        """Applies the MLP block to the inputs.

        This method processes the input tensor through two dense layers with a GELU activation
        function in between. Dropout is applied to the output of each dense layer.

        Args:
            inputs (jnp.ndarray): Input tensor of shape `(batch_size, ..., input_dim)`.
            deterministic (bool): A flag indicating whether to apply dropout in a deterministic
                manner. If `True`, dropout is skipped, which is typically desired at evaluation time.

        Returns:
            jnp.ndarray: The result of processing the input tensor through the MLP block,
                of shape `(batch_size, ..., output_dim)`.
        """
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim

        x = nn.Dense(features=self.mlp_dim, dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(
            inputs
        )

        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        output = nn.Dense(
            features=actual_out_dim, dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)

        return output
