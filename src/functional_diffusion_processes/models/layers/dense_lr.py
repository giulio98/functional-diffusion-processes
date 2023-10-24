from typing import Tuple, TypeVar

import flax.linen as nn
import jax
import jax.numpy as jnp

T = TypeVar("T")


# noinspection PyAttributeOutsideInit
class DenseLR(nn.Module):
    """A linear layer with learnable learning rates specific to the META-SGD algorithm.

    The META-SGD algorithm, introduced in the paper "Meta-SGD: Learning to Learn Quickly for
    Few-Shot Learning" (https://arxiv.org/abs/1707.09835), is an extension of MAML
    (Model-Agnostic Meta-Learning) where the learning rates are also learned during
    meta-training. In this implementation, a dense layer with learnable per-parameter
    learning rates is defined.

    Attributes:
        features (int): The number of output features.
        latents (jnp.ndarray): Latent variables used in the layer.
        learning_rate_min_val (float): The minimum value for the initialized learning rates.
        learning_rate_max_val (float): The maximum value for the initialized learning rates.

    Methods:
        setup() -> None:
            Initializes the learnable learning rates for the kernel and bias of the layer.
        __call__(inputs: jnp.ndarray) -> Tuple[T, T]:
            Returns the learnable learning rates for the kernel and bias when the layer is called.
    """

    features: int
    latents: jnp.ndarray
    learning_rate_min_val: float
    learning_rate_max_val: float

    def setup(self) -> None:
        """Set up the learnable parameters (kernel_lr and bias_lr) of the DenseLR module.

        The learning rates are initialized with random values within the specified range
        [learning_rate_min_val, learning_rate_max_val].
        """
        self.kernel_lr = self.param(
            "kernel",
            lambda rng, shape: jax.random.uniform(
                key=rng,
                shape=shape,
                minval=self.learning_rate_min_val,
                maxval=self.learning_rate_max_val,
            ),
            (self.latents.shape[-1], self.features),
        )
        self.bias_lr = self.param(
            "bias",
            lambda rng, shape: jax.random.uniform(
                key=rng,
                shape=shape,
                minval=self.learning_rate_min_val,
                maxval=self.learning_rate_max_val,
            ),
            (self.features,),
        )

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> Tuple[T, T]:
        """Call the DenseLR module and return the kernel_lr and bias_lr parameters.

        This method is called when the DenseLR module is invoked. It returns the learnable
        learning rates for the kernel and bias.

        Args:
            inputs (jnp.ndarray): The input array.

        Returns:
            Tuple[T, T]: A tuple containing the kernel_lr and bias_lr parameters.
        """
        return self.kernel_lr, self.bias_lr
