from abc import ABC
from typing import Dict, TypeVar

import flax.linen as nn
import flax.linen.initializers as init
import jax.numpy as jnp

from .base_maml import BaseMAML
from .layers.dense_lr import DenseLR

T = TypeVar("T")


# noinspection PyAttributeOutsideInit
class MLPModulationLR(BaseMAML, ABC):
    """A Multi-Layer Perceptron (MLP) model with modulation.

    Inspired by the papers "From data to functa: Your data point is a function and you can treat it like one" (https://arxiv.org/abs/2201.12204)
    and "WIRE: Wavelet Implicit Neural Representations" (https://arxiv.org/abs/2301.05187).

    This implementation introduces skip connections and utilizes the Gabor wavelet activation function.

    Inherits from `BaseMAML` for meta-learning capabilities.

    Attributes:
        model_config (Dict): Configuration dictionary for the model.
        optimizer_inner (Dict): Inner optimizer configuration.
        inner_steps (int): Number of inner optimization steps.
        layer_sizes (List[int]): Sizes of layers.
        modulation_freq (float): Frequency for the modulation.
        exp_const (float): Exponential constant for the modulation.
        use_dense_lr (bool): Flag to indicate usage of dense layer with learnable learning rates.
        skip_connections (int): Flag to indicate usage of skip connections.
        uniform_min_val (float): Minimum value for the uniform distribution of learnable learning rates.
        uniform_max_val (float): Maximum value for the uniform distribution of learnable learning rates.

    Methods:
        setup(): Set up the model with configurations specified in `model_config`.
        __call__(inputs: jnp.ndarray): Forward pass through the model.
    """

    model_config: Dict
    optimizer_inner: Dict
    inner_steps: int

    def setup(self):
        """Set up the MLPModulation module.

        Initialize the model configurations from `model_config`.
        """
        self.layer_sizes = self.model_config["layer_sizes"]
        self.modulation_freq = self.model_config["modulation_freq"]
        self.exp_const = self.model_config["exp_const"]
        self.use_dense_lr = self.model_config["use_dense_lr"]
        self.skip_connections = self.model_config["skip_connections"]
        self.uniform_min_val = self.model_config["uniform_min_val"]
        self.uniform_max_val = self.model_config["uniform_max_val"]

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies the MLP Modulation to the input.

        Pass the input through a series of dense layers with modulation
        and optionally with learnable learning rates and skip connections.

        Args:
            inputs (jnp.ndarray): A tensor representing the input to the MLP.

        Returns:
            jnp.ndarray: A tensor representing the output of the MLP, after passing through the modulated layers.
        """
        latents = inputs

        for i, layer_size in enumerate(self.layer_sizes[:-1]):
            scale = 1 / latents.shape[-1] if i == 0 else jnp.sqrt(6 / latents.shape[-1]) / self.modulation_freq
            kernel_init = lambda *args: (init.uniform(scale=2)(*args) - 1) * scale
            if self.use_dense_lr:
                DenseLR(
                    features=layer_size,
                    latents=latents,
                    learning_rate_min_val=self.uniform_min_val,
                    learning_rate_max_val=self.uniform_max_val,
                )(inputs=latents)

            latents = nn.Dense(features=layer_size, kernel_init=kernel_init)(inputs=latents)

            latents = jnp.sin(self.modulation_freq * latents) * jnp.exp(-self.exp_const * latents**2)

            # Skip connections
            if self.skip_connections == 1:
                latents = jnp.concatenate([latents, inputs], axis=-1)

        scale = jnp.sqrt(6 / latents.shape[-1]) / self.modulation_freq
        kernel_init = lambda *args: (init.uniform(scale=2)(*args) - 1) * scale
        y_inner = nn.Dense(self.layer_sizes[-1], kernel_init=kernel_init)(latents)
        if self.use_dense_lr:
            DenseLR(
                features=self.layer_sizes[-1],
                latents=latents,
                learning_rate_min_val=self.uniform_min_val,
                learning_rate_max_val=self.uniform_max_val,
            )(inputs=latents)
        return y_inner
