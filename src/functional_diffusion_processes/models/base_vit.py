import abc
from typing import Any, Callable, Mapping, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from omegaconf import DictConfig

from ..utils.common import make_coordinates

Params = FrozenDict[str, Any]


class BaseViT(nn.Module, abc.ABC):
    """Abstract base class for Vision Transformer (ViT) models.

    Introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (https://arxiv.org/abs/2010.11929).

    Attributes:
        model_config (DictConfig): Configuration dictionary for the model.
    """

    model_config: DictConfig

    @abc.abstractmethod
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
        """Performs the forward pass of the model.

        Args:
            inputs (jnp.ndarray): Input data.
            train (bool): Indicates whether the model is in training mode.

        Returns:
            jnp.ndarray: Model's output.

        Raises:
            NotImplementedError: If this method is not overridden by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the __call__ method.")

    def initialize_model(self, rng: jax.random.PRNGKey, batch_input: jnp.ndarray) -> FrozenDict[str, Mapping[str, Any]]:
        """Initializes the model with dummy inputs.

        Args:
            rng (jax.random.PRNGKey): The random number generator key.
            batch_input (jnp.ndarray): The input data for batch.

        Returns:
            FrozenDict[str, Mapping[str, Any]]: The initialized model.
        """
        return self.init(rng, batch_input, train=False)

    @staticmethod
    def initialize_input(shape: Tuple[int, ...]) -> jnp.ndarray:
        """Creates input for the model based on the specified shape.

        Args:
            shape (Tuple[int, ...]): The shape of the input.

        Returns:
            jnp.ndarray: The created input.
        """
        batch_size = shape[0]
        num_channels = shape[-1]
        grid_size = shape[1:-1]
        coordinates = make_coordinates(batch_size, grid_size, num_channels)
        return coordinates

    def make_update_params_fn(self) -> Callable:
        """Creates a function to update model parameters.

        Returns:
            Callable: The created function to update model parameters.
        """

        def apply_forward(
            rng: jax.random.PRNGKey, params: Params, batch_input: jnp.ndarray, batch_corrupted: jnp.ndarray, psm: Any
        ) -> Tuple[jax.random.PRNGKey, jnp.ndarray, None]:  # noqa
            """Updates model parameters in a forward pass.

            Args:
                rng (jax.random.PRNGKey): The random number generator key.
                params (Params): The model parameters.
                batch_input (jnp.ndarray): The input data for the batch.
                batch_corrupted (jnp.ndarray): The corrupted version of the output tensor.
                psm (Any): Power special matrix.

            Returns:
                Tuple[jax.random.PRNGKey, jnp.ndarray, None]: A tuple containing a new random key,
                the model output, and the inner loss (which is None in this case).
            """
            _, new_rng = jax.random.split(rng)
            dropout_rng = jax.random.fold_in(rng, jax.lax.axis_index("device"))
            model_output = self.apply(params, rngs={"dropout": dropout_rng}, inputs=batch_input, train=True)
            loss_inner = None
            return new_rng, model_output, loss_inner

        return apply_forward

    def make_predict_fn(self) -> Callable:
        """Creates a function for making predictions with the model.

        Returns:
            Callable: The created function for making predictions.
        """

        def predict(
            params: Params,
            batch_corrupted: jnp.ndarray,
            batch_input: jnp.ndarray,
            time: jnp.ndarray,
            psm: jnp.ndarray,
            shape: Tuple[int, ...],
        ) -> jnp.ndarray:  # noqa
            """Makes predictions with the model.

            Args:
                params (Params): The model parameters.
                batch_corrupted (jnp.ndarray): The corrupted version of the output tensor.
                batch_input (jnp.ndarray): The input data for the batch.
                time (jnp.ndarray): The time tensor.
                psm (jnp.ndarray): Power special matrix.
                shape (Tuple[int, ...]): The shape of the input tensor.

            Returns:
                jnp.ndarray: The model's output.
            """
            b, g, c = batch_corrupted.shape
            t_aux = jnp.reshape(time, (b, 1, 1))
            t_aux = jnp.broadcast_to(t_aux, (b, g, 1))
            batch_input = batch_input.at[:, :, -1:].set(t_aux)
            batch_input = batch_input.at[:, :, len(shape) : len(shape) + c].set(batch_corrupted)
            model_output = self.apply(params, batch_input, train=False)
            return model_output

        return predict
