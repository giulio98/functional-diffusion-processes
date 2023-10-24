import abc
import logging
from functools import partial
from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict, unfreeze
from omegaconf import DictConfig

from ..utils.common import clip_learning_rates, make_coordinates, merge_learning_rates, separate_learning_rates

Params = FrozenDict[str, Any]
T = TypeVar("T")


pylogger = logging.getLogger(__name__)


@partial(jax.vmap, in_axes=0)
def mean_square_error(y_corrupted: jnp.ndarray, y_reconstructed: jnp.ndarray, y_psm: jnp.ndarray) -> jnp.ndarray:
    """Calculate the mean squared error between the predicted and actual values of a batch.

    Args:
        y_corrupted (jnp.ndarray): The actual y perturbed, with shape (output_size,).
        y_reconstructed (jnp.ndarray): The predicted y, with shape (output_size,).
        y_psm (jnp.ndarray): The Power special matrix, with shape (1,).

    Returns:
        jnp.ndarray: The mean squared error for each y, with shape (1,).
    """
    return jnp.sum(jnp.square(jnp.abs(y_corrupted * y_psm - y_reconstructed * y_psm)))


class BaseMAML(nn.Module, abc.ABC):
    """Abstract model class for implementing Model-Agnostic Meta-Learning (MAML).

    The Model-Agnostic Meta-Learning (MAML) algorithm is designed to train models
    in a manner that they can be fine-tuned for new tasks with a small number of examples.
    This implementation is based on the MAML algorithm introduced in the paper
    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
    (https://arxiv.org/abs/1703.03400).

    Attributes:
        model_config (DictConfig): Configuration dictionary for the model.
        optimizer_inner (optax.GradientTransformation): Inner optimizer configuration.
        inner_steps (int): Number of inner optimization steps.

    Methods:
        __call__(self, inputs: jnp.ndarray) -> jnp.ndarray: Implement the forward pass of the model.
        initialize_model(self, rng: jax.random.PRNGKey, batch_input: jnp.ndarray) -> FrozenDict[str, Mapping[str, Any]]: Initialize the model with dummy inputs.
        initialize_input(self, shape: Tuple[int, ...]) -> jnp.ndarray: Create input tensor for the model based on the specified shape.
        make_update_params_fn(self) -> Callable[..., Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]]: Create a function to update the model parameters.
        make_update_inner_fn(self, optimizer_inner: optax.GradientTransformation, n_steps: int) -> Callable[..., Tuple[jnp.ndarray, jnp.ndarray]]: Create a function to update model parameters for inner optimization.
        make_predict_fn(self) -> Callable[..., jnp.ndarray]: Creates a function for making predictions with the model.
    """

    model_config: DictConfig
    optimizer_inner: optax.GradientTransformation
    inner_steps: int

    @abc.abstractmethod
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Implement the forward pass of the model.

        Args:
            inputs (jnp.ndarray): Input tensor to the model.

        Returns:
            jnp.ndarray: Output tensor from the model.

        Raises:
            NotImplementedError: If this method is not overridden by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the __call__ method.")

    def initialize_model(self, rng: jax.random.PRNGKey, batch_input: jnp.ndarray) -> FrozenDict[str, Mapping[str, Any]]:
        """Initialize the model with dummy inputs.

        This method initializes the model parameters by passing a batch of dummy inputs
        through the model. This is a common practice to infer the dimensions of the model's
        parameters.

        Args:
            rng (jax.random.PRNGKey): A random key for generating initial model parameters.
            batch_input (jnp.ndarray): A batch of dummy inputs for initializing the model.

        Returns:
            FrozenDict[str, Mapping[str, Any]]: The initialized model parameters.
        """
        self.optimizer_inner = hydra.utils.instantiate(self.optimizer_inner)
        return self.init(rng, batch_input)

    def initialize_input(self, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Create input tensor for the model based on the specified shape.

        Args:
            shape (Tuple[int, ...]): Shape of the input tensor.

        Returns:
            jnp.ndarray: Initialized input tensor.
        """
        batch_size = shape[0]
        num_channels = shape[-1]
        grid_size = shape[1:-1]
        if not self.model_config.y_input:
            num_channels = None
        coordinates = make_coordinates(batch_size, grid_size, num_channels)
        return coordinates

    def make_update_params_fn(self) -> Callable[..., Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]]:
        """Create a function to update the model parameters.

        This method creates a function that performs the forward pass of the model
        and updates the model parameters.

        Returns:
            Callable[..., Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]]: Function to update model parameters.
        """
        update_inner_fn = self.make_update_inner_fn(
            optimizer_inner=self.optimizer_inner,
            n_steps=self.inner_steps,
        )

        def apply_forward(
            rng: jax.random.PRNGKey,
            params: Params,
            batch_input: jnp.ndarray,
            batch_corrupted: jnp.ndarray,
            psm: jnp.ndarray,
        ) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]:
            """Apply the (outer) forward pass and update the model parameters.

            Args:
                rng (jax.random.PRNGKey): Random key.
                params (Params): Initial model parameters.
                batch_input (jnp.ndarray): Input tensor to the model.
                batch_corrupted (jnp.ndarray): Corrupted version of the output tensor.
                psm (jnp.ndarray): Power special matrix.

            Returns:
                Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]: A tuple containing a new random key, the model output, and the inner loss.
            """
            params_adapted, loss_inner = update_inner_fn(params, batch_input, batch_corrupted, psm)
            model_output = jax.vmap(self.apply)(params_adapted, batch_input)

            return rng, model_output, loss_inner

        return apply_forward

    def make_update_inner_fn(
        self, optimizer_inner: optax.GradientTransformation, n_steps: int
    ) -> Callable[[Params, jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """Create a function to update model parameters for inner optimization.

        This method creates a function that performs the inner optimization updates
        during the meta-training phase, which is a key component of the MAML algorithm.

        Args:
            optimizer_inner (optax.GradientTransformation): The optimizer used for inner optimization.
            n_steps (int): The number of optimization steps.

        Returns:
            Callable[..., Tuple[jnp.ndarray, jnp.ndarray]]: Function to update model parameters for inner optimization.
        """

        @partial(jax.vmap, in_axes=0)
        @partial(jax.grad, has_aux=True)
        def loss_inner_fn(params_i: Params, batch_input: T, y_corrupted: T, psm: T) -> T:
            """Computes the loss for inner optimization.

            This inner method computes the loss for inner optimization by comparing
            the model's output against the corrupted batch using mean square error.
            The method is vectorized using JAX's vmap function for efficiency.

            Args:
                params_i (Params): Model parameters.
                batch_input (T): Input batch.
                y_corrupted (T): Corrupted batch.
                psm (T): Power special matrix.

            Returns:
                T: Loss value.
            """
            c = y_corrupted.shape[-1]
            model_output = self.apply(params_i, batch_input)
            if len(psm.shape) == 3:
                model_output_freq = jnp.fft.fft2(model_output.reshape(*psm.shape[:-1], c), norm="ortho", axes=(0, 1))
                y_corrupted_freq = jnp.fft.fft2(y_corrupted.reshape(*psm.shape[:-1], c), norm="ortho", axes=(0, 1))
            else:
                model_output_freq = jnp.fft.fft(model_output.reshape(*psm.shape[:-1], c), norm="ortho", axis=0)
                y_corrupted_freq = jnp.fft.fft(y_corrupted.reshape(*psm.shape[:-1], c), norm="ortho", axis=0)
            mse = mean_square_error(
                y_corrupted_freq.reshape(-1, c),
                model_output_freq.reshape(-1, c),
                psm.reshape(-1, 1),
            )
            loss: jnp.ndarray = jnp.mean(mse)

            return loss, loss

        def apply_inner_forward(
            params: Params, batch_input: jnp.ndarray, batch_corrupted: jnp.ndarray, psm: jnp.ndarray
        ):
            """Applies inner forward pass for updating model parameters.

            Args:
                params (Params): Model parameters.
                batch_input (jnp.ndarray): Input batch.
                batch_corrupted (jnp.ndarray): Corrupted batch.
                psm (jnp.ndarray): Power special matrix.

            Returns:
                Tuple[jnp.ndarray, jnp.ndarray]: Updated model parameters and inner loss.
            """

            def inner_opt_loop(
                carry: Tuple[Params, jnp.ndarray, int, Any, jnp.ndarray], _: None
            ) -> Tuple[Tuple[Params, jnp.ndarray, int, Any, jnp.ndarray], None]:
                """Inner optimization loop for updating model parameters.

                Args:
                    carry (Tuple[Params, jnp.ndarray, int, optax.OptState, jnp.ndarray]): Tuple containing model parameters,
                        loss vector, iteration index, optimizer state, and corrupted batch.
                    _ (None): A throwaway variable as no second argument is used in this function.

                Returns:
                    Tuple[Params, jnp.ndarray, int, optax.OptState, jnp.ndarray]: Updated tuple with new model parameters,
                        updated loss vector, incremented iteration index, updated optimizer state, and corrupted batch.
                """
                params_i, loss_inner_vec, it, opt_inner_state_params, batch_corrupted_i = carry

                grad_params, (loss) = loss_inner_fn(params_i, batch_input, batch_corrupted_i, psm)
                loss_inner_vec = loss_inner_vec.at[it].set(jnp.mean(loss))

                if self.model_config.use_dense_lr:
                    # separate learning rates from grad_params
                    grad_params_true, _ = separate_learning_rates(unfreeze(grad_params))

                    # separate learning rates from params_i
                    params_i_true, learning_rates = separate_learning_rates(unfreeze(params_i))

                    # calculate updates using meta-sgd
                    updates_params = jax.tree_map(
                        lambda g, lr: -jnp.clip(lr, 0, 1) * g,
                        grad_params_true,
                        learning_rates,
                    )

                    # merge updates_params and learning_rates
                    merged_updates = merge_learning_rates(unfreeze(updates_params), unfreeze(learning_rates))
                    params_i1 = optax.apply_updates(params_i, merged_updates)

                    # after update of params clip learning rates to [0, 1]
                    params_i1 = clip_learning_rates(params_i1)
                else:
                    updates_params, opt_state = optimizer_inner.update(grad_params, opt_inner_state_params, params_i)
                    params_i1 = optax.apply_updates(params_i, updates_params)
                return (
                    params_i1,
                    loss_inner_vec,
                    it + 1,
                    opt_inner_state_params,
                    batch_corrupted,
                ), _

            base_params = jax.tree_map(
                lambda base_param: jnp.stack(
                    [
                        base_param,
                    ]
                    * batch_input.shape[0],
                    axis=0,
                ),
                params,
            )
            loss_inner = jnp.zeros((n_steps,))
            i = 0
            initial_state = (
                base_params,
                loss_inner,
                i,
                optimizer_inner.init(base_params),
                batch_corrupted,
            )
            params_adapted, loss_inner, *_ = jax.lax.scan(inner_opt_loop, initial_state, xs=None, length=n_steps)[0]
            return params_adapted, loss_inner

        return apply_inner_forward

    def make_predict_fn(
        self,
    ) -> Callable[
        [Params, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray
    ]:
        """Creates a function for making predictions with the model.

        This method creates a function that can be used to make predictions with the model.

        Returns:
            Callable[..., jnp.ndarray]: A function for making predictions with the model.
        """
        update_inner_fn = self.make_update_inner_fn(
            optimizer_inner=self.optimizer_inner,
            n_steps=self.inner_steps,
        )

        def predict(
            params: Params,
            batch_corrupted: jnp.ndarray,
            batch_input: jnp.ndarray,
            time: jnp.ndarray,
            psm: jnp.ndarray,
            shape: jnp.ndarray,
        ) -> jnp.ndarray:
            """Make predictions using the model.

            Args:
                params (Params): Model parameters.
                batch_corrupted (jnp.ndarray): Corrupted version of the output tensor.
                batch_input (jnp.ndarray): Input tensor to the model.
                time (jnp.ndarray): Time tensor.
                psm (jnp.ndarray): Power special matrix.
                shape (jnp.ndarray): Shape of the input tensor.

            Returns:
                jnp.ndarray: Reconstructed output tensor.
            """
            b, g, c = batch_corrupted.shape
            t_aux = jnp.reshape(time, (b, 1, 1))
            t_aux = jnp.broadcast_to(t_aux, (b, g, 1)) * 2 - 1
            batch_input = batch_input.at[:, :, -1:].set(t_aux)
            if self.model_config.y_input:
                batch_input = batch_input.at[:, :, len(shape) : len(shape) + c].set(batch_corrupted)
            params_adapted, _ = update_inner_fn(params, batch_input, batch_corrupted, psm)
            batch_reconstructed = jax.vmap(self.apply)(params_adapted, batch_input)

            return batch_reconstructed

        return predict

    def make_super_resolution_fn(
        self,
    ) -> Callable[
        [Params, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray
    ]:
        """Creates a function for making super resolution output with the model.

        This method creates a function that can be used to make super resolution task with the model.

        Returns:
            Callable[..., jnp.ndarray]: A function for making super resolution output with the model.
        """
        update_inner_fn = self.make_update_inner_fn(
            optimizer_inner=self.optimizer_inner,
            n_steps=self.inner_steps,
        )

        def super_resolution_fn(
            params: Params,
            batch_corrupted: jnp.ndarray,
            batch_input: jnp.ndarray,
            time: jnp.ndarray,
            psm: jnp.ndarray,
            shape: jnp.ndarray,
            target_shape: Optional[jnp.ndarray] = None,
        ) -> jnp.ndarray:
            """Make last prediction for super resolution task using the model.

            Args:
                params (Params): Model parameters.
                batch_corrupted (jnp.ndarray): Corrupted version of the output tensor.
                batch_input (jnp.ndarray): Input tensor to the model.
                time (jnp.ndarray): Time tensor.
                psm (jnp.ndarray): Power special matrix.
                shape (jnp.ndarray): Shape of the input tensor.
                target_shape (Optional[jnp.ndarray]): Target shape of the output tensor.

            Returns:
                jnp.ndarray: Reconstructed output tensor at super-resolution.
            """
            b, g, c = batch_corrupted.shape
            t_aux = jnp.reshape(time, (b, 1, 1))
            t_aux = jnp.broadcast_to(t_aux, (b, g, 1)) * 2 - 1
            batch_input = batch_input.at[:, :, -1:].set(t_aux)
            if self.model_config.y_input:
                batch_input = batch_input.at[:, :, len(shape) : len(shape) + c].set(batch_corrupted)
            params_adapted, _ = update_inner_fn(params, batch_input, batch_corrupted, psm)
            if self.model_config.y_input:
                batch_reconstructed = jax.vmap(self.apply)(params_adapted, batch_input)
                batch_input = batch_input.at[:, :, len(shape) : len(shape) + c].set(batch_reconstructed)
            batch_input = batch_input.reshape((b, *shape, -1))

            new_h, new_w = target_shape

            batch_input_new = jax.image.resize(batch_input, (b, new_h, new_w, batch_input.shape[-1]), method="bilinear")
            batch_input_new = batch_input_new.reshape((b, new_h * new_w, -1))
            batch_reconstructed = jax.vmap(self.apply)(params_adapted, batch_input_new)

            return batch_reconstructed

        return super_resolution_fn
