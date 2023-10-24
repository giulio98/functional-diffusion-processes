from typing import Any

import jax
from flax.training import train_state


class TrainState(train_state.TrainState):
    """The training state for the model."""

    opt_state_params: Any
    ema_params: Any
    rng: jax.random.PRNGKey
