import abc
from typing import Any, Callable

from ..sdetools import SDE


class Loss(abc.ABC):
    """Abstract class representing a loss function.

    Provides a framework for defining custom loss functions by enforcing the implementation
    of `construct_loss_fn` method in any derived classes. This class holds a reference to
    a stochastic differential equation (SDE) object which is used to calculate the weight factor for the loss.

    Attributes:
        sde (SDE): The stochastic differential equation instance associated with this loss.
    """

    def __init__(self, sde: SDE) -> None:
        """Initializes the Loss instance with a given SDE.

        Args:
            sde (SDE): An SDE instance which might be used in the loss computation.
        """
        self.sde = sde

    def construct_loss_fn(self, model: Any) -> Callable:
        """Abstract method to construct a loss function for a given model.

        This method should be implemented by any derived class to define the loss
        computation specific to the type of loss being implemented.

        Args:
            model (Any): The model for which to construct the loss function.

        Returns:
            Callable: A callable representing the constructed loss function.

        Raises:
            NotImplementedError: If the method is not implemented by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the construct_loss_fn method.")
