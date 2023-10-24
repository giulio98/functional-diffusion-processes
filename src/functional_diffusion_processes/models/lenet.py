from typing import Tuple

from tensorflow.python.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.models import Sequential


class LeNet5(Sequential):
    """Implementation of LeNet-5 architecture for image classification tasks introduced in the paper "Gradient-Based Learning Applied to Document Recognition" by LeCun et al.

    This particular implementation is used for calculating the Frechet Inception Distance (FID)
    for the MNIST dataset.

    Args:
        input_shape (Tuple[int, int, int]): Dimensions of the input image. Conventionally (height, width, channels).
        nb_classes (int): The number of target classes. For MNIST, this is 10 (one for each digit).

    Methods:
        __init__: Initializes the LeNet-5 model with the specified input shape and number of classes.
    """

    def __init__(self, input_shape: Tuple[int, int, int], nb_classes: int) -> None:
        """Initialize the LeNet5 model with given input shape and number of classes.

        Args:
            input_shape (Tuple[int, int, int]): The shape of the input images (height, width, channels).
            nb_classes (int): The number of output classes.
        """
        super().__init__()

        self.add(
            Conv2D(
                filters=6,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation="tanh",
                input_shape=input_shape,
                padding="same",
            )
        )
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
        self.add(
            Conv2D(
                filters=16,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation="tanh",
                padding="valid",
            )
        )
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
        self.add(Flatten())
        self.add(Dense(units=120, activation="tanh"))
        self.add(Dense(units=84, activation="tanh"))
        self.add(Dense(units=nb_classes, activation="softmax"))

        self.compile(optimizer="adam", loss=categorical_crossentropy, metrics=["accuracy"])
