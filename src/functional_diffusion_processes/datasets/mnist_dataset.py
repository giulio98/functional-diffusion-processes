import abc
import logging
from typing import Any

import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig

from .image_dataset import ImageDataset

pylogger = logging.getLogger(__name__)


class MNISTDataset(ImageDataset, abc.ABC):
    """Class for handling the MNIST dataset.

    Inherits from the ImageDataset class and provides specific implementations
    for loading and preprocessing the MNIST dataset.

    Attributes:
        dataset_builder (tfds.core.DatasetBuilder): Builder for the MNIST dataset.
    """

    def __init__(self, data_config: DictConfig, split: str, evaluation: bool = False) -> None:
        """Initializes the MNISTDataset object with dataset configurations.

        Args:
            data_config (DictConfig): Configuration settings for loading the dataset.
            split (str): Specifies the dataset split to load ('train', 'val', 'test', etc.).
            evaluation (bool): Indicates if the dataset is used for evaluation.
        """
        super().__init__(data_config, split, evaluation)
        self.dataset_builder = tfds.builder(name="mnist", data_dir=self.data_config.data_dir)

    def _resize_op(self, image: Any, size: int) -> Any:
        """Resizes the input image to the specified size and normalizes its values to the range [0,1].

        Additionally, binarizes the image if it is to be used as a mask for the inpainting task
        as per the dataset configuration.

        Args:
            image (Any): A tensor representing the input image.
            size (int): The target size for each dimension of the output image.

        Returns:
            Any: A tensor representing the resized, normalized, and possibly binarized image.
        """
        # convert to range [0,1]
        pylogger.info("Converting image to range [0,1]...")
        image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)

        # resize to size
        pylogger.info("Resizing image to size {}...".format(size))

        image = tf.image.resize(images=image, size=[size, size])

        # binarize the image
        if self.data_config.is_mask:
            pylogger.info("Binarizing the image...")
            image = tf.where(image < 0.5, 1, 0)

        return image
