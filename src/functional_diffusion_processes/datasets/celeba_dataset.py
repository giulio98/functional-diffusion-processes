import abc
import logging
from typing import Any

import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig

from .dataset_utils import central_crop, resize_small
from .image_dataset import ImageDataset

pylogger = logging.getLogger(__name__)


class CELEBADataset(ImageDataset, abc.ABC):
    """CelebA dataset class for loading and preprocessing the CelebA dataset.

    Inherits from the ImageDataset class and provides specific implementations for loading
    and preprocessing the CelebA dataset.

    Attributes:
        dataset_builder (tfds.core.DatasetBuilder): Builder for the CelebA dataset.
    """

    def __init__(self, data_config: DictConfig, split: str, evaluation: bool = False) -> None:
        """Initialize the CELEBADataset class.

        Initializes the dataset class by calling the super class constructor and sets up the dataset builder
        for the CelebA dataset.

        Args:
            data_config (DictConfig): Configuration parameters for the dataset.
            split (str): Specifies which split of the dataset to load, e.g., 'train', 'validation', or 'test'.
            evaluation (bool): Indicates whether the dataset is for evaluation purposes.

        """
        super().__init__(data_config, split, evaluation)
        self.dataset_builder = tfds.builder(name="celeb_a", data_dir=self.data_config.data_dir)

    def _resize_op(self, image: Any, size: int) -> Any:
        """Resize and crop an image to a specified size.

        This method first normalizes the pixel values of the input image to the range [0, 1],
        then applies a central crop to obtain a square image of size 140.
        Finally, it resizes the cropped image to the specified resolution.

        Args:
            image (Any): A tensor representing the input image.
            size (int): The resolution to resize the image to.

        Returns:
            Any: A tensor representing the resized and cropped image.
        """
        pylogger.info("Converting image to range [0,1]...")
        image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
        pylogger.info("Resizing and cropping image to size {}...".format(size))
        image = central_crop(image=image, size=140)
        image = resize_small(image=image, resolution=size)

        return image
