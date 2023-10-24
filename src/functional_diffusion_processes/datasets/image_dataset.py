import abc
import logging
from typing import Any, Callable, Dict

import tensorflow as tf
from omegaconf import DictConfig

from src.functional_diffusion_processes.datasets import BaseDataset
from src.functional_diffusion_processes.utils.common import make_grid_image, process_images

pylogger = logging.getLogger(__name__)


class ImageDataset(BaseDataset, abc.ABC):
    """Base class for handling image datasets.

    Provides a structured way to load, preprocess, and post-process image data.
    This class can be extended to handle specific image datasets as required.

    Attributes:
        data_config (DictConfig): Configuration settings for loading the dataset.
        split (str): Specifies the dataset split to load ('train', 'val', 'test', etc.).
        evaluation (bool): Indicates if the dataset is used for evaluation.
    """

    def __init__(self, data_config: DictConfig, split: str, evaluation: bool = False) -> None:
        """Initializes the ImageDataset object with dataset configurations.

        Args:
            data_config (DictConfig): Configuration settings for loading the dataset.
            split (str): Specifies the dataset split to load ('train', 'val', 'test', etc.).
            evaluation (bool): Indicates if the dataset is used for evaluation.
        """
        super().__init__(data_config, split, evaluation)

    @staticmethod
    def _resize_op(image: Any, size: int) -> Any:
        """Resizes the input image to the specified size and normalizes its values to the range [0,1].

        Args:
            image (Any): A tensor representing the input image.
            size (int): The target size for each dimension of the output image.

        Returns:
            Any: A tensor representing the resized and normalized image.
        """
        # convert to range [0,1]
        pylogger.info("Converting image to range [0,1]...")
        image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)

        # resize to size
        pylogger.info("Resizing image to size {}...".format(size))

        image = tf.image.resize(images=image, size=[size, size])

        return image

    def preprocess_fn(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses the input data by resizing, possibly flipping, and applying uniform dequantization.

        Args:
            d (Dict[str, Any]): A dictionary containing the input data with keys 'image' and optionally 'label'.

        Returns:
            Dict[str, Any]: A dictionary containing the preprocessed data, with keys 'data' and optionally 'label'.
        """
        image = self._resize_op(image=d["image"], size=self.data_config.image_width_size)

        pylogger.info("Preprocessing images for split {}...".format(self.split))

        if self.data_config.random_flip and not self.evaluation:
            pylogger.info("Applying random flips...")
            image = tf.image.random_flip_left_right(image=image, seed=self.data_config.seed)

        if self.data_config.uniform_dequantization:
            pylogger.info("Applying uniform dequantization...")
            image = (
                tf.random.uniform(shape=image.shape, dtype=tf.float32, seed=self.data_config.seed) + image * 255.0
            ) / 256.0

        image = tf.reshape(
            tensor=image,
            shape=(-1, self.data_config.output_size),
        )
        pylogger.info("Image reshaped to shape {}...".format(image.shape))

        return dict(data=image, label=d.get("label", None))

    def postprocess_fn(self, batch_data: Any, inverse_scaler: Callable) -> Any:
        """Post-processes the output data by reverting the preprocessing steps.

        Args:
            batch_data (Any): A batch of data to postprocess.
            inverse_scaler (Callable): A function to invert the scaling applied to the data.

        Returns:
            Any: A batch of postprocessed data, arranged in a grid for visualization.
        """
        batch_post_processed = make_grid_image(
            ndarray=process_images(images=batch_data),
            inverse_scaler=inverse_scaler,
        )
        return batch_post_processed
