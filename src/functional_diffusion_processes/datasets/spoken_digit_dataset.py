import abc
import logging

import tensorflow_datasets as tfds
from omegaconf import DictConfig

from .audio_dataset import AudioDataset

pylogger = logging.getLogger(__name__)


class SpokenDigitDataset(AudioDataset, abc.ABC):
    """Class for handling the Spoken Digit dataset.

    Inherits from the AudioDataset class and provides specific implementations
    for loading and preprocessing the Spoken Digit dataset.

    Attributes:
        dataset_builder (tfds.core.DatasetBuilder): Builder for the Spoken Digit dataset.
    """

    def __init__(self, data_config: DictConfig, split: str, evaluation: bool = False) -> None:
        """Initializes the SpokenDigitDataset object with dataset configurations.

        Args:
            data_config (DictConfig): Configuration settings for loading the dataset.
            split (str): Specifies the dataset split to load ('train', 'val', 'test', etc.).
            evaluation (bool): Indicates if the dataset is used for evaluation.

        Raises:
            ValueError: If the specified dataset split is not recognized.
        """
        super().__init__(data_config, split, evaluation)
        self.dataset_builder = tfds.builder(name="spoken_digit", data_dir=self.data_config.data_dir)
