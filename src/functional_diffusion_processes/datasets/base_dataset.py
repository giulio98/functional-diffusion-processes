import abc
import logging
from typing import Any, Callable, Dict, Iterator

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig

pylogger = logging.getLogger(__name__)


class BaseDataset(abc.ABC):
    """Abstract base class for defining datasets.

    Provides a template for loading, preprocessing, and iterating over datasets.
    It encapsulates common dataset configurations and operations while allowing for dataset-specific
    preprocessing and post-processing through abstract methods.

    Attributes:
        dataset_builder: A builder object for loading the dataset.
        data_config (DictConfig): Configuration parameters for the dataset.
        split (str): Specifies which split of the dataset to load, e.g., 'train', 'validation', or 'test'.
        evaluation (bool): Indicates whether the dataset is for evaluation purposes.
        dataset_options: Options for configuring the dataset pipeline.
    """

    def __init__(self, data_config: DictConfig, split: str, evaluation: bool = False) -> None:
        """Abstract base class for defining datasets.

        This class provides a skeleton for defining datasets, with abstract methods for
        preprocessing data, generating batches of data, and resizing images. Subclasses
        must implement these methods to define their specific datasets.

        Args:
            data_config (DictConfig): A dictionary-like object containing the configuration for
                loading the dataset.

            split (str): A string specifying which split of the dataset to load.

            evaluation (bool): A boolean specifying whether the dataset is for evaluation purposes.
        """
        self.dataset_builder = None
        self.data_config = data_config
        self.split = split
        self.evaluation = evaluation
        self.dataset_options = tf.data.Options()
        self.dataset_options.experimental_optimization.map_parallelization = True
        self.dataset_options.experimental_threading.private_threadpool_size = 48
        self.dataset_options.experimental_threading.max_intra_op_parallelism = 1

    @abc.abstractmethod
    def preprocess_fn(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract method for preprocessing input data.

        Subclasses should override this method to implement dataset-specific preprocessing.

        Args:
            d (Dict[str, Any]): A dictionary containing the input data.

        Returns:
            Dict[str, Any]: A dictionary containing the preprocessed data.
        """
        raise NotImplementedError("Subclasses must implement preprocess_fn method.")

    @abc.abstractmethod
    def postprocess_fn(self, batch_data: Any, inverse_scaler: Callable) -> Any:
        """Abstract method for postprocessing output data.

        Subclasses should override this method to implement dataset-specific post-processing.

        Args:
            batch_data (Any): A batch of data to postprocess.
            inverse_scaler (Callable): A function to inverse the scaling of the data.

        Returns:
            Any: A dictionary containing the postprocessed data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the postprocess_fn method.")

    def _generator(self) -> Iterator[Any]:
        """Generate batches of preprocessed data.

        Loads the dataset, shuffles the data, applies preprocessing, and batches the data.
        Subclasses might override this method to implement dataset-specific batching logic.

        Returns:
            Iterator[Any]: An iterator that generates batches of preprocessed data.
        """
        # load the dataset
        if isinstance(self.dataset_builder, tfds.core.DatasetBuilder):
            read_config = tfds.ReadConfig(options=self.dataset_options)
            if self.data_config.download:
                self.dataset_builder.download_and_prepare()
            ds = self.dataset_builder.as_dataset(
                split=self.split,
                shuffle_files=False,
                read_config=read_config,
                as_supervised=False,
            )
        else:
            ds = self.dataset_builder.with_options(options=self.dataset_options)

        ds = ds.shuffle(buffer_size=10000, seed=self.data_config.seed)

        # apply the preprocessing function to each element in the dataset
        ds = ds.map(map_func=self.preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # determine the batch size per device
        ds = ds.batch(batch_size=self.data_config.batch_size, drop_remainder=True)
        ds = ds.batch(batch_size=jax.device_count(), drop_remainder=True)

        ds = ds.repeat(count=100000 if not self.evaluation else 1)

        return iter(ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator that generates batches of preprocessed data.

        Calls the `_generator` method to obtain an iterator for generating preprocessed data batches.

        Returns:
            Iterator[Any]: An iterator that generates batches of preprocessed data.
        """
        return self._generator()

    def __len__(self) -> int:
        """Return the number of examples in the dataset.

        Obtains the total number of examples in the specified dataset split from the dataset builder's info attribute.

        Returns:
            int: The number of examples in the dataset.
        """
        return self.dataset_builder.info.splits[self.split].num_examples
