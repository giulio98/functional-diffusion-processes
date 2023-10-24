import abc
import logging
from typing import Any, Callable, Dict

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from pydub import AudioSegment, effects

from src.functional_diffusion_processes.datasets import BaseDataset

pylogger = logging.getLogger(__name__)


class AudioDataset(BaseDataset, abc.ABC):
    """Base class for defining audio datasets.

    This class serves as the foundation for defining datasets containing audio data.
    It includes methods for preprocessing, resizing, and normalizing audio data.
    Subclasses may override these methods to implement dataset-specific processing and resizing logic.
    """

    def __init__(self, data_config: DictConfig, split: str, evaluation: bool = False) -> None:
        """Initialize an AudioDataset instance.

        Args:
            data_config (DictConfig): Configuration for loading the dataset, including paths, audio properties, etc.
            split (str): Specifies which split of the dataset to load (e.g., 'train', 'validation', 'test').
            evaluation (bool, optional): Indicates whether the dataset is for evaluation purposes. Defaults to False.
        """
        super().__init__(data_config, split, evaluation)

    @staticmethod
    def normalize_audio(audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize the amplitude of the audio data to a standard range.

        This method utilizes PyDub's effects module to perform audio normalization.

        Args:
            audio_np (np.ndarray): Audio data represented as a NumPy array.
            sample_rate (int): The sample rate of the audio data.

        Returns:
            np.ndarray: The normalized audio data as a NumPy array.
        """
        # Convert numpy array to AudioSegment
        audio_segment = AudioSegment(audio_np.tobytes(), frame_rate=int(sample_rate), sample_width=2, channels=1)

        # Normalize with PyDub
        normalized_audio_segment = effects.normalize(audio_segment)

        # Convert back to numpy
        normalized_audio_np = np.array(normalized_audio_segment.get_array_of_samples())

        return normalized_audio_np

    def _resize_op(self, audio: tf.Tensor, size: int) -> tf.Tensor:
        """Resize the input audio to a specified size and normalize its amplitude to the range [0, 1].

        If the audio length is less than the specified size, zero padding is applied to reach the desired size.
        If the audio length is greater, it is truncated to the specified size.

        Args:
            audio (tf.Tensor): Input audio data as a TensorFlow tensor.
            size (int): The target size for the audio data.

        Returns:
            tf.Tensor: The resized and normalized audio data as a TensorFlow tensor.
        """
        # Normalize dataset
        pylogger.info("Normalizing audio...")
        audio = tf.cast(audio, dtype=tf.int16)
        # Calculate current length of the audio
        pylogger.info("Resizing audio to size {}...".format(size))
        audio_length = tf.shape(audio)[0]
        audio = tf.cond(
            audio_length < size,
            lambda: tf.concat([audio, tf.zeros(size - audio_length, dtype=audio.dtype)], axis=0),
            lambda: audio[:size],
        )
        audio_np = tf.numpy_function(self.normalize_audio, [audio, self.data_config.audio_sample_rate], tf.int16)
        audio = tf.convert_to_tensor(audio_np, dtype=tf.int16)
        audio = tf.cast(audio, dtype=tf.float32)
        pylogger.info("Converting audio to range [-1, 1]...")
        max_intensity = self.data_config.audio_max_intensity
        audio = audio / max_intensity
        return audio

    def preprocess_fn(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the input audio data.

        This method resizes the audio data to a specified size based on the dataset configuration and normalizes the amplitude to the range [-1, +1].

        Args:
            d (Dict[str, Any]): A dictionary containing the input audio data and any associated metadata.

        Returns:
            Dict[str, Any]: A dictionary containing the preprocessed audio data and any associated metadata.
        """
        pylogger.info("Preprocessing audios for split {}...".format(self.split))
        audio = self._resize_op(
            audio=d["audio"], size=int(self.data_config.audio_sample_rate * self.data_config.audio_max_duration)
        )
        audio = tf.reshape(
            tensor=audio,
            shape=(-1, self.data_config.output_size),
        )
        pylogger.info("Audio reshaped to shape {}...".format(audio.shape))
        return dict(data=audio, label=d.get("label", None))

    def postprocess_fn(self, batch_data: Any, inverse_scaler: Callable) -> Any:
        """Postprocess the output audio data.

        This method applies the inverse of the preprocessing steps to revert the audio data to its original form.

        Args:
            batch_data (Any): A batch of audio data to postprocess.
            inverse_scaler (Callable): A function that applies the inverse of the preprocessing steps.

        Returns:
            Any: A batch of postprocessed audio data.
        """
        max_intensity = self.data_config.audio_max_intensity
        batch_audio = inverse_scaler(batch_data)
        batch_audio = batch_audio * max_intensity
        batch_post_processed = tf.cast(batch_audio, tf.int16)
        audio_np = tf.numpy_function(
            self.normalize_audio, [batch_post_processed, self.data_config.audio_sample_rate], tf.int16
        )
        batch_post_processed = tf.convert_to_tensor(audio_np, dtype=tf.int16)
        return batch_post_processed
