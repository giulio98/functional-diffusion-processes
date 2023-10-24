# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for computing FID/Inception scores."""
import abc
import gc
import logging
import os
from typing import Callable, Dict, List, Optional, Union

import jax
import numpy as np
import six
import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow.python.keras.models import Model

from ..datasets import BaseDataset, ImageDataset
from .metrics_utils import get_inception_model

INCEPTION_OUTPUT = "logits"
INCEPTION_FINAL_POOL = "pool_3"
_DEFAULT_DTYPES = {
    INCEPTION_OUTPUT: tf.float32,
    INCEPTION_FINAL_POOL: tf.float32,
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299

pylogger = logging.getLogger(__name__)


class InceptionFeatureExtractor(abc.ABC):
    """Abstract base class for extracting features using Inception models.

    Provides methods to extract features from datasets using specified Inception models. It supports both
    Inception v3 and other versions, as well as distributed feature extraction on multiple devices.

    Attributes:
        inception_model (tf.Module or Model): The loaded Inception model.
        inception_v3 (bool): Indicates if Inception v3 model is used.
        model_name (str): Name of the Inception model.
    """

    def __init__(self, model_name: str, inception_v3: bool = False) -> None:
        """Initializes the feature extractor with specified Inception model.

        Args:
            model_name (str): Name of the Inception model.
            inception_v3 (bool, optional): Indicator to use Inception v3. Defaults to False.
        """
        self.inception_model = get_inception_model(inception_model_name=model_name, inception_v3=inception_v3)
        self.inception_v3 = inception_v3
        self.model_name = model_name

    @staticmethod
    def classifier_fn_from_tfhub(
        output_fields: Union[str, List[str], None],
        inception_model: Union[tf.Module, Model],
        return_tensor: bool = False,
    ) -> Callable:
        """Creates a classifier function for feature extraction.

        Args:
            output_fields (Union[str, List[str], None]): Fields to select from the model outputs.
            inception_model (Union[tf.Module, Model]): Loaded Inception model.
            return_tensor (bool, optional): Return a single tensor instead of dictionary. Defaults to False.

        Returns:
            Callable: Function to extract features from input images.
        """
        if isinstance(output_fields, six.string_types):
            output_fields = [output_fields]

        def _classifier_fn(images):
            output = inception_model(images)
            if output_fields is not None:
                output = {x: output[x] for x in output_fields}
            if return_tensor:
                assert len(output) == 1
                output = list(output.values())[0]
            return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

        return _classifier_fn

    @tf.function
    def _run_inception_jit(self, inputs: tf.Tensor, num_batches: int = 1) -> Dict[str, tf.Tensor]:
        """Runs Inception model on input tensors.

        Args:
            inputs (tf.Tensor): Input images tensor.
            num_batches (int, optional): Number of batches to process. Defaults to 1.

        Returns:
            Dict[str, tf.Tensor]: Dictionary of output tensors from the Inception model.
        """
        if not self.inception_v3:
            inputs = (tf.cast(x=inputs, dtype=tf.float32) - 127.5) / 127.5
        else:
            inputs = tf.cast(x=inputs, dtype=tf.float32) / 255.0

        return tfgan.eval.run_classifier_fn(
            input_tensor=inputs,
            num_batches=num_batches,
            classifier_fn=self.classifier_fn_from_tfhub(None, self.inception_model),
            dtypes=_DEFAULT_DTYPES,
        )

    @tf.function
    def _run_inception_distributed(self, input_tensor: tf.Tensor, num_batches: int = 1) -> Dict[str, tf.Tensor]:
        """Runs Inception model on input tensors in a distributed manner.

        Args:
            input_tensor (tf.Tensor): Input images tensor.
            num_batches (int, optional): Number of batches to process. Defaults to 1.

        Returns:
            Dict[str, tf.Tensor]: Dictionary of output tensors from the Inception model.
        """
        num_tpus = jax.local_device_count()
        input_tensors = tf.split(input_tensor, num_tpus, axis=0)
        pool3 = []
        logits = [] if not self.inception_v3 else None
        device_format = "/TPU:{}" if "TPU" in str(jax.devices()[0]) else "/GPU:{}"
        for i, tensor in enumerate(input_tensors):
            with tf.device(device_format.format(i)):
                tensor_on_device = tf.identity(tensor)
                res = self._run_inception_jit(
                    inputs=tensor_on_device,
                    num_batches=num_batches,
                )
                if not self.inception_v3:
                    pool3.append(res["pool_3"])
                    logits.append(res["logits"])
                else:
                    pool3.append(res)

            with tf.device("/CPU"):
                return {
                    "pool_3": tf.concat(values=pool3, axis=0),
                    "logits": tf.concat(values=logits, axis=0) if not self.inception_v3 else None,
                }

    # noinspection PyProtectedMember
    def extract_features(
        self,
        dataset: Union[BaseDataset, np.ndarray],
        num_batches: int = 1,
        save_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> np.ndarray:
        """Extracts features from the specified dataset or array using the Inception model.

        Args:
            dataset (Union[BaseDataset, np.ndarray]): Dataset or array of data for feature extraction.
            num_batches (int, optional): Number of batches to process. Defaults to 1.
            save_path (Optional[str], optional): Path to save extracted features. Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to None.

        Returns:
            np.ndarray: Array of extracted features.
        """
        all_pools = []
        if isinstance(dataset, ImageDataset):
            pylogger.info("Extracting features from dataset...")
            ds_iter = iter(dataset)
            batch_id = -1
            # for batch_id in range(len(dataset)):
            while True:
                try:
                    batch = next(ds_iter)
                    batch_id = batch_id + 1
                except StopIteration:
                    break
                if jax.host_id() == 0:
                    pylogger.info("Making FID stats -- step %d" % batch_id)
                batch_ = jax.tree_map(f=lambda x: x._numpy(), tree=batch)
                batch_ = (
                    np.clip(batch_["data"] * 255.0, 0, 255)
                    .astype(np.uint8)
                    .reshape(
                        (
                            -1,
                            dataset.data_config.image_width_size,
                            dataset.data_config.image_height_size,
                            dataset.data_config.output_size,
                        )
                    )
                )
                for i in range(batch_.shape[0]):
                    clean_path = os.path.join(save_path, f"{dataset_name.lower()}_clean/real_data_{batch_id}_{i}.npy")
                    np.save(clean_path, batch_[i])
                # Force garbage collection before calling TensorFlow code for Inception network
                gc.collect()
                latents = self._run_inception_distributed(
                    input_tensor=batch_,
                    num_batches=num_batches,
                )
                all_pools.append(latents["pool_3"])

                # Force garbage collection again before returning to JAX code
                gc.collect()
            all_pools = np.concatenate(all_pools, axis=0)  # Combine into one
            return all_pools
        else:
            latents = self._run_inception_distributed(
                input_tensor=dataset,
                num_batches=num_batches,
            )
            return latents
