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
import gc
import io
import logging
import os
import time
from typing import Any, Tuple

import jax
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
from omegaconf import DictConfig

from ..datasets.base_dataset import BaseDataset
from ..metrics.metrics_utils import load_dataset_stats
from .feature_extractor import InceptionFeatureExtractor

pylogger = logging.getLogger(__name__)


class FIDMetric:
    """Class for computing the Frechet Inception Distance (FID) metric.

    This class facilitates the computation of the FID metric, which measures the similarity between two distributions of images.
    It precomputes features for the real dataset using a specified Inception feature extractor and provides methods to compute
    and store features for generated images, and to compute the FID and Inception Score (IS).

    Attributes:
        metric_config (DictConfig): Configuration parameters for the FID metric.
        feature_extractor (InceptionFeatureExtractor): Inception feature extractor for computing the FID metric.
        dataset (BaseDataset): Dataset object providing real samples for FID computation.
        generated_pools (list): List to store features of generated images.
        generated_logits (list): List to store logits of generated images.
        real_features (dict): Dictionary to store precomputed features of real dataset.
    """

    def __init__(
        self,
        metric_config: DictConfig,
        feature_extractor: InceptionFeatureExtractor,
        dataset: BaseDataset,
    ) -> None:
        """Initializes the FIDMetric class with specified configurations, feature extractor, and dataset.

        Args:
            metric_config (DictConfig): Configuration parameters for the FID metric.
            feature_extractor (InceptionFeatureExtractor): Inception feature extractor for computing the FID metric.
            dataset (BaseDataset): Dataset object providing real samples for FID computation.
        """
        self.metric_config = metric_config
        self.feature_extractor = feature_extractor
        self.dataset = dataset
        self.generated_pools = []
        self.generated_logits = []
        try:
            self.real_features = load_dataset_stats(
                save_path=metric_config.real_features_path,
                dataset_name=metric_config.dataset_name,
            )
        except FileNotFoundError:
            self._precompute_features(
                dataset_name=metric_config.dataset_name,
                save_path=metric_config.real_features_path,
            )
            self.real_features = load_dataset_stats(
                save_path=metric_config.real_features_path,
                dataset_name=metric_config.dataset_name,
            )

    def _precompute_features(self, dataset_name: str, save_path: str) -> None:
        """Precomputes and saves features for the real dataset.

        Args:
            dataset_name (str): Name of the dataset.
            save_path (str): Path where the computed features will be saved.
        """
        tf.io.gfile.makedirs(path=save_path)

        tf.io.gfile.makedirs(os.path.join(save_path, f"{dataset_name.lower()}_clean"))

        # Use the feature extractor to compute features for the real dataset
        all_pools = self.feature_extractor.extract_features(
            dataset=self.dataset, save_path=save_path, dataset_name=dataset_name
        )

        # Save latent represents of the Inception network to disk or Google Cloud Storage
        filename = f"{dataset_name.lower()}_stats.npz"

        if jax.host_id() == 0:
            pylogger.info("Saving real dataset stats to: %s" % os.path.join(save_path, filename))

        with tf.io.gfile.GFile(os.path.join(save_path, filename), "wb") as f_out:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, pool_3=all_pools)
            f_out.write(io_buffer.getvalue())

    def compute_fid(self, eval_dir, num_sampling_round) -> Tuple[float, float]:
        """Computes the FID and Inception Score (IS) for the generated and real images.

        Args:
            eval_dir (str): Directory path for evaluation.
            num_sampling_round (int): Number of sampling rounds.

        Returns:
            Tuple[float, float]: A tuple containing the FID and Inception Score.
        """
        real_pools = self.real_features["pool_3"]
        if not self.feature_extractor.inception_v3 and not self.feature_extractor.inception_v3 == "lenet":
            if len(self.generated_logits) == 0 or len(self.generated_pools) == 0:
                if jax.host_id() == 0:
                    # Load all statistics that have been previously computed and saved for each host
                    for host in range(jax.host_count()):
                        stats = tf.io.gfile.glob(os.path.join(eval_dir, "statistics_*.npz"))
                        wait_message = False
                        while len(stats) < num_sampling_round:
                            if not wait_message:
                                print("Waiting for statistics on host %d" % (host,))
                                wait_message = True
                            stats = tf.io.gfile.glob(os.path.join(eval_dir, "statistics_*.npz"))
                            time.sleep(10)

                        for stat_file in stats:
                            with tf.io.gfile.GFile(stat_file, "rb") as fin:
                                stat = np.load(fin)

                                self.generated_pools.append(stat["pool_3"])
                                self.generated_logits.append(stat["logits"])

            all_logits = np.concatenate(self.generated_logits, axis=0)[: self.metric_config.num_samples]
            inception_score = tfgan.eval.classifier_score_from_logits(logits=all_logits)
        else:
            inception_score = -1

        all_pools = np.concatenate(self.generated_pools, axis=0)[: self.metric_config.num_samples]

        fid = tfgan.eval.frechet_classifier_distance_from_activations(activations1=real_pools, activations2=all_pools)

        return fid, inception_score

    def compute_and_store_generated_features(self, images: Any, sample_dir: str, round_num: int) -> None:
        """Computes features for the generated images and stores them in a specified directory.

        Args:
            images (Any): Tensor representing the generated images.
            sample_dir (str): Directory where the features will be stored.
            round_num (int): Round number in the training process.
        """
        latents = self.feature_extractor.extract_features(images)

        self.generated_pools.append(latents["pool_3"])

        gc.collect()

        if self.feature_extractor.model_name == "inception" or self.feature_extractor.inception_v3:
            self.generated_logits.append(latents["logits"])
            with tf.io.gfile.GFile(os.path.join(sample_dir, f"statistics_{round_num}.npz"), "wb") as f_out:
                io_buffer = io.BytesIO()
                np.savez_compressed(
                    io_buffer,
                    pool_3=latents["pool_3"],
                    logits=latents["logits"],
                )

                f_out.write(io_buffer.getvalue())

        elif self.feature_extractor.model_name == "lenet":
            with tf.io.gfile.GFile(os.path.join(sample_dir, f"statistics_{round_num}.npz"), "wb") as f_out:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, pool_3=latents["pool_3"])
                f_out.write(io_buffer.getvalue())
