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
import os
from typing import Dict, Union

import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from tensorflow.python.keras.models import Model

from ..models.lenet import LeNet5

INCEPTION_TFHUB = "https://tfhub.dev/tensorflow/tfgan/eval/inception/1"


def load_dataset_stats(save_path: str, dataset_name: str) -> Dict[str, np.ndarray]:
    """Load the pre-computed dataset statistics.

    This function loads the pre-computed statistics of a dataset from a specified file path.
    These statistics include feature representations computed and saved earlier,
    which are required for the computation of the Frechet Inception Distance (FID).

    Args:
        save_path (str): A string representing the path where the dataset statistics are stored.
        dataset_name (str): A string representing the name of the dataset for which to load statistics.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the dataset statistics,
            keyed by the type of statistic with values as numpy arrays.

    Raises:
        FileNotFoundError: If the specified file containing the dataset statistics does not exist.
    """
    filename = f"{dataset_name.lower()}_stats.npz"
    filename = os.path.join(save_path, filename)

    if not tf.io.gfile.exists(filename):
        raise FileNotFoundError(f"No such file or directory: '{filename}'")

    with tf.io.gfile.GFile(filename, "rb") as fin:
        stats = np.load(fin)
    return stats


def get_lenet_model() -> Model:
    """Load the pre-trained LeNet5 model and create an activation model.

    This function loads a pre-trained LeNet5 model from the disk, and constructs an activation model
    that outputs the activations of the seventh layer of the LeNet5 model, which can be used for
    feature extraction.

    Returns:
        Model: The activation model for the LeNet5 model.

    Raises:
        FileNotFoundError: If the model weights file cannot be found.
        OSError: If there is an error loading the model weights.
    """
    try:
        model = LeNet5((32, 32, 1), 10)
        model.load_weights("./models/lenet5/model-best.h5")
    except FileNotFoundError:
        raise FileNotFoundError("Model weights file not found.")
    except Exception as e:
        raise OSError(f"Error loading model weights: {e}")
    # Create a new model that outputs the activations of the seventh layer
    activation_model = Model(inputs=model.input, outputs=model.layers[6].output)

    return activation_model


def get_inception_model(inception_model_name: str, inception_v3: bool = None) -> Union[tf.Module, Model]:
    """Load the specified Inception model.

    This function loads an Inception model as specified by the user. If the user specifies the LeNet model,
    it calls the `get_lenet_model` function to get the activation model of a pre-trained LeNet5 model.
    If Inception v3 is specified, it loads the Inception v3 model, otherwise, it loads the default Inception model from TF-Hub.

    Args:
        inception_model_name (str): A string representing the name of the Inception model to load.
        inception_v3 (Optional[bool]): If True, load the Inception v3 model. Defaults to None.

    Returns:
        Union[tf.Module, Model]: The loaded Inception model.

    Raises:
        Exception: If there is an error loading the model.
    """
    try:
        if inception_model_name == "lenet":
            return get_lenet_model()
        else:
            if inception_v3:
                return tfhub.load("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4")
            else:
                return tfhub.load(INCEPTION_TFHUB)
    except Exception as e:
        raise Exception(f"Error loading the model: {e}")
