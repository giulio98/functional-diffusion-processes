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

import abc
from typing import Any, Callable, Union

from omegaconf import DictConfig

from ..sdetools import SDE
from .correctors import Corrector
from .predictors import Predictor


class Sampler(abc.ABC):
    """Abstract base class for creating sampler objects.

    This class serves as a template for creating sampler objects which are
    designed to generate samples of a stochastic process governed by a
    specified stochastic differential equation (SDE). The process of sampling
    is carried out by employing specified predictor and corrector methods.

    Attributes:
        predictor (Predictor): The predictor method to be used in the sampling process.
        corrector (Corrector): The corrector method to be used in the sampling process.
        sde (SDE): The stochastic differential equation governing the process to be sampled.
        sampler_config (DictConfig): Configuration settings for the sampler.

    Methods:
        make_sampler(predict_fn: Callable) -> Callable:
            Abstract method to create a sampling function based on the specified predictor,
            corrector, and SDE.
    """

    def __init__(self, predictor: Predictor, corrector: Corrector, sde: SDE, sampler_config: DictConfig) -> None:
        """Initializes the Sampler object with specified predictor, corrector, SDE, and configuration.

        Args:
            predictor (Predictor): The predictor method for the sampler.
            corrector (Corrector): The corrector method for the sampler.
            sde (SDE): The stochastic differential equation governing the process.
            sampler_config (DictConfig): Configuration settings for the sampler.
        """
        super().__init__()
        self.predictor = predictor
        self.corrector = corrector
        self.sampler_config = sampler_config
        self.sde = sde

    def make_sampler(self, predict_fn: Callable, auxiliary_fn: Union[Any, Callable]) -> Callable:
        """Abstract method to create a sampler function.

        This method is intended to be overridden by derived classes to provide
        specific implementations for creating a sampler function. The sampler
        function will utilize the specified predictor and corrector methods
        along with the provided SDE to generate samples of the stochastic process.

        Args:
            predict_fn (Callable): The model prediction function.
            auxiliary_fn (Callable): The auxiliary prediction function for the model.

        Returns:
            Callable: The constructed sampling function.

        Raises:
            NotImplementedError: If this method is not overridden by a derived class.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the make_sampler method.")
