# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod


class BasePredictionOutputsProcessor(ABC):
    """
    This is the base processor for prediction outputs.
    Users need to implement the abstract methods in order
    to process the prediction outputs.
    """

    @abstractmethod
    def process(self, predictions, worker_id):
        """
        The method that uses to process the prediction outputs produced
        from a single worker.

        Arguments:
            predictions: The raw prediction outputs from the model.
            worker_id: The ID of the worker that produces this
                batch of predictions.
        """
        pass
