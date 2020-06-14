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

"""
The decorator to declare the input tensors of a keras model.
For keras subclass model, it has a core method `def call(self, inputs)`.
But we don't know how many tensors the model accepts just from `inputs`.
To solve this, we can use this decorators just as follows:

Example:
@declare_model_inputs("wide_embeddings,", "deep_embeddings")
class WideAndDeepClassifier(tf.keras.Model):
    def __init__(self):
        pass

    def call(self, inputs):
        pass

And then we can get the input tensor names from the property of the model class
`WideAndDeepClassifier._model_inputs` => ["wide_embeddings", "deep_embeddings"]
"""


def declare_model_inputs(*args):
    def decorator(clz):
        input_names = list(args)
        if not input_names:
            raise ValueError("Model input names should not be empty.")

        if not all(isinstance(name, str) for name in input_names):
            raise ValueError("Model input names should be string type.")

        setattr(clz, "_model_inputs", input_names)

        return clz

    return decorator
