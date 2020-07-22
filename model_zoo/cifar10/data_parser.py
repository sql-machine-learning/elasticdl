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

import tensorflow as tf

from elasticdl.python.common.constants import Mode


def parse_data(record, mode):
    if mode == Mode.PREDICTION:
        feature_description = {
            "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32)
        }
    else:
        feature_description = {
            "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }
    r = tf.io.parse_single_example(record, feature_description)
    features = {
        "image": tf.math.divide(tf.cast(r["image"], tf.float32), 255.0)
    }
    if mode == Mode.PREDICTION:
        return features
    else:
        return features, tf.cast(r["label"], tf.int32)
