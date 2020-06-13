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

from collections import namedtuple

FeatureInfo = namedtuple("FeatureInfo", ["name", "op_name", "dtype", "param"])


class TransformOp(object):
    HASH = "HASH"
    BUCKETIZE = "BUCKETIZE"
    LOOKUP = "LOOKUP"


def get_id_boundaries(feature_group):
    boundaries = [0]
    for feature_info in feature_group:
        boundaries.append(boundaries[-1] + get_max_id(feature_info))
    return boundaries


def get_max_id(feature_info):
    if feature_info.op_name == TransformOp.LOOKUP:
        return len(feature_info.param) + 1
    elif feature_info.op_name == TransformOp.HASH:
        return feature_info.param
    elif feature_info.op_name == TransformOp.BUCKETIZE:
        return len(feature_info.param) + 1
    else:
        raise ValueError("The op is not supported")
