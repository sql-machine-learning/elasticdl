#!/usr/bin/env bash
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

# Generate mnist dataset
DATA_PATH=$1

python /scripts/image_label.py --dataset mnist \
        --records_per_shard 4096 "$DATA_PATH"

# Generate frappe dataset
python /scripts/frappe_recordio_gen.py --data /root/.keras/datasets \
    --output_dir "$DATA_PATH"/frappe

# Generate heart dataset
python /scripts/heart_recordio_gen.py --data_dir /root/.keras/datasets \
    --output_dir "$DATA_PATH"/heart
