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

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("../data/tf_serving_gpu.csv")

f = plt.figure(figsize=(13, 6))
plt.plot(df["time"], df["gpu"], label="A high-priority TF serving job")

df = pd.read_csv("../data/elastic_training_gpu.csv")

plt.plot(df["time"], df["gpu"], label="A low-priority ElasticDL training job")

df = pd.read_csv("../data/overall_cluster_gpu.csv")

plt.plot(df["time"], df["gpu"], label="Overall cluster utilization")

plt.title(
    "Two jobs with different priorities running together",
    fontsize=14,
    fontfamily="Times New Roman",
)
plt.xlabel(xlabel="time(sec)", fontsize=14, fontfamily="Times New Roman")
plt.ylabel(ylabel="Utilized GPU", fontsize=14, fontfamily="Times New Roman")
plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
plt.xlim(0, 500)
plt.ylim(0, 11)
plt.subplots_adjust(hspace=0.5)
plt.show()
