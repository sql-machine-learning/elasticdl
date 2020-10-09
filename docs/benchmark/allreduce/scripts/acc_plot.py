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

if __name__ == "__main__":
    df = pd.read_csv("../data/cifar10_resnet20_acc.csv")

    f = plt.figure(figsize=(10, 8))

    x = df["epoch_index"]
    y = df["baseline_4workers_0"]
    plt.plot(x, y, "--", label="baseline_4workers")

    y = df["baseline_4workers_1"]
    plt.plot(x, y, "--", label="baseline_4workers")

    y = df["baseline_2workers_0"]
    plt.plot(x, y, "-.", label="baseline_2workers")

    y = df["baseline_2workers_1"]
    plt.plot(x, y, "-.", label="baseline_2workers")

    y = df["elastic_2_4workers_0"]
    plt.plot(x, y, label="elastic_2_4workers")

    y = df["elastic_2_4workers_1"]
    plt.plot(x, y, label="elastic_2_4workers")

    plt.title("The Accuracy of Resnet20 on cifar10 test dataset")
    plt.xlabel(
        xlabel="Iteration epoches", fontsize=18, fontfamily="Times New Roman"
    )
    plt.ylabel(ylabel="Accuracy", fontsize=18, fontfamily="Times New Roman")
    plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
    plt.ylim((0.3, 1))
    f.savefig("../data/experiment_3.pdf", bbox_inches="tight")
    f.savefig("../data/experiment_3.png", bbox_inches="tight")
