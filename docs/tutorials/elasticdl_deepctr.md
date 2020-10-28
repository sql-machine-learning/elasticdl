# ElasticDL：Distributed Train the DeepCTR Model on Kubernetes

This document shows how to use ElasticDL to train premade Keras Models
on Kubernetes. In the document,  we use the model in DeepCTR which is a
popular package in CTR estimation and the dataset provided by Cretio from
Kaggle Advertising Challenge contest. There are 13 columns of integer
features I1-I13 and 26 columns of categorical features C1-C26 in the dataset.

## 1. Prerequisites

- Install Minikube, preferably >= v1.11.0, following the installation guide.
Minikube runs a single-node Kubernetes cluster in a virtual machine on your
personal computer.
- Install Docker CE, preferably >= 18.x, following the guide for building
Docker images containing user-defined models and the ElasticDL framework.
- Install Python, preferably >= 3.6, because the ElasticDL command-line
tool is in Python.
- Install [pyrecordio](https://pypi.org/project/pyrecordio/),
[deepctr](https://pypi.org/project/deepctr/) and TensorFlow >= 2.0,
[elasticdl_client](https://pypi.org/project/elasticdl-client/).

## 2. Convert Data into RecordIO

ElasticDL requires that the sample data can be queried by an integer
index, so users need to convert the training data in [criteo_sample.txt](https://raw.githubusercontent.com/shenweichen/DeepCTR/master/examples/criteo_sample.txt)
into RecordIO files.

```python
import argparse
import os
import pathlib
import sys

import recordio
import tensorflow as tf

COLUMNS = (
    ["label"]
    + [("I" + str(i)) for i in range(1, 14)]
    + [("C" + str(i)) for i in range(1, 27)]
)


def convert_data_to_tf_example(sample_data, columns):
    features = {}
    column_data = sample_data.split(",")
    for i, column_name in enumerate(columns):
        value = column_data[i].strip()
        if column_name[0] == "I" or column_name == "label":
            value = 0.0 if value == "" else float(value)
            feature = tf.train.Feature(
                float_list=tf.train.FloatList(value=[value])
            )
        else:
            feature = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value.encode("utf-8")])
            )
        features[column_name] = feature

    example = tf.train.Example(
        features=tf.train.Features(feature=features)
    ).SerializeToString()

    return example


def convert_to_recordio_files(file_path, dir_name, records_per_shard=10240):
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    writer = None
    with open(file_path, "r") as f:
        f.readline() # Skip the header in the first line.
        for index, row in enumerate(f):
            if index % records_per_shard == 0:
                if writer:
                    writer.close()

                shard = index // records_per_shard
                file_path_name = os.path.join(dir_name, "data-%05d" % shard)
                writer = recordio.Writer(file_path_name)
            example = convert_data_to_tf_example(row, COLUMNS)
            writer.write(example)

        if writer:
            writer.close()

# We use the same dataset for training and evaluation for simplity.
convert_to_recordio_files("./criteo_sample.txt", "./data/criteo_records/train")
convert_to_recordio_files("./criteo_sample.txt", "./data/criteo_records/test")
```

Besides RecordIO format, ElasticDL also supports other storages with the index
like MaxCompute Table.

## 3. Define Training APIs for ElasticDL with Premade Models

Users need to define `forward`, `loss`, `optimizer` and `feed` functions
for ElasticDL to submit a distributed training job.

`forward`: Return a Kera Model instance.

`loss`: Return the loss calculating with the model output and label.

`optimizer`: Return a TensorFlow/Keras optimizer instance.

`feed`: Return a dataset using TensorFlow.

For example, we define those APIs for the WDL model in DeepCTR.

```python
import tensorflow as tf
from deepctr.feature_column import DenseFeat, SparseFeat
from deepctr.models import WDL


def forward():
    sparse_features = [("C" + str(i)) for i in range(1, 27)]
    dense_features = [("I" + str(i)) for i in range(1, 14)]
    fixlen_feature_columns = [
        SparseFeat(
            feat,
            vocabulary_size=10000,
            embedding_dim=4,
            dtype="string",
            use_hash=True,
        )
        for i, feat in enumerate(sparse_features)
    ] + [DenseFeat(feat, 1,) for feat in dense_features]

    model = WDL(fixlen_feature_columns, fixlen_feature_columns, task="binary")
    return model


def loss(labels, predictions):
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            y_true=tf.cast(labels, tf.float32), y_pred=predictions,
        )
    )


def optimizer(lr=0.001):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def feed(dataset, mode, _):
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(parse_data, num_parallel_calls=8)

    return dataset
```

If we want to evaluate the model using a validation dataset during the training,
we also need to define `eval_metric_fn` like:

```python
def eval_metrics_fn():
    return {"auc": tf.keras.metrics.AUC()}
```

We can also see those codes in the [model zoo](https://github.com/sql-machine-learning/elasticdl/blob/develop/model_zoo/deepctr/wdl.py)
of ElasticDL.

## 4. Submit a Training Job on Minikube

### Start a Kubernetes cluster locally using Minikube

Minikube can uses VirtualBox, hyperkit and other hypervisors to create
the virtual machine cluster, like:

```bash
minikube start --vm-driver=virtualbox --cpus 2 --memory 6144 --disk-size=50gb
```

Then, we need to configure the local environment to re-use the Docker daemon
inside the Minikube instance using the following command.

```bash
eval $(minikube docker-env)
```

Lastly, we need to enable RBAC of Kubernetes by the following command.

```bash
kubectl apply -f \
https://raw.githubusercontent.com/sql-machine-learning/elasticdl/develop/elasticdl/manifests/elasticdl-rbac.yaml
```

If you happen to live in a region where raw.githubusercontent.com is banned,
you might want to Git clone the above repository to get the YAML file.

```bash
git clone https://github.com/sql-machine-learning/elasticdl
kubectl apply -f elasticdl/manifests/elasticdl-rbac.yaml
```

### Build the docker image

ElasticDL will launch docker containers to train the model on Kubernetes.
We need to build a docker image with the definition in step 3.
We can use ElasticDL client to build the image.

```bash
git clone https://github.com/sql-machine-learning/elasticdl
cd elasticdl
elasticdl zoo init --model_zoo=model_zoo
elasticdl zoo build --image=elasticdl:wdl_test .
```

Then, we can view the image by `docker image ls`.

### Submit a training job using ParameterServerStrategy

Firstly, we need to change the directory to the root of `./data` by `cd ../`.
Because, we need to mount the `./data` with RecordIO files to containers.
We can check whether `data` folder exists by `ls`. Then, we can sumit
a job by the command

```bash
elasticdl train \
  --image_name=elasticdl:wdl_test \
  --model_zoo=model_zoo \
  --model_def=deepctr.wdl.custom_model \
  --dataset_fn=feed \
  --training_data=/data/criteo_data/train \
  --validation_data=/data/criteo_data/test \
  --num_epochs=10 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --worker_resource_request="cpu=0.4,memory=1024Mi" \
  --ps_resource_request="cpu=0.2,memory=1024Mi" \
  --minibatch_size=32 \
  --num_minibatches_per_task=2 \
  --num_ps_pods=1 \
  --num_workers=2 \
  --log_loss_steps=1 \
  --evaluation_steps=5 \
  --job_name=test-wdl \
  --image_pull_policy=Never \
  --volume="host_path=${PWD}/data,mount_path=/data" \
  --distribution_strategy=ParameterServerStrategy \
  --output=/data/savedmodel/
```

If we want to submit a training job on ACK of Aliyun or GKE
of Google Cloud, we need to upload RecordIO files to the
PersistentVolume of the cluster. Then, we can submit a training
job like the above command on Minikube.

### Check Job Status

After the job submission, we can run the command `kubectl get pods` to list
related containers.

```txt
NAME                          READY   STATUS    RESTARTS   AGE
elasticdl-test-wdl-master     1/1     Running   0          116s
elasticdl-test-wdl-ps-0       1/1     Running   0          87s
elasticdl-test-wdl-worker-0   1/1     Running   0          87s
elasticdl-test-wdl-worker-1   1/1     Running   0          86s
```

We can also trace the training progress by watching the log from the master
container. The following command watches the evaluation metrics changing
over iterations.

```bash
kubectl logs elasticdl-test-mnist-master | grep "Evaluation"
```

The output looks like the following.

```txt
[2020-10-20 11:38:12,302] [INFO] [evaluation_service.py:227:complete_task] Evaluation metrics[v=5]: {'auc': 0.56953645}
[2020-10-20 11:38:13,259] [INFO] [evaluation_service.py:227:complete_task] Evaluation metrics[v=10]: {'auc': 0.55696714}
[2020-10-20 11:38:15,242] [INFO] [evaluation_service.py:227:complete_task] Evaluation metrics[v=15]: {'auc': 0.5685904}
[2020-10-20 11:38:16,228] [INFO] [evaluation_service.py:227:complete_task] Evaluation metrics[v=20]: {'auc': 0.57771325}
```

After the job finished, we can get SavedModel in the `${PWD}/data/savedmodel`.
