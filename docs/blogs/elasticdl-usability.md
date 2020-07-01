# ElasticDL: 像写单机程序一样写分布式深度学习程序

## 分布式深度学习程序难写

为了从海量数据中学习规律，我们需要编写分布式深度学习程序来完成训练任务。这在工业场景中尤为常见。

分布式深度学习程序的编写是相对困难的，编程者既要了解深度学习，也要了解分布式系统开发。
在一个分布式深度学习系统中，需要启动和监控若干个 worker，对数据和计算任务进行拆分，并且分发给 workers。
此外，还需要考虑 worker 之间的通信（communication）和 同步（synchronization）。
随着计算规模的增加，worker
数目也会增加。当 worker 数目很多时，作业在执行过程中有 worker 挂掉的概率也会变得很大。
如果一个 worker 挂掉，则整个作业重启，那么重启之后可能又会有 worker 挂掉导致重启，于是作业不断陷入重启。
此时，需要结合深度学习训练算法的数学性质，设计容错机制。
这要求编程者必须同时是深度学习和分布式系统的专家。

我们对编写分布式深度学习程序的现有开源方案进行了调研。一方面，TensorFlow 是当今最受欢迎的深度学习平台，在蚂蚁集团内部，TensorFlow
在诸多业务场景中被广泛使用；
另一方面，Kubernetes 是目前最先进的分布式操作系统，是公有云和私有云的事实工业标准。
因此，本文重点讨论在 Kubernetes 上运行 TensorFlow 分布式训练程序的解决方案。调研结果参见下表。

|  | 模型定义 | 分布式调度工具 |
| --- | --- | --- |
| 方案一 | TensorFlow Estimator API | Kubeflow TF-operator |
| 方案二 | TensorFlow Keras API | Kubeflow TF-operator |
| 方案三 | Horovod with TensorFlow | Kubeflow MPI-operator |

现有开源方案的使用仍有一定门槛。在模型定义方面，TensorFlow Estimator API 仅支持 graph execution，不支持 eager
execution，调试代码和网络各层输出较为麻烦。
TensorFlow 2.x 默认支持 eager execution，并且推荐使用更加精简的 Keras API
来定义模型。
TensorFlow Keras API 提高开发效率，降低使用门槛，与 eager execution
配合之后，使得程序更为直观，也更易调试。
目前 TensorFlow 2.x Keras API 还暂不支持 ParameterServer 分布式策略，对 AllReduce 分布式策略提供了实验性的支持。
而Horovod 有一定的侵入性，用户除了熟悉 TensorFlow API之外，还需学习 Horovod API。
另一方面，用户依赖 Kubeflow 项目提供的 Kubernetes Operator
在 Kubernetes 集群上运行分布式训练作业，这要求用户对 Kubernetes 分布式操作系统有一定掌握。

我们为此设计和开发了 ElasticDL
分布式计算框架，让编程者只需了解深度学习，不需要了解分布式系统开发。
同时，ElasticDL 从易用性的角度出发，直接支持了 TensorFlow 2.x 的 Keras API。

就像 MapReduce 框架中只需要用户完形填空两个函数：map 和 reduce，ElasticDL
需要用户填写 forward、loss、optimizer、feed 等函数。
其中 forward 定义深度学习的前向计算过程，
ElasticDL 会调用 TensorFlow eager mode 中提供的 Gradient Tape 接口，
来自动推导对应的后向计算过程（backward pass）；
loss 指定模型训练时使用的损失函数；
optimizer 指定模型训练时使用的优化器；
feed 用来定制化训练数据到 TensorFlow 的 tensor的转换过程。

所有这些函数的编程只需要了解 TensorFlow
API，不需要对分布式训练有任何背景知识。
这些函数也可以在单机上用小数据做调试验证，然后就可以放心地交给 ElasticDL
做分布式的容错的大规模训练了。

不同于 Kubeflow 通过增加 Kubernetes Operator 来定制
Kubernetes 的思路，ElasticDL 为每个作业引入一个 master（类似于 Google MapReduce）。
这个 master 作为作业的一部分，而不是 Kubernetes 的一部分，
不仅了解集群情况，更了解深度学习作业本身，所以有充分的信息来做更优的调度。
比如 master 可以请 Kubernetes 把两个 worker 启动在同一台物理机上，共用一个
GPU。
这样，一个进程读数据的时候，请另外一个进程来做计算，从而让 GPU
的利用率总是很高。

## ElasticDL 简化分布式深度学习程序编写

### 用户只需提供模型定义

如上文所述，用户使用 Keras API 定义模型结构。这里我们用一个 MNIST 手写数字识别的例子来详细说明。

```python
def custom_model():
    inputs = tf.keras.Input(shape=(28, 28), name="image")
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```

除了模型定义之外，用户还需要指定 dataset, loss，optimizer 以及 evaluation函数

```python
def loss(labels, predictions):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predictions, labels=labels
        )
    )

def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)

def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }

def dataset_fn(dataset, mode, _):
    def _parse_data(record):
        if mode == Mode.PREDICTION:
            feature_description = {
                "image": tf.io.FixedLenFeature([28, 28], tf.float32)
            }
        else:
            feature_description = {
                "image": tf.io.FixedLenFeature([28, 28], tf.float32),
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

    dataset = dataset.map(_parse_data)

    if mode == Mode.TRAINING:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset
```

在 TensorFlow 2.x
中，上述定义的每个接口都可以单独测试，我们可以很方便的在本地调试模型定义。

同时，TensorFlow 2.x 默认支持 eager execution，ElasticDL worker
可以直接调用模型定义，进行前向计算。
在反向计算中，worker 可以通过 TensorFlow 2.x 暴露的 Gradient Tape接口
来计算得到梯度。

### ElasticDL 提供 Training Loop

我们通常使用 mini-batch SGD 的方法来训练深度学习模型。ElasticDL worker
会进行如下步骤来完成对一个 mini-batch 的训练：

1. 读取一个 mini-batch 的训练数据
2. 获取模型参数，进行 forward 计算
3. 进行 backward 计算，得到梯度
4. 把梯度以某种方式进行聚合，并更新模型

我们需要一个更大的 training loop 来包含上述的四个步骤，确保 worker
可以持续的读取下一个 mini-batch 的数据，继续训练，直到满足终止条件。

ElasticDL master 中实现了这样的 training
loop，其关键点是通过动态数据分发，解决分布式训练中的数据读取问题。
首先 master 会根据数据索引将数据分片，然后为每个分片的索引创建一个 task。
ElasticDL worker 会向 master 请求拿到 task。拿到 task 之后，worker
可以数据的索引找到对应的数据分片。

ElasticDL master 中还为这些 task 维护了三个队列，todo/doing/done 队列。
任务开始时，master 会将所有 task 放入 todo 队列。每分发一个 task 给 worker，
都会把这个 task 从 todo 队列挪到 doing 队列。
如果一个 worker 被抢占或者因为其他原因失败，master 可以通过监控 doing 队列 task
的 timeout，
把这个 task 挪回到 todo 队列中。
如果 worker 顺利完成一个 task，master 则会收到通知，把这个 task 从 doing
队列挪到 done 队列。

由于ElasticDL master 负责把数据索引分发给所有的 worker，所以我们只需要给 master
配置数据源即可。
目前 ElasticDL 支持 [RecordIO](https://github.com/wangkuiyi/recordio) 文件和
 [MaxCompute](https://www.alibabacloud.com/zh/product/maxcompute) 表两种数据源。
用户只需配置训练数据集的 RecordIO 文件路径或者 MaxCompute 表名。

同时使用动态数据分发机制之后，worker 数目也可以动态变化。
新加入的 worker 可以直接向 master 请求分配数据分片，从而更方便地支持弹性调度
worker 的数量。

### ElasticDL 高效完成 Training Loop

Training loop 中的关键一步是把来自多个 worker 的梯度进行高效聚合。一种常用的梯度聚合策略是 Parameter Server (PS) 策略。
在 PS 策略下，模型参数被分成若干个 shard，存储在一组 PS 上。
worker 首先向 PS 请求参数，然后使用本地训练数据计算梯度，并把梯度发送给PS。
PS 使用 worker 上传来的梯度来迭代更新模型参数。

ElasticDL master 会首先进行组网，协调 worker 和 PS 之间进行通信。ElasticDL 的 master 将会被首先创建，
然后由 master 启动 worker pod，以及 parameter server pod，并且建立通信。
ElasticDL master 可以监控每个 pod 的状态，当有 pod 挂掉时，master 会重新拉起新的 pod。

其次，ElasticDL 使用 Go 实现了 parameter
server，具有良好的吞吐能力和可扩展性。
在搜索推荐广告等场景中，神经网络模型通常包含较大的 embedding
table。在一些情况下，embedding table 的大小会超过单机内存。
为此，我们针对 embedding table
做了一些额外的优化。

- embedding vector 在 PS 上惰性初始化，用户无需提前指定 embedding table 的大小
- 把一个 embedding table 拆分到多个 PS 上，均衡存储与通信负载
- worker 从 PS 请求参数时，先滤除重复 ID，只取回不同 ID 的参数，减少通信量
- worker 向 PS 发送梯度时，先把相同 ID 的梯度进行合并，减少通信量

通过上述设计与实现，ElasticDL 可以很高效的完成搜索推荐广告模型的训练。

ElasticDL 自去年9月份开源以来，我们对 Parameter Server
持续迭代开发，不断提升性能。
我们以一个推荐中常用的 deepFM 模型来进行测试，测试中使用 frappe 数据集。
在每次实验中，我们启动一个 parameter server 和四个 worker，训练10个
epoch。

| Parameter Server 实现 | 训练时间（秒） |
| --- | --- |
| By Redis (2019.9) | 1350 |
| By Go (2020.2) | 106 |

从上表中我们可以看出 Go Parameter Server 相比于之前的实现有10倍以上的提升。

## 使用 ElasticDL 进行 Kaggle 实战

在本小节中，我们将使用 ElasticDL 进行一次 Kaggle 实战。
本例中使用的是 Kaggle 上 Display Advertising Challenge 中的 criteo
数据集，这是一个关于广告点击率预估的比赛。
我们使用 xDeepFM 模型来进行建模，所有的实例代码都放在了 ElasticDL 的 [model
zoo](https://github.com/sql-machine-learning/elasticdl/tree/develop/model_zoo/dac_ctr)中。

### 数据预处理

1. 下载 criteo [数据集](https://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset)。

1. 然后我们需要把原始数据转换为 RecordIO 文件格式。我们提供了如下的转换脚本：

   ```bash
   python convert_to_recordio.py \
      --records_per_shard 400000 \
      --output_dir ./dac_records \
      --data_path train.txt
   ```

   原始数据会被按照 19:1 的比例，拆分为训练集和验证集，转换后的数据放在dac_records 目录中。

1. 对原始数据进行特征统计。对于连续的特征，我们统计得出均值和方差；对于离散的特
   征，我们得出特征值个数。我们把统计后的数据放在一个文件中，供后续使用。

### 模型定义

xDeepFM 模型由三部分组成，分别是 linear logits，dnn logits 和 xfm logits。
借助 Keras API，我们可以很清晰的描述模型结构。
这里贴出 dnn logits 部分的描述代码，完整的模型定义可以参见 model zoo。

```python
deep_embeddings = lookup_embedding_func(
    id_tensors, max_ids, embedding_dim=deep_embedding_dim,
)
dnn_input = tf.reshape(
    deep_embeddings, shape=(-1, len(deep_embeddings) * deep_embedding_dim)
)
if dense_tensor is not None:
    dnn_input = tf.keras.layers.Concatenate()([dense_tensor, dnn_input])

dnn_output = DNN(hidden_units=[16, 4], activation="relu")(dnn_input)

dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(
    dnn_output
)
```

### 提交训练任务

我们首先在 Google Cloud 上创建一个 GKE 集群，并且把转换好的 RecordIO
训练数据上传到集群上。
详细的过程可以参考 ElasticDL 的
[gcloud教程](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/tutorials/elasticdl_cloud.md)。

然后，我们在本地制作一个镜像，该镜像包含了 xDeepFM 模型定义，以及相关依赖包。

```bash
FROM tensorflow
RUN pip install elasticdl
COPY model_zoo /model_zoo
```

我们需要把该镜像推送到 GKE 集群能够访问到的仓库中，比如说 docker hub 的仓库中。

最后，我们通过 ElasticDL client 工具向 GKE 集群提交训练作业。
我们使用 ParameterServer 分布式策略进行训练，本次作业中，我们启动了2个 parameter serve pods 和
5个 worker pods 共同参与训练。

```bash
elasticdl train \
  --image_name=${your_docker_hub_repo}/elasticdl:ci \
  --model_zoo=model_zoo \
  --model_def=dac_ctr.elasticdl_train.custom_model \
  --volume="mount_path=/data,claim_name=fileserver-claim" \
  --minibatch_size=512 \
  --num_minibatches_per_task=50 \
  --num_epochs=20 \
  --num_workers=5 \
  --num_ps_pods=2 \
  --use_async=True \
  --use_go_ps=True \
  --training_data=/data/dac_records/train  \
  --validation_data=/data/dac_records/val \
  --master_resource_request="cpu=1,memory=1024Mi,ephemeral-storage=1024Mi" \
  --worker_resource_request="cpu=4,memory=2048Mi,ephemeral-storage=1024Mi" \
  --ps_resource_request="cpu=8,memory=6000Mi,ephemeral-storage=1024Mi" \
  --evaluation_steps=10000 \
  --job_name=test-edl \
  --log_level=INFO \
  --image_pull_policy=Always \
  --distribution_strategy=ParameterServerStrategy
```

约迭代8万个 step 后模型收敛，AUC 可以达到 0.8002 左右。
