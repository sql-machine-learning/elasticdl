# ElasticDL: 像写单机程序一样写分布式深度学习程序

2019 年秋天，在上海的 Google Developer Day 活动中，来自蚂蚁金服的 ElasticDL 团队
展示了 ElasticDL 的[第一个开源版本](https://events.google.cn/intl/en/developerdays2019/agenda/#table-row-2-34)。
本文更新这大半年来 ElasticDL 项目的进展。

ElasticDL 是一套分布式深度学习训练框架，其首要设计意图是简化分布式编程。它允许用
户只提供用 TensorFlow 2.0 API 描述的模型，而不需要用户写训练过程代码。

同时，ElasticDL 提供基于 Kubernetes 的弹性调度的能力 —— 当机群资源不足时，一个训
练作业里的进程减少；当其他作业结束释放资源后，进程数量随之增加。这样的做法比
TensorFlow distribution strategy 以及 PyTorch Elastic 专注容错（进程减少的情况下
作业不失败，但不会增加进程数量）更进一步，在实践中可以让机群的利用高达 90%。不仅
如此，每个作业的启动等待时间都相应缩短。

限于篇幅，本文主要介绍 ElasticDL 简化分布式深度学习系统开发的特点。弹性调度的思
路和 benchmark 留待下篇。

## 简化分布式深度学习编程

为了从海量数据中学习规律，我们需要编写分布式深度学习程序来完成训练任务。这在工业
场景中尤为常见。

可分布式深度学习程序的编写很难 —— 编程者既要了解深度学习，也要了解分布式系统开发。
在一个分布式深度学习系统中，需要启动和监控若干个 workers。因为既要拆分训练数据给
workers，还要综合各个 worker 算出的 gradients 来更新模型，所以涉及通信
（communication）和 同步（synchronization）。此外，当 worker 数目很多时，作业在
执行过程中有 worker 挂掉的概率也会变得很大。如果一个 worker 挂掉，则整个作业重启
或者恢复到最近的 checkpoint（fault recovery），那么重启之后可能又会有 worker 挂
掉导致重启，于是作业不断陷入重启和恢复，永远也无法完成。这进一步要求编程者具备设
计容错（fault tolerance）系统的能力。其实不仅分布式深度学习，其他分布式机器学习
程序、分布式离线和在线数据处理程序等各种分布式程序的写作，都对编程者有类似上述要
求。

一个常见的解决思路是为特定类型的作业提供分布式编程框架，让用户只需要完形填空一样
补上业务逻辑，而分布式计算（包括通信、同步、和容错）都由框架的代码来完成。一个典
型的例子是离线数据处理程序用 MapReduce 框架来写。不管是 Google MapReduce 还是
Hadoop MapReduce，用户基本都只需填写 map 和 reduce 两个函数的实现即可。类似的，
在线数据流系统基于 Storm 和 Flink 来写，用户只需提供 bolts 和 nuts 这样的业务逻
辑定义。

在 ElasticDL 之前，蚂蚁金服的同事们使用过多种框架和类似框架的高层 API。这些方案
大都基于 TensorFlow 和 Kubernetes。

1. TensorFlow Estimator 作为构建在 TensorFlow 之上的一层 API，允许用户只需定义模
   型，而训练过程封装在一个函数调用里。这个函数调用可以把分布式作业启动在
   Kubernetes 上，前提是 Kubernetes 机群部署了 Kubeflow 项目提供的 TF-operator。
   这个方案的局限是：它仅支持 TensorFlow 的 graph mode，不支持 eager execution；
   而 eager execution 可以大幅简化调试，尤其方便跟踪网络各层输出。

2. Keras API 支持 TensorFlow 2.x 和 eager execution。目前 TensorFlow 2.x Keras
   API 还暂不支持 ParameterServer 分布式策略，对 AllReduce 分布式策略提供了实验
   性的支持。

3. Horovod 对用户代码有侵入性，用户除了必须熟悉 TensorFlow API 之外，还需学习
   Horovod API。

| 方案 | 模型定义方式 | 分布式执行机制 |
| --- | --- | --- |
| Estimator | TensorFlow Estimator API | Kubeflow TF-operator |
| Keras | TensorFlow Keras API | Kubeflow TF-operator |
| Horovod | Horovod with TensorFlow | Kubeflow MPI-operator |
| ElasticDL | TensorFlow Keras API | ElasticDL master process per job |

以上三个方案的共同局限是，虽然具备一定的容错能力，不过不支持弹性调度。而且它们都
依赖部署 Kubernetes operator，了解 Kubernetes 对 AI 专家来说颇有挑战。

针对这些局限，我们设计和开发了 ElasticDL 分布式计算框架。用户定义可以用
TensorFlow 2.x 的 Keras API 来定义模型。并且，分布式执行不要求 Kubernetes 机群有
任何特殊配置，而是利用每个作业里的 master 进程来协调训练数据分配、通信、同步、和
容错 —— 这也是 ElasticDL 除了容错，支持弹性调度的原因。

### 基于 ElasticDL 框架的编程

就像 MapReduce 框架中只需要用户完形填空两个函数：map 和 reduce，ElasticDL需要用
户填写 forward、loss、optimizer、feed 函数。其中 forward 定义深度学习的前向计算
过程（forward pass），ElasticDL 会调用 TensorFlow eager execution 的
GradientTape 机制来自动推导对应的后向计算过程（backward pass）；loss 函数返回模
型训练时使用的损失函数；optimizer 函数返回模型训练时使用的优化器；feed 定制化训
练数据到 TensorFlow 模型输入（tensors）的转换过程。

所有这些函数的编程只需要了解 TensorFlow API，不需要对分布式训练有任何背景知识。
写完之后，用户可以在单机上用小数据做调试验证。如果通过，可以不做任何代码修改就提
交到 Kubernetes 机群上做分布式的容错的大规模训练。

不同于 Kubeflow/TF-operator 给每个集群部署一个 Kubernetes Operator 的方式，
ElasticDL 为每个作业引入一个 master 进程。通过调用 Kubernetes API，master 进程了
解集群情况；同时，作为作业的一部分，master 还了解深度学习作业的特点 —— 包括利用
Python inspection 机制了解上述各个函数的特点，其中调用的 API 函数等。所以，
master 有非常充分的信息来做更优的调度。比如 master 可以请 Kubernetes 把两个
worker 启动在同一台物理机上，共用一个 GPU —— 当一个 worker 读数据的时候，请另外一个
 worker 来做计算，从而始终保持较高的 GPU 利用率。

### 一个例子

我们用一个 MNIST 手写数字识别的例子来说明。

```python
def forward():
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

上述每个函数都很容易做单独测试（unit test）。而且，利用 TensorFlow 2.x eager
execution，上述函数很容易 log 每一层的输出。基于个特点，ElasticDL worker 在调用
forward 函数的时候，可以打印中间结果，便于调试和复现问题。

## ElasticDL 的弹性训练过程

给定上述模型定义，ElasticDL 的 master 进程按照 asynchronous 或者 synchronous SGD
方法，协调 workers 来做训练。当使用 asynchronous SGD 方法时，master 会启动一个高
性能的 parameter server，供各个 workers 使用。当使用 synchronous SGD 时，
ElasticDL 使用和才云科技合作研发的一个 Kubernetes-native 的 fault-tolerable
AllReduce 实现 FTlib。

### Master 负责动态数据划分

弹性训练过程的一个容易被忽略的前提是动态数据划分（dynamic data partitioning）。
在用 MPI 写分布式程序的时候，因为作业中进程数量是恒定的，所以经常采用静态数据
划分的做法 —— 在训练之前把训练数据预先分成 N 个文件，对应作业中的 N 个 worker 进
程。这个做法在弹性调度的时候就失效了 —— 因为弹性调度时，作业中的进程数量是可变的。
为此，需要实现动态数据划分。

ElasticDL 的动态数据划分是基于索引的。ElasticDL 要求训练数据是一个或者多个
[RecordIO](https://github.com/wangkuiyi/recordio) 格式的文件，或者是
[MaxCompute](https://www.alibabacloud.com/zh/product/maxcompute) 数据库系统中的
表（table）。这两种数据源都允许 master 进程在开始训练之前，在基本存储单元
（block）间快速跳跃着扫描数据，把数据分成小段，称之为任务（task）。每个 task 包
括的内容如下：

1. 文件名或者表名，
2. 第一条记录相对于文件（或者表）开始处的偏移（offset），
3. 这个 task 里的总记录数。

扫描结果是很多 tasks，master 把这些 tasks 放进一个 TODO 队列里。这个队列不一定需
要是 master 进程里的数据结构，可以是放在 etcd 里的 —— 因为 etcd 是不死的，所以
master 即使被高优先级作业抢占了，这个信息也不会丢失；可以通过在资源富余时重启
master 进程来恢复作业状态。

扫描和划分数据的同时，master 开始请 Kubernetes 启动 workers，总数不超过用户指定
的数量 N（最大并发度）。每当一个 worker 启动起来了，master 会收到 Kubernetes 发
来的通知；master 在一个 etcd 数据结构里记录“活着”的 workers。

扫描和划分数据结束之后，master 就依次从 TODO 队列里取出 task，通过 gRPC 发给某一
个活着的 worker，同时 master 把这个 task 挪进 DOING 队列里。接收到 task 的
worker 负责打开文件（或者表），并且从指定的 offset 开始依次读取记录，并且更新本
地模型。根据用户选择的 asynchronous 或者 synchronous 算法，workers 会通过调用
parameter server 或者 AllReduce 来协调更新全局模型。

当一个 worker 处理完了接收到的 task，它通过 gRPC 返回一个表示成功的标记；master
就把这个 task 从 DOING 队列挪到 DONE 队列了。当所有 task 都从 TODO 挪进了 DONE，
则说明一个 epoch 完成了。

如果一个 worker 失败了（比如被更高优先级作业抢占了），则 master 的 gRPC call 会
timeout；此时，master 把对应的 task 从 DOING 队列挪回 TODO 队列了。下一次有
worker 完成 task 时，master 会把这个 task 再发出去。这里有一个细节：有的 task 可
能被某个 worker 使用了一部分，也因此影响到了模型更新；此时 worker 被抢占，那么这
部分已经被处理的数据会因为 task 的下一次分发，被重复使用。不过这个并不影响机器学
习训练要求数据统计一致性的假设。而且其他动态数据划分方法造成的数据复用情况可能更
严重。

### Woker 调用 TensorFlow Eager Execution

ElasticDL worker 接收到的一个 task 通常包括多个 minibatches。对于每个 task，
worker 打开对应的文件或者表，随后做如下操作：

1. 读取一个 mini-batch 的训练数据。
2. 用本地模型（local model）作为参数调用用户定义的 forward 函数以计算 cost。如果
   模型很大，则部分参数可能来自于 parameter server。
3. 给定 cost，worker 利用 TensorFlow eager execution 的 GradientTape 机制，进行
   backward 计算，得到梯度（gradient）。
4. 如果是 synchronous SGD，此时 worker 调用 AllReduce 实现 FTlib 来同步
   gradients 并且更新模型。如果是 asynchronous SGD，worker 不定时的向 parameter
   server 上传 gradients，也不定时地从 parameter server 获取全局模型参数。

### 高效训练的优化

相对于 2019 年秋季 ElasticDL 在 Google Developer Day 上亮相时的状态，最近几个月
ElasticDL 项目针对性能优化做了很多工作。当时 ElasticDL 使用 Redis 作为 parameter
server。现在有了自己的用 Go 语言写的 parameter server。相对于 Redis， ElasticDL
parameter server 可以做一些深度学习计算，从而减少 worker 和 parameter server 之
间通信的次数。

这个变化和其他优化工作一起让同样的训练作业，总体训练时间下降了约 13 倍。最近一个
基于 DeepFM 模型的试验展示，用两个 parameter server 进程和四个 workers 进程来训
练，10 个 epochs 的总体时间从 1350 秒（ElasticDL 的 2019年9月版本）下降到 106 秒
（2020年2月版本）。这些优化策略包括：

- 在 parameter server 上惰性初始化（lazy initialize） embedding vectors —— 在使
  用到 vector 的时候才初始化。
- 把一个 embedding table 拆分到多个 parameter server 进程里以均衡存储与通信负载。
- worker 从 PS 请求 embedding vectors 时，先滤除重复的 embedding ID，只取回不同
  ID 的 vectors，从而减少通信量。
- worker 向 PS 发送梯度时，先把相同 ID 的梯度进行合并（调用 TensorFlow 的
  embedding vector combinanation 函数），从而减少通信量。

## 一个 ElasticDL 的使用实例

在本小节中，我们将使用 ElasticDL 进行一次 Kaggle 实战。本例中使用的是 Kaggle 上
Display Advertising Challenge 中的 criteo数据集，这是一个关于广告点击率预估的比
赛。我们使用 xDeepFM 模型来进行建模，所有的实例代码都放在了 ElasticDL 的 [model
zoo](https://github.com/sql-machine-learning/elasticdl/tree/develop/model_zoo/dac_ctr)
中。

### 数据预处理

1. 下载 criteo [数据
   集
   ](https://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset)
   。

1. 然后我们需要把原始数据转换为 RecordIO 文件格式。我们提供了如下的转换脚本：

   ```bash
   python convert_to_recordio.py \
      --records_per_shard 400000 \
      --output_dir ./dac_records \
      --data_path train.txt
   ```

   原始数据会被按照 19:1 的比例，拆分为训练集和验证集，转换后的数据放在
   dac_records 目录中。

1. 对原始数据进行特征统计。对于连续的特征，我们统计得出均值和方差；对于离散的特
   征，我们得出特征值个数。我们把统计后的数据放在一个文件中，供后续使用。

### 模型定义

xDeepFM 模型由三部分组成，分别是 linear logits，dnn logits 和 xfm logits。借助
Keras API，我们可以很清晰的描述模型结构。这里贴出 dnn logits 部分的描述代码，完
整的模型定义可以参见 model zoo。

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

我们首先在 Google Cloud 上创建一个 GKE 集群，并且把转换好的 RecordIO训练数据上传
到集群上。详细的过程可以参考 ElasticDL 的[gcloud教
程
](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/tutorials/elasticdl_cloud.md)
。

然后，我们在本地制作一个镜像，该镜像包含了 xDeepFM 模型定义，以及相关依赖包。

```bash
FROM tensorflow
RUN pip install elasticdl
COPY model_zoo /model_zoo
```

我们需要把该镜像推送到 GKE 集群能够访问到的仓库中，比如说 docker hub 的仓库中。

最后，我们通过 ElasticDL client 工具向 GKE 集群提交训练作业。我们使用
ParameterServer 分布式策略进行训练，本次作业中，我们启动了2个 parameter serve
pods 和5个 worker pods 共同参与训练。

```bash
elasticdl train \
  --image_name=${your_docker_hub_repo}/elasticdl:ci \
  --model_zoo=model_zoo \
  --model_def=dac_ctr.elasticdl_train.forward \
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
