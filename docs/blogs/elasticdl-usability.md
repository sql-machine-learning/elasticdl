# ElasticDL: 像写单机程序一样写分布式深度学习程序

## 分布式深度学习程序难写

一个深度学习的训练任务往往需要较多的训练数据，和较长的训练时间。一个通常的做法是把单机程序给分布式化，利用集群的资源，启动多个 worker，来共同完成一个训练任务。

分布式深度学习程序的编写是相对困难的，编程者既要了解深度学习，也要了解分布式系统开发。
在一个分布式深度学习系统中，需要启动和监控若干个 workers 进程，对数据和计算任务进行拆分，并且分发给 workers。
此外，还需要考虑 workers 之间的通信（communication）和 同步（synchronization）。
随着计算规模的增加，workers 进程数目也会增加。当计算规模很大时，包含数十个进程的作业在执行过程中一个进程都不挂的概率几乎是0。
如果一个进程挂掉，则整个作业重启，那么这个作业会陷入永不停歇的重启，无法结束。
此时，需要结合深度学习训练算法的数学性质，设计容错机制。
这要求编程者必须同时是深度学习和分布式系统的专家。

我们为此设计和开发了 ElasticDL 分布式计算框架，让编程者只需了解深度学习，不需要了解分布式系统开发。

就像 MapReduce 框架中只需要用户完形填空两个函数：map 和 reduce，ElasticDL 只需要用户填写 forward、cost、feed 三个函数。
其中 forward 定义深度学习的前向计算过程，
ElasticDL 会调用 TensorFlow eager mode 中提供的 Gradient Tape 接口，
来自动推导对应的后向计算过程（backward pass）；
cost 指定模型训练时使用的 cost 函数；
feed 用来定制化训练数据到 TensorFlow 的 tensor的转换过程。

所有的这些函数的编程只需要了解 TensorFlow API，不需要对分布式训练有任何背景知识。
这些函数也可以在单机上用小数据做调试验证，然后就可以放心地交给 ElasticDL 做分布式的容错的大规模训练了。

ElasticDL 要求分布式计算平台是 Kubernetes。
一方面 Kubernetes 是目前最先进的分布式操作系统，是公有云和私有云的事实工业标准；
另一方面，ElasticDL 一改 Kubeflow 通过增加 Kubernetes operator 的方式定制 Kubernetes 的思路，
为每个作业引入一个 master 进程（类似 Google MapReduce）。
这个 master 进程作为作业的一部分，而不是 Kubernetes 的一部分，
不仅了解集群情况，更了解深度学习作业本身，所以有充分的信息来做更优的调度。
比如 master 进程可以请 Kubernetes 把两个 workers 启动在同一台物理机上，共用一个 GPU。
这样，一个进程读数据的时候，请另外一个进程来做计算，从而让 GPU 的利用率总是很高。

TensorFlow 是当今最受欢迎的深度学习框架。在蚂蚁金服内部，TensorFlow 在诸多业务场景中被广泛使用。
我们发现 AllReduce 和 Parameter Server 是分布式训练程序中常用两种梯度聚合策略。
在图像语音模型中，AllReduce 策略被广泛的使用。
在搜索广告推荐模型中，我们更倾向于使用 Parameter Server 策略。

我们调研了目前在 Kubernetes 上运行 TensorFlow 分布式训练程序的一些开源解决方案。

| 分布式策略 | 模型定义 | Kubernetes 任务提交工具 |
| --- | --- | --- |
| ParameterServer | TensorFlow Estimator | Kubeflow TF-operator |
| AllReduce | Keras + Horovod | Kubeflow MPI-operator |

我们发现 TensorFlow Estimator 仅支持 graph execution，不支持 eager execution，调试代码和网络各层输出比较麻烦。并且，用户需要组合使用不同的工具，来编写不同分布式策略的训练程序。

TensorFlow 2.x 默认支持 eager execution，并且推荐使用更加精简的 Keras API 来定义模型。
TensorFlow Keras API 提高开发效率，降低使用门槛，与 eager execution 配合之后，使得程序更为直观，也更易调试。
目前 TensorFlow 2.x 的 ParameterServer 和 AllReduce 分布式策略对 Keras API 的支持还不完善。

而 ElasticDL 从易用性的角度出发，直接支持了 TensorFlow 2.x 的 Keras API。
ElasticDL 同时提供统一的 ElasticDL client 命令行工具来提交作业。

| 分布式策略 | 模型定义接口 | Kubernetes 任务提交工具 |
| --- | --- | --- |
| ParameterServer | TensorFlow Keras | ElasticDL client |
| AllReduce | TensorFlow Keras | ElasticDL client |

统一的模型定义接口和统一的任务提交工具，极大地减少了用户的心智负担，提高了工作效率。

## ElasticDL 是如何解决问题的

在 ElasticDL 中，用户专注于使用 TensorFlow Keras API 描述单机程序，而不需要关心分布式程序的写法。ElasticDL 会自动把单机程序转为分布式训练程序。下面我们用一个mnist的训练例子来详细说明。

### 使用 Keras API 定义模型

用户使用 Keras API 定义模型结构：

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

在 TensorFlow 2.x 中，上述定义的每个接口都可以单独测试，我们可以很方便的在本地调试模型定义。

同时，TensorFlow 2.x 默认支持 eager execution，ElasticDL worker 可以直接调用模型定义，进行前向计算。
在反向计算中，worker 可以通过 TensorFlow 2.x 暴露的 Gradient Tape接口 来计算得到梯度。

### ElasticDL master 提供 training loop

我们通常使用 mini-batch SGD 的方法来训练深度学习模型。ElasticDL worker 会进行如下步骤来完成对一个 mini-batch 的训练：

1. 读取一个 mini-batch 的训练数据
2. 进行 forward 计算
3. 进行 backward 计算，得到梯度
4. 把梯度以某种方式进行聚合，并更新模型

我们需要一个更大的 training loop 来包含这四个步骤，确保 worker 可以持续的读取下一个 mini-batch 的数据，继续训练，直到满足终止条件。

ElasticDL master 中实现了这样的training loop，其关键点是通过动态数据分发，解决分布式训练中的数据读取问题。

首先 master 会根据数据索引将数据分片，然后为每个分片的索引创建一个 task。
ElasticDL worker 会向 master 请求拿到 task。拿到 task 之后，worker 可以数据的索引找到对应的数据分片。

ElasticDL master 中还为这些 task 维护了三个队列，todo/doing/done 队列。
任务开始时，master 会将所有 task 放入 todo 队列。每分发一个 task 给 worker，
都会把这个 task 从 todo 队列挪到 doing 队列。
如果一个 worker 被抢占或者因为其他原因失败，master 可以通过监控 doing 队列 task 的 timeout，
把这个 task 挪回到 todo 队列中。
如果 worker 顺利完成一个 task，master 则会收到通知，把这个 task 从 doing 队列挪到 done 队列。

由于ElasticDL master 负责把数据索引分发给所有的 worker，所以我们只需要给 master 配置数据源即可。
目前 ElasticDL 支持 RecordIO 文件和 MaxCompute 表两种数据源。
用户只需配置训练数据集的 RecordIO 文件路径或者 MaxCompute 表名。

同时使用动态数据分发机制之后，worker 数目也可以动态变化。
新加入的 worker 可以直接向 master 请求分配数据分片，从而更方便地支持弹性调度 worker 的数量。

### ElasticDL 命令行工具提交作业

在本地完成对模型的调试之后，我们可以借助 ElasticDL 提供的命令行工具向 Kubernetes 集群提交分布式训练作业。我们只需要指定模型定义文件和一些额外的参数，包括资源配置等。

```bash
elasticdl train \
 --image_name=elasticdl:ci \
 --model_zoo=model_zoo \
 --model_def=mnist_functional_api.mnist_functional_api.custom_model \
 --training_data=/data/mnist/train \
 --num_epochs=1 \
 --master_resource_request="cpu=1,memory=4096Mi,ephemeral-storage=1024Mi" \
 --worker_resource_request="cpu=1,memory=4096Mi,ephemeral-storage=1024Mi" \
 --ps_resource_request="cpu=1,memory=4096Mi,ephemeral-storage=1024Mi" \
 --minibatch_size=64 \
 --num_ps_pods=1 \
 --num_workers=2 \
 --job_name=test-train \
 --distribution_strategy=ParameterServerStrategy \
 --output=model_output
 ```

 在上述例子中，我们指定了 Parameter Server 的分布式策略，由一个parameter server 和 两个 worker 共同完成训练任务。
 ElasticDL的master pod将会被首先创建，然后由 master 负责启动 worker pod，以及parameter server pod，并且建立通信。
 ElasticDL master可以监控每个pod的状态，当有pod挂掉时，master会重新拉起新的pod。

## Parameter Server 的改进

在搜索广告等场景，模型中可能包含较大的 embedding table，其内存会超过单机内存。我们通常使用 Parameter Server (PS) 分布式策略来训练此类模型。
在 PS 策略下，PS 上存储着模型参数，worker 从 PS 上请求参数。
worker 在本地使用训练数据计算梯度之后，把梯度再发送到 PS 上，PS 使用 worker 传来的梯度来迭代更新模型参数。

ElasticDL 用 Go 实现了 Parameter Server，具有良好的吞吐能力和可扩展性。并且，我们针对 embedding table 做了一些额外的优化。

- embedding vector 惰性初始化，用户无需提前指定 embedding table 的大小
- 把一个 embedding table 拆分到多个 PS 上存储与更新，均衡存储与通信的负载
- worker 从 PS 请求参数时，先滤除重复 ID ，只请求不同的参数，减少通信量
- worker 向 PS 发送梯度时，本地先把相同 ID 的梯度进行合并，减少通信量

通过上述设计与实现，ElasticDL 可以很高效的完成搜索推荐广告模型的训练。
我们用一个推荐中常用的 deepFM 模型为例，来说明 ElasticDL 相比于去年9月开源时的性能提升。

**TODO** 添加性能对比

## 使用 ElasticDL 进行 Kaggle 实战

本例中使用的是 Kaggle 上的 Display Advertising Challenge 挑战的 criteo 数据集。
我们使用 ElasticDL 训练一个 xDeepFM 模型。

**TODO** 加上更详细的过程说明
