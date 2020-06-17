# ElasticDL: 像写单机程序一样写分布式深度学习程序

## 深度学习程序 for Kubernetes 不容易写

一个深度学习的训练任务往往需要较多的训练数据，需要较长的训练时间。通常的做法是我们把单机程序给分布式化，利用集群的资源，启动多个 worker，来共同完成一个训练任务。

分布式程序的编写是相对困难的。用户需要关心一些额外的控制逻辑，比如如何把训练数据分发到各个 worker；如何能够在训练若干 step 之后，对验证集发起一次 evaluation。

同时，用户需要在分布式系统上启动多个 worker，让多个 worker 之间建立通信，来共同参与训练。这对希望专注于模型调优的算法同学来说是很不友好的。

在蚂蚁金服内部，TensorFlow 被广泛的在诸多业务场景中使用。Kubernetes 已经成为分布式操作系统的事实标准。
因此，我们接下来将对比在 Kubernetes 上运行 TensorFlow 的分布式训练程序的一些开源解决方案。

AllReduce 和 Parameter Server 是两种常用的分布式梯度聚合策略。在图像语音模型中，我们通常使用 AllReduce 策略。
在搜索广告推荐模型中，我们使用 Parameter Server 策略。

|     | TensorFlow 1.x  | TensorFlow 2.x Estimator API| TensorFlow 2.x Keras API|
|  ----  | ----  | --- | ---|
| AllReduce  | Horovod + Kubeflow | Limited Support| Kubeflow |
| Parameter Server  | Kubeflow |  Limites Support | Supported planned post 2.3|

我们可以观察到在Kubernetes 上执行 TensorFlow 2.x 的分布式训练程序的解决方案还暂不完整。
另外需要指出的是，Estimator API 仅支持 graph execution，不支持 eager execution，调试代码和网络各层输出比较麻烦。

## ElasticDL 是如何解决问题的

TensorFlow 2.x 支持 eager execution，并且推荐使用 更加精简的 Keras API 来定义模型。
ElasticDL 支持 TensorFlow 2.x 的 Keras API，为用户提供了良好的体验。
用户专注于描述单机程序，而不需要关心分布式程序的写法。ElasticDL 会自动把单机程序转为分布式训练程序。下面我们用一个mnist的训练例子来详细说明。

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

### ElasticDL master 提供 training loop

为了解决分布式训练中的数据读取问题，ElasticDL 在 master 中实现了动态数据分发。
首先 master 把训练数据的索引创建成一系列的 task，worker 会向 master 请求拿到 task。拿到 task 之后，worker 可以数据的索引找到对应的数据分片。

ElasticDL master 中还为这些 task 维护了三个队列，todo/doing/done 队列。
每次 master 分发一个task，都会把这个 task 从 todo 队列挪到 doing 队列。
如果一个 worker 被抢占或者因为其他原因失败，master 可以通过监控 doing 队列 task 的 timeout，
把这个 task 挪回到 todo 队列中。
如果 worker 顺利完成一个 task，master 则会收到通知，把这个 task 从 doing 队列挪到 done 队列。

ElasticDL 目前支持 RecordIO 和 MaxCompute 两种数据源。用户只需指定训练集的文件夹路径，或者 table 名字，训练数据的动态分发都是用户无感知的。

使用动态数据分发机制之后，我们不必为每个 worker 提前指定要读取的数据。更进一步的，worker 数目可以动态变化，从而更方便的支持弹性调度。

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
 --num_minibatches_per_task=2 \
 --num_ps_pods=1 \
 --num_workers=2 \
 --grads_to_wait=1 \
 --use_async=True \
 --job_name=test-train \
 --log_level=INFO \
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

- embedding vector 惰性初始化，用户无需提前指定embedding table的高度
- 把一个 embedding table 拆分到多个PS上存储与更新，均衡存储/通信/更新的负载
- worker从PS请求参数时，先滤除重复ID，只请求不同的参数，减少通信量
- worker向PS发送梯度时，本地先把相同ID的梯度进行合并，减少通信量

通过上述设计与实现，ElasticDL可以很高效的完成搜索推荐广告模型的训练。
我们用一个推荐中常用的deepFM模型为例，来说明 ElasticDL 在 Parameter Server 上对性能的提升。

## 用户实例：ElasticDL在花呗营销场景落地
