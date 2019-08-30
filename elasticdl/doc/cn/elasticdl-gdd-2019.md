# ElasticDL: Kubernetes-native 的弹性分布式深度学习系统

今天，蚂蚁金服在 Google Developer Day 上宣布开源了基于 TensorFlow 2.0
eager execution mode 的分布式深度学习系统 ElasticDL。项目负责人王益和
我们分享了 ElasticDL 项目的设计意图和现状。

## 分布式深度学习的技术思路

基于 TensorFlow 的分布式训练系统大致可以分为以下四类。

|---------------|----------------|----------------|
|               | TensorFlow 1.x | TensorFlow 2.x |
|               |  graph mode    | eager mode     |
|---------------|----------------|----------------|
| in TensorFlow | TensorFlow's   | TensorFlow     |
| C++ runtime   | ps-based       | distribution   |
|               | distribution   | strategies     |
|               |                | (early stage)  |
|---------------|----------------|----------------|
| on TensorFlow | Uber Horovod   | Ant Financial  |
| Python API    |                | ElasticDL      |
|               |                | (early stage)  |
|---------------|----------------|----------------|


其中，ElasticDL 位于田字格的右下角。之所以选择这条技术思路，是为了利用
Kubernetes 实现容错和弹性调度。

## 高性能计算和云计算

在深度学习技术研发的早期，涉及的人员相对少，公用一个计算机群的人相对少，
计算作业之间的协调可以通过口头交流实现。大家更关心缩短运行时间，也就是
从作业启动到结束的这段时间。高性能计算技术（HPC）是解决这个问题的有效
途径，比如 NVIDIA 的 cuBLAS 和 cuDNN 优化高性能数学计算、NCCL 优化 GPU
之间的通信效率。

随着深度学习技术的大规模使用，很多工程师和研究员公用一个机群，通过商量
来协调调度显然不可行了，大家开始使用机群管理系统调度分布式作业。这其中，
Kubernetes 近年来一枝独秀，已经在各大公有云中广泛使用。

## 云计算和弹性调度

在 Kubernetes 上启动分布式 TenosrFlow 作业的常用方式是使用 Google
Cloud 开源的 Kubeflow。Kubeflow 是 Kubernetes 的一个”插件“，它询问
Kubernetes 计划分配那几台机器来运行一个分布式作业中的各个进程，随后告
知每个进程，所有其他进程的 IP 地址和 port。从而保证一个作业里各个进程
之间互相知道对方。

为什么需要让所有进程互相知道对方呢？这是 TensorFlow ps-based
distribution 方式（上述表格中的左上）要求的。TenosrFlow 1.x 原生的分布
式训练功能让一个作业中所有进程都执行 TensorFlow 1.x runtime 程序。这些
进程互相通信，互相协调成为一个“分布式 runtime“，来解释执行表示深度学习
计算过程的*计算图*（graph）。在开始分布式训练之初，graph 被 TensorFlow
runtime 拆解成若干子图；每个进程负责执行一个子图 —— 任何一个进程失败
（可能是被更高优先级作业抢占），则整个大图的执行就失败了。所以
TensorFlow 原生的分布式训练能力不是*容错*的（fault-tolerant）。不过，
它是可以从错误恢复（fault-recoverable）—— TensorFlow Python API 提供
checkpoint 的能力；如果一个作业失败了，可以重启作业，从最近的
checkpoint 开始继续执行。

Kubeflow 可以在 Kubernetes 上跑 TensorFlow 原生的分布式计算能力。但是
因为后者并不能容错，所以 Kubeflow 并不能无中生有。不能容错，也意味着不
能弹性调度。

## 对弹性调度的诉求

在很多人公用计算机群的情况下，支持弹性调度意味着极大提升团队效率和集群
的总体利用率。前者支持快速迭代以保持技术领先；后者决定企业成本和云计算
业务的盈利能力。

一个展示弹性调度效果的例子如下。假设一个机群里有 N 个 GPU，一个作业包
括一个进程，占用了 N/2 个 GPU。第二个作业需要 N/2+1 个 GPU；但是此时机
群里空闲 GPU 只有 N/2 个。如果没有弹性调度能力，那么第二个作业被迫等待，
直到第一个作业结束释放资源。这个等待时间很可能和第二个作业的运行时间同
量级。此时，集群的利用率很低，是 1/2。如果有弹性调度，那么第二个作业可
以马上启动，用 N/2 个 GPU 做计算。日后如果有更多空闲资源了，调度系统可
以增加其进程数量，充分利用资源。

另一个例子是，假设有一个作业已经在执行了，此时一个新的更高优先级的作业
需要资源，所以调度系统杀掉了（preempt）了第一个作业的几个进程来腾出资
源启动第二个作业。如果没有弹性调度和容错，那么第一个作业会失败，所有进
程都结束。直到有足够资源重启它，并且沿着最近的 checkpoint 继续。如果有
弹性调度，则第一个作业的剩下的进程可以继续执行，只是因为可用的 进程
（GPU）少了，所以速度慢一些而已。

以上两个例子都展示了弹性调度对集群利用率的提升，以及对团队工作效率的保
障。需要注意的是：**容错和弹性调度互为因果**。容错的意思是，作业不受其
中进程数量变化影响。弹性调度时，作业里的进程数量会随集群 workload 情况
增减，所以作业必须是容错的，才能和调度系统配合，实现弹性调度。也因为如
此，弹性调度依赖 **分布式编程框架和调度系统配合**。

今天，很多分布式编程框架都可以和 Kubernetes 配合实现容错和弹性调度。比
如 用于离线数据处理的 Hadoop MapReduce、用于在线数据处理的 Storm、在线
流数据引擎 Flink、分布式存储系统 Redis 和 HBase。其中适合深度学习的框
架有 Paddle EDL。基于 TensorFlow 的支持弹性调度的深度学习系统，据我们
所知，ElasticDL 是第一个。

## Kubernetes-native 的弹性调度

ElasticDL 通过实现一个 Kubernetes-native 的框架，调用 TensorFlow 2.0，
来实现弹性深度学习。

所谓 Kubernetes-native 指的是一个程序调用 Kubernetes API 来起止进程。
Google MapReduce 是一个 Borg-native 的分布式计算框架。用户通过运行一个
Borg 的客户端程度启动一个 MapReduce 作业。Borg 客户端调用 Borg API 提
交作业，并且启动一个 master 进程。这个 master 调用 Borg API 启动其他
workers 进程。ElasticDL 也类似，用户调用 ElasticDL 的命令行客户端程序
启动作业。这个客户端程序调用 Kubernetes API，启动 master 进程。master
进程继续调用 Kubernetes API 启动其他进程。master 进程也可以调用
Kubernetes API 监控其他进程。

如果 worker 挂了，按照分布式深度学习训练算法的数学特性，可以不用处理，
即可确保训练过程继续。如果一个 parameter server 进程挂了，master 会选
择一个 worker 进程，让它转换角色替补上挂掉的 parameter server 进程。在
以上两种情况下，master 都会调用 Kubernetes API，请它再启动一个额外的
worker 进程。如果启动成功，master 要带它入门，加入到与其他进程的协作中。
master 进程的状态（主要是三个 task queues：todo、doing、done）可以保留
在 Kubernetes 集群的 etcd 存储系统中。这样，万一 master 挂了，重启的
master 进程可以从 etcd 继承前世的状态。

以上是一个简化的描述。 ElasticDL 实现了多种分布式计算模式，每种模式实
现 fault-tolerance 的方式略有不同。我们会在后续文章中详细介绍。

Kubernetes-native 架构使得 master 进程有机会与 Kubernetes 协作实现容错
和弹性调度。不过，因为 ElasticDL 调用 Kubernetes API，也就意味着
ElasticDL 只能运行在 Kubernetes 上。

TensorFlow 原生的分布式计算能力不是 Kubernetes-native 的。所以
TensorFlow 不是绑定在 Kubernetes 这个平台上的。这是大家如果要用现有技
术在 Kubernetes 运行 TenorFlow 作业的话，需要依赖 Kubernetes 的扩展
Kubeflow 的原因。

理论上，不调用 Kubernetes API 也是可以实现一定程度的容错的。即使没有
Kubernetes 的通知，master 可以通过检查其他继承的心跳（heartbeat）或者
检查 TCP 链接状态，判断其他进程的生死存亡。但是，不调用 Kubernetes API
（或者其他调度系统的 API），master 无法通知调度系统重启进程，也无法得
知新启动的进程的信息，并且帮助它加入作业。这种“非 Kubernetes-native”的
容错方式颇为被动，只能接受资源紧张时一些进程被抢占而挂掉的事实，而不能
在其他作业释放资源后增加进程充分利用空闲资源。


## TensorFlow 2.0

如上文解释，为了保证 TensorFlow 最核心的 runtime 是平台无关的，我们没
法通过修改 runtime 实现完备的主动的容错和弹性调度。所以如文首的田字格
所示，ElasticDL 和 Uber Horovod 都是在 TensorFlow 的 Python API 上包一
层。

Horovod 基于 TenosrFlow 1.x。 一个 Horovod 作业的每个进程调用单机版
TensorFlow 做本地计算，然后收集 gradients，并且通过 AllReduce 调用汇聚
gradients 并且更新模型。Horovod 也是平台无关的，所以它提供的 AllReduce
操作不支持容错和弹性调度。这一点和 ElasticDL 不一样。

和 ElasticDL 一样的是，Horovod 需要从 TensorFlow “偷偷截获 gradients”，
在 TensorFlow 1.x 中，深度学习计算是表示成一个计算图（graph），并且由
TensorFlow runtime 解释执行，所以 Horovd 为了获得每个进程算的
gradients 并且 AllReduce 它们，就得 hack 进入图执行的过程。为此，
Horovod 要求使用者使用特定的 optimizer 代替 TensorFlow 提供的
optimizer，从而可以在优化模型阶段透露出 gradients。

一个调用 Horovod 的用户程序的结构如下。其中标记为 `(*)` 和 `(**)` 的部
分是 Horovod 要求用户写的，帮助 Horovod 截获 TensorFlow 计算得到的
graidents 的代码。如果用户不慎忘记写了，那么程序执行结果就不对了。

```python
hvd.init()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

loss = ...  # Build model...
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt) # (*)
train_op = opt.minimize(loss)

hooks = [hvd.BroadcastGlobalVariablesHook(0)] # (**)
with tf.train.MonitoredTrainingSession(checkpoint_dir，config, hooks) as s:
  while not s.should_stop():
    s.run(train_op)
```

ElasticDL 没有这些问题，因为它依赖的是 TensorFlow 2.0。TensorFlow 2.0
主推的 eager execution mode 采用和解释执行图完全不同的深度学习计算方式。
类似 PyTorch 的做法，前向计算过程把对基本计算单元（operator）的调用记
录在一个内存数据结构 tape 里，随后，反向计算过程（计算 gradients 的）
可以回溯这个 tape，以此调用 operator 对应的 gradient operator。这个
tape 提供一个操作让用户可以获取每个参数的 gradient。

ElasticDL 通过调用 TensorFlow 2.0 API 可以很直接地获取 gradients：

```python
with tf.GradientTape(persistent=True) as tape:
    outputs = self._model.call(features, training=True)
    loss = self._loss(outputs, labels)
    # Add regularization loss if any
    if self._model.losses:
        loss += tf.math.add_n(self._model.losses)
grads = tape.gradient(loss, self.get_trainable_items())
```

而且上面这段代码不是需要用户写的，而是 ElasticDL 的一部分。ElasticDL
用户需要写的代码对应上述 Horovod 代码范例中的一行 —— 定义模型。

```python
loss = ...  # Build model...
```

## 极简的 API 和使用方式

训练一个模型不只需要上述模型定义，还需要指定数据、优化目标（cost）、和
优化算法（optimizer）。用户总是希望能以尽量精简的方式指定这些信息，以
尽量少的代码描述训练作业。

ElasticDL 和 TensorFlow 其他的 high-level API，例如 Keras 和 TenorFlow
Estimator 一样， 几乎调用一个 API 函数就可以执行一个分布式训练作业。下
面这个程序使用 Keras。Keras 使用 TensorFlow 原生分布式训练能力，不支持容
错和弹性调度。

```python
mirrored_strategy = tf.distribution.MirroredStrategy()
with mirrored_stategy.scope():
    model = elasticdl.model_zoo.IncomePrediction()
    dataset = elasticdl.dataset.MaxCompute(
        "SELECT age, address FROM employee",
        (numeric_column("age"),
         categorical_column_with_hash_bucket("address")])
    model.compile(loss='mse', optimizer='sgd')
    model.fit(dataset, epoch=2)
    model.evaluate(dataset)
```

ElasticDL 的 API 相对更加精简一些。上述范例程序对应的 ElasticDL 版本如下：

```python
elasticdl.train(
    lambda: elasticdl.model_zoo.IncomePrediction()
    lambda: elasticdl.dataset.MaxCompute(
        "SELECT age, address FROM employee",
        (numeric_column("age"),
         categorical_column_with_hash_bucket("address")])
    loss='mse', optimizer='sgd')
```

主要的区别在于：在 Keras 程序里用户要选择分布式执行策略；而在
ElasticDL 程序里则不需要。这是因为 ElasticDL 自动选择分布式训练算法和
策略。

简单的说，对于有很大参数（需要 model parallelism）的模型，ElasticDL 使
用 asynchrnous SGD。这个方法配合 delayed model update 能把网络通信量减
少一个数量级。很多 NLP、搜索、推荐、广告的模型都符合这一类。
Asynchronous SGD 对于这类模型的表现比较稳定。对于图像识别和语音识别这
一类参数不太大的模型，ElasticDL 团队在开发一个 Kubernetes-native 的
AllReduce。和 Horovod 使用的 AllReduce 一样，ElasticDL AllReduce 把进
程间通信的拓扑组织成一个环，从而实现高性能的模型更新。与之不同的是，
ElasticDL AllReduce 是容错的 —— 在有进程失败导致 AllReduce 调用失败的
情况下，master 组织剩下的活着的进程构造一个新的环。

ElasticDL 项目希望通过这样的分而治之的策略，提供高性能并且易用的深度学习系统。

## ElasticDL 和 SQLFlow 的关系

今年早些时候，王益团队 开源了 SQLFlow。用户可以用扩展后的 SQL 语法，非
常精炼地描述整个数据流和 AI 流程。

比如，如果我们要为一个电子商务网站构造一个推荐系统，需要开发日志收集、
在线数据清洗、特征工程、模型训练，验证和预测等模块。每个模块可能需要投
入一个团队数轴甚至数月的时间。

最近几年里，很多互联网服务开始把数据直接上传到通用数据库中，比如蚂蚁金
服的很多数据是在 ODPS（也就是阿里云上的 MaxCompute 服务）以及新一代的
[智能数据系
统
](https://www.infoq.cn/article/tdpG65SjWSsdBgHmCTc5?utm_source=related_read&utm_medium=article)
。这促使我们考虑把数据清洗和预处理放在数据库中做，而特征工程、自动机器
学习、和训练过程在 ElasticDL 这样的 AI 引擎里做。SQLFlow 把扩展语法的
SQL 程序翻译成一个 Python 程序，把两部分链接起来。

在这样的场景中，如果 AI 需要很多参数，则用户也就需要在 SQL 程序中提供
这些参数。比如下面 SQL 语句从数据库中提取用户的年龄、工作部门、和工作
地点，来预测其收入。

```sql
SELECT age, address, department, income FROM employee
TRAIN DNNRegressor
COLUMN age, vocabularize(department), bucketize(address, 1000),
TARGET income,
WITH hidden_layers=[100, 150, 50],
     dist_strategy=mirrored, gpus=8
INTO my_first_model
```

其中，`TRAIN` 从句指定要训练的模型；`COLUMN` 从句指定如何把数据映射成
特征；`TARGET` 指定要预测的值；`WITH` 指定训练过程中的各种参数，其中
`dist_strategy` 是调用 Keras/TensorFlow 做训练是需要指定的分布式策略，
`gpus` 指定需要的资源。而这些，在 SQLFlow 调用 ElasticDL 的时候都是不
需要的，因为 ElasticDL 自动选择分布式策略和算法。

从这个例子可以看出，如果要让用户能提供尽量少的参数，人工智能引擎还需要
更加智能，提供包括 AutoML 和 auto feature engineering 的功能。
ElasticDL 项目任重道远。我们期待把上述 SQL 程序简化为如下形式：

```sql
SELECT * FROM employee TARGET income INTO my_first_model
```

## ElasticDL 项目的现状

ElasticDL 项目处于早期探索阶段。API 还在演化过程中。这次开源的版本，尚
不包括自动选择分布策略和算法的代码。相比在 TensorFlow runtime 中实现分
布式计算，基于 TensorFlow 2.0 eager mode 的 Python API 实现的分布式训
练性能差距还很大。ElasticDL 团队在和 Google Brain 团队合作，开发上述
asynchronous SGD + delayed model update 能力、以及 Kubernetes-native
AllReduce。希望在下一个版本中可以提供给大家使用。

目前 ElasticDL 实现的基于 parameter server 的分布式SGD 训练方法验证了
容错和弹性调度。并且在 Google Cloud 上的 Kubernetes 1.12 集群和阿里云
Sigma 3.1（一个 Kubernetes 的高性能实现）上都可以运行。并且，ElasticDL
团队开发了 SQLFlow 生成 ElasticDL 程序的 code generator。

我们希望尽早开源 ElasticDL 和尽早分享其设计意图，能汇聚来自不同公司和
社区的力量，一起探索 Google TensorFlow 2.0 和 Kubernetes 的分布式训练
生态，早日实现便捷的端到端的人工智能开发套件。
