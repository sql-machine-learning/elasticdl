# ElasticDL：同时提升分布式训练作业的研发效率和集群利用率

在开发分布式深度学习作业时，往往无法同时兼容作业开发效率和集群利用率。
目前在 Kubernetes 上进行分布式训练的作业，需要申请的资源完全分配后，
才能开始分布式训练，比如 Kubeflow 的 TF-operator 在 Kubernetes 上
提交的 TensorFlow 训练任务。这种情况下，影响研发效率的主要因素是资源
分配的等待时间，当资源不足时，需要等待其他用户的作业释放资源。
如果想保证能快速开始训练任务，可以给用户分配专有资源池，但是此时
集群资源没法给其他用户共享，会降低集群资源率。

如果分布式训练作业支持弹性调度，当集群可用资源少于申请资源时，也可以拉起
训练作业，当有其他作业释放资源时，能自动扩容到所申请的资源量，这样就能
缩短用户作业等待时间，也能提升集群资源利用率。

ElasticDL 基于 Kubernetes API 实现分布式深度学习训练任务的弹性调度，
旨在提供分布式深度学习训练作业的研发效率和集群资源利用率。


## Kubernetes 弹性调度

## 一、弹性调度

### 弹性调度和刚性调度

- 刚性调度[（gang scheduling）](https://en.wikipedia.org/wiki/Gang_scheduling): 运行时如果有一个进程挂了（比如被更高优先级的作业抢占了资源）
，则整个作业挂掉。等资源足够再启动所有的 N 个进程了，则可以重启或者从最近的 checkpoint 恢复作业。
  - Google 开源的 Kubeflow 是一系列 Kubernetes 扩展 operator，支持在 Kubernetes 上分布式地运行 TensorFlow 作业。因为 TensorFlow runtime 目前支持一定程度的容错，所以作业执行过程中，如果有一些 worker 挂了，剩下的可以继续。但在资源富裕的情况下，却不支持自动恢复 worker 的数量。

- 弹性调度（elastic scheduling）：**训练作业运行过程中，进程数量的变化不影响作业。**
  - 运行过程中，一个或者几个进程被高优先级的作业抢占，剩下的进程不受影响地继续进行。如果将来资源富裕了，系统可以加几个进程，此时作业仍然不受影响地继续运行。
  - Facebook 开源的 PyTorch Elastic 也是类似的扩展，支持分布式 PyTorch 作业。在启动分布式训练任务时需要指定 worker 的数量上限和下限，任何一个 worker 均可加入或者脱离训练任务，当 worker 数量发生变化是，任务中的所有 worker 需要重新协商，建立新的分布式训练任务，然后从之前某个 checkpoint 恢复模型进行训练。在 Kubernetes 优先级抢占环境下，如果 worker 数量变化频繁，其需要反复停止训练并重新开始，弹性效率较低；同时如果所剩 worker 少于下限，训练作业将会失败。
  - ElasticDL 可以启动和调度分布式 TensorFlow、PyTorch 作业。也很容易支持分布式 XGBoost 作业。

## ElasticDL 弹性调度分布式架构

ElasticDL 采用 master-worker 的分布式架构，下图为采用 Parameter Server 分布式策略的 ElasticDL 架构。

![ElasticDL Parameter Server Architecture](../figures/elasticdl_ps_architecture.jpg)

图中的 master 主要包含两个角色：

1. 负责启动 worker pod 和 PS pod，并管理所启动 pod 的生命周期，当有 pod 因为被抢占等原因失败时，会重启 pod。
1. 将数据分片并构造成 task，每个 task 包含一个数据分片，并将 task 分派给 worker 来计算梯度。如果 worker 负责的 task 失败，则 master 会将此 task 分配给其他 worker 重新计算梯度。

图中的每个 worker 都拥有完整的模型定义，其从 master 获取 task 获取数据分片后计算梯度上报给 PS，PS 负责梯度更新。

## ElasticDL 弹性调度 benchmark

为了说明 ElasticDL 弹性调度的优点，我们做了两组实验来说明 ElasticDL 可以带来用户体验和集群利用率的双丰收。

### 实验一：多个 AI 训练作业

考虑两个 AI 训练作业需要的资源总和略超过集群的情况：

- 如果没有弹性调度，则两个作业顺序执行。第二个作业的发起人需要等很久 —— 用户体验不好。并且任何时刻只有一个作业在运行 —— 集群资源用不满。
- 如果有弹性调度，则两个作业并发执行，虽然后启动的作业拿不到期待的全部资源，但是也马上就开始执行了 —— 用户体验好。因为两个作业并发 —— 集群被用满。

我们做了一个实验来验证上述好处，这个实验可以在 ASI 集群和开源 Kubernetes 集群上复现。

![Utilized CPU with Training Jobs](../figures/utilized_cpu_with_jobs.jpg)

上图对应的实验里，我们用刚性调度的方式提交了两个训练作业，每个作业都需要 175 个 CPU。
而集群总 CPU 数是 320，不足以同时运行两个作业，所以依次运行它们。可以看到第一个作业在 650 秒时结束。
随后集群花了一点时间调度，然后开始运行第二个作业，直到 1300 秒时结束。

下图对应的实验里，我们用 ElasticDL 来执行同样的两个训练作业。第一个作业提交之后的 300 秒，
我们提交了第二个作业。第二个作业⻢上就开始运行，用满了集群剩下的资源，而不需要等到 第一个作业结束。
在 650 秒时，第一个作业结束。随后，在 1100 秒时，第二个作业也结束了。因为弹性调度，
使得两个作业尽量同时运行，所以总结束时间比也上图要早。

总结:

- 用户等待作业启动时间几乎是 0。这对于 AI 很重要，因为用户最关注的是第一个迭代尽快 开始 —— 如果第一个迭代 fail 了，很可能是用户程序的 bug。
- 集群利用率高。第二个弹性调度实验执行期间，有一段时间集群利用率是 100%;其他时间也不低于第一个刚性调度实验。
- 作业完成更快。第二个试验里，两个作业用了约 1100 秒;第一个实验里需要约 1300 秒。

### 实验二：AI 作业和在线服务混布

运行各种在线服务的生产集群，通常需要留出余量资源，以应付突然增⻓的用户请求量。我们希望 利用这些 “余量” 来做 AI 训练，从而提升集群利用率。
下面实验验证:通过用较低优先级运行 ElasticDL 训练作业，在用户请求增加的时候，Kubernetes 自动扩容在线服务(nginx);
此时 ElasticDL 作业自动释放资源，配合在线服务的扩容。当流量高峰过去之后，Kubernetes 自动缩容 nginx 服务，
此时，ElasticDL 自动利用释放的资源。

![Utilized CPU with An Nginx job](../figures/utilized_cpu_with_nginx.jpg)

图中紫色曲线是 nginx 服务使用的 CPU 数量，随用户请求数量变化。绿色曲线是 ElasticDL 训练作业使用的 CPU 数量，随 nginx 的资源需求自动变化。蓝色曲线是机群的总体资源利用率 —— 保持在 90% 以上。

### 实验三：训练时调整 worker 数量不影响收敛性

有用户担心训练过程中 worker 的数量发生变化，会导致不收敛。实际情况下从未发生这类问题。
使用 [Kaggle Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) 的数据集,
用 ElasticDL 和用刚性调度的 Kubeflow tf-operator 分别训练 wide & deep 模型，收敛曲线如下:

![AUC with different worker number](../figures/auc_with_different_workers.jpg)

## ElasticDL 在蚂蚁金服花呗推荐场景的实践

