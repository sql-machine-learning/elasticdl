---
header-includes: |
   \usepackage{fontspec} 
   \usepackage{xeCJK} 
   \setCJKmainfont{Noto Sans SC}
---

# ElasticDL：同时提升用户体验和集群利用率

## 弹性调度和刚性调度

- 刚性调度[（gang scheduling）](https://en.wikipedia.org/wiki/Gang_scheduling): **一个作业里的 n 个进程，要么都运行，要么都死掉。**

  - Google 开源的 Kubeflow 是一个 Kubernetes 扩展 operators，支持在 Kubernetes 上分布式地运行 TensorFlow 作业。因为 TensorFlow runtime 目前支持一定程度的容错，所以作业执行过程中，如果有一些 workers 挂了，剩下的可以继续。不过不支持因为日后资源富余，恢复 workers 数量。
  - Facebook 开源的 PyTorch Elastic 也是类似的扩展，支持分布式 PyTorch 作业。号称 Elastic，其实是 job 失败后从 checkpoint 重启。
  - XGBoost、MXNet 社区也习惯于复用 Kubeflow。用 MPI 写的程序也可以利用 Kubeflow 拉起。
  
  以上 Kubernetes operators 都可以在蚂蚁金服的 ASI（旧称 Sigma）上部署。

  Gang scheduling 的特点是：运行时如果有一个进程挂了（比如被更高优先级的作业抢占了资源），则整个作业挂掉。等资源足够再启动所有的 n 个进程了，则可以重启（或者从最近的 checkpoint 恢复）。
  
- 弹性调度（elastic scheduling）：**训练作业运行过程中，进程数量的变化不影响作业。**

  - 运行过程中，一个或者几个进程被高优先级的作业抢占，剩下的进程不受影响地继续进行。如果将来资源丰沛了，系统可以加几个进程，此时作业仍然不受影响地继续运行。
  - ElasticDL 可以启动和调度分布式 TensorFlow、PyTorch 作业。也很容易支持分布式 XGBoost 作业。


## ElasticDL 和其他方案的对比

AI 作业的 elastic scheduling 需要了解 AI 作业的模型，所以更适合实现在 AI 训练作业专用的框架里，而不是通用的 Kubernetes operators 里。为此我们写了 ElasticDL 框架。目前公司内部 Kubemaker 项目部署的 Kubernetes operators 支持刚性调度的分布式 AI 作业。


|             | Gang scheduling | Elastic scheduling |
|-------------|-----------------|--------------------|
| TensorFlow  | Kubeflow        | ElasticDL          |
| PyTorch     | PyTorch Elastic | ElasticDL          |
| XGBoost     | Distributed XGBoost | ElasticDL      |

Elastic scheduling 的实现可以带来用户体验和集群利用率的双丰收，而一般的技术都是在两个优点之间做权衡（compromise）。我们做了两组实验来说明 ElasticDL 的优点。


## 实验一：多个 AI 训练作业

考虑两个 AI 训练作业需要的资源总和略超过集群的情况：

- 如果没有 elastic scheduling，则两个作业顺序执行。第二个作业的发起人需要等很久 —— 用户体验不好。并且任何时刻只有一个作业在运行 —— 集群资源用不满。
- 如果有 elastic scheduling，则两个作业并发执行，虽然后启动的作业拿不到期待的全部资源，但是也马上就开始执行了 —— 用户体验好。因为两个作业并发 —— 集群被用满。

我们做了一个实验来验证上述好处。这个实验可以在 ASI 集群和开源 Kubernetes 集群上复现。实验结果如下图。

![](./data/1.pdf)

上图对应的实验里，我们用 gang scheduling 的方式提交了两个训练作业，每个作业都需要 175 个 CPU。而集群总 CPU 数是 320，不足同时运行两个作业，所以依次运行它们。可以看到第一个作业在 650 秒时结束。随后集群花了一点时间调度，然后开始运行第二个作业，直到 1300 秒时结束。

下图对应的实验里，我们用 ElasticDL 来执行同样的两个训练作业。第一个作业提交之后的 300 秒，我们提交了第二个作业。第二个作业马上就开始运行，用满了集群剩下的资源，而不需要等到第一个作业结束。在 650 秒时，第一个作业结束。随后，在 1100 秒时，第二个作业也结束了。因为弹性调度，使得两个作业尽量同时运行，所以总结束时间比也上图要早。

总结：

- 用户等待作业启动时间几乎是 0。这对于 AI 很重要，因为用户最关注的是第一个迭代尽快开始 —— 如果第一个迭代 fail 了，很可能是用户程序的 bug。
- 集群利用率高。第二个实验（elastic scheduling）执行期间， 有一段时间集群利用率是 100%；其他时间也不低于第一个实验（gang scheduling）。
- 作业完成更快。第二个试验里，两个作业用了约 1100 秒；第一个实验里需要约 1300 秒。


## 实验二：AI 作业和在线服务混布

运行各种在线服务的生产集群，通常需要留出余量资源，以应付突然增长的用户请求量。我们希望利用这些“余量”来做 AI 训练，从而提升集群利用率。下面实验验证：通过用较低优先级运行 ElasticDL 训练作业，在用户请求增加的时候，Kubernetes 自动扩容在线服务（NGINX）；此时 ElasticDL 作业自动释放资源，配合在线服务的扩容。当流量高峰过去之后，Kubernetes 自动缩容 NGINX 服务，此时，ElasticDL 自动利用释放的资源。

![](./data/2.pdf)

图中紫色曲线是 NGINX 服务使用的 CPU 数量，随用户请求数量变化。绿色曲线是 ElasticDL 训练作业使用的 CPU 数量，随 NGINX 的资源需求自动变化。蓝色曲线是机群的总体资源利用率 —— 保持在 90% 以上。


## 实验三：训练时更改 worker 数量不影响收敛性

有用户担心训练过程中 worker 的数量发生变化，会导致不收敛。实际情况下从未发生这类问题。用 ElasticDL 和用 gang scheduling 分别训练 wide-and-deep DNN model，收敛曲线如下：

![](./data/3.pdf)

可以看到，采用 gang scheduling 持续用 4 个或者 8 个 workers，和用 ElasticDL 并且 worker 数量在 4 到 8 之间变化，得到的收敛曲线很难分辨。差别在自然误差范围之内。


## ElasticDL 的 PAI 组件的用户反馈

花呗营销的同事建议我们把 ElasticDL 也封装成一个 PAI 组件，进一步降低用户门槛。2020年5月22日，我们发布了这个组件。不到两周的时间增加了 14 个用户，并且收到一些反馈。

1. 花呗营销

   > 1. 模型迭代速度快，花呗场景的DBAN模型相比 ALPS，训练20w steps，由3.5h 缩短到约 2h
   > 2. 模型精度更高，花呗场景的DBAN模型离线验证集 AUC 和在线 ABTest 的结果都优于 ALPS 训练的模型
   > 3. 模型开发效率高，ElasticDL 使用 Keras API 开发模型，可以本地调试模型。

1. 相互宝技术部-相互宝产品及智能理赔技术组

   > 用 FELib + ALPS 需要为每个特征列配置预处理的 conf 文件，在特征列数目较多时，会比较费时费力。而业务上的项目时间往往比较紧急。使用ElasticDL-DeepCTR 不用配置 conf，自动做特征预处理并提供了预制算法，节省了很多模型开发时间。

1. 蚂蚁集团-数字金融线-数字金融线技术部-财富技术部-基金与理财服务组-投顾与开放技术组
1. 蚂蚁集团-CTO线-数据科学部

1. 饿了么：
   1. 新零售&新餐饮研发部： ElasticDL-DeepCTR wide&deep 算法比LR、FM、GBDT+LR 验证集AUC要高
   1. 用户画像和商品知识图谱组

1. 口碑：
   1. 用户增长部： 训练模型部署离线打分，用于圈人建模
   1. 行业算法部

1. 阿里集团-新零售智能引擎事业群-搜索推荐事业部-算法技术-营销技术：商品销量预测，流量调控机制，属于手淘聚划算/c2m/重点业务。
1. 阿里集团-CTO线-数据技术及产品部-数据资产及算法平台-数据算法-OneLocation与地动仪
1. 阿里集团-阿里云智能事业群-阿里云-基础产品事业部-云安全-安全研发-智能安全&大中台-智能安全-安全算法


## 未来工作计划

1. 目前在同一个优先级里的资源，调度时各个作业先来先得。将来可以加一个 Kubernetes operator，和同一个优先级的多个 ElasticDL 作业的 masters 协商，支持“互相礼让”，支持“资源匀动”。

1. 提供更底层的 Kubernetes operators 和其他扩展，为 Kubernetes 调度器提供输入。这样，ElasticDL master 在启动 worker 和 parameter server 的时候，可以有更多细致要求 —— ”启动 4 个 worker 在同一个机架上，用某规格的虚拟 GPU；启动另外 2 个 worker 在同一个机架里，用另一规格的 GPU”。

1. 因为 ElasticDL master 管理一个作业里的进程，我们可以利用 master 知道这个作业的模型、数据、训练算法等信息，做更细致的调度 —— 比如每两个 workers 复用一个 GPU，这样一个 worker 在读数据时，另一个 worker 在利用 GPU 做计算，从而更大限度地提升利用率。
