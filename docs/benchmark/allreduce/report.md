# Benchmark: ElasticDL Calling the Fault-Tolerable AllReduce from Elastic Horovod

We usually use synchronous distributed SGD to train deep learning models in CV
and NLP. fault tolerance and Elastic scheduling can improve the resource utilization
and user experience. Horovod has released fault-tolerance AllReduce in v0.20.0.
ElasticDL implements elastic synchronous training on Kubernetes by calling
Horovod. So, we experiment to verify the performance of ElasticDL.

## Experiment 1: Elastic Scheduling of Multiple ElasticDL Jobs

Supposed the GPU requirement of two jobs is a little over the total of the
cluster. Without elastic scheduling, the latter job cannot start until
another job completes. There are two problems:

- Bad user experience. The submitter of the latter job must wait a long time.
- Low resource utilization. The rest of GPU in the cluster is free.

Using elastic scheduling, the latter job can start using the rest of GPU and
does not need to wait for the completion of the earlier job.
After the earlier job completes, the latter job can scale up to speed up
training. So, elastic scheduling can improve user experience and
resource utilization.

To verify the benefit of elastic scheduling, we perform an experiment on a
Kubernetes cluster with GPU. The experiment result is shown in the following
figure.

![overlap jobs](./data/experiment_1.png)

In the upper figure, we use gang scheduling to submit two training jobs one
after another, and each job needs 4 GPUs. However, we create a cluster
only with 6 GPUs using Alibaba Cloud Container Service for Kubernetes (ACK).
So, we cannot simultaneously run those two jobs on the cluster.
The latter cannot start until the earlier job completes.
We can find the earlier job completes at 465s. Then,
the latter job starts at 518s and ends at 968s. After the
earlier job completes, the latter job waits for 53s because ElasticDL must
launch the master pod firstly. The master pod does not need GPU,
so the GPU utilization is 0 before the master launches worker pods.

In the figure below, we use ElasticDL to submit the two same jobs.
We submit the latter job at 120s, and the latter job starts immediately.
All 6 GPUs on the cluster are used. After the earlier job ends at 474s,
the latter job scale up to 4 GPUs and ends at 809s. Due to elastic scheduling,
the elapsed time is less than gang scheduling.

## Experiment 2: AI Training Jobs and Serving Jobs are Running on a GPU Cluster

We usually set some rest on the serving cluster in case requests increase
sharply. We hope to utilize the rest to train an AI model to improve the
utilization of the cluster. On the experiment, we create a cluster with
8 NVIDIA T4 GPUs on ACK and start a TensorFlow Serving job for inference
with a Resnet20 model. Then we submit an ElasticDL training job with 6
low-priority GPUs configuration. When inference requests increase,
Kubernetes will scale up pods of the serving job, and the ElasticDL job will
release pods. When requests decrease, Kubernetes will scale down pods of
the serving job, and the ElasticDL job will relaunch pods to speed up training.

![preemption](./data/experiment_2.png)

In the figure, the ElasticDL GPUs vary with the GPUs used by the serving job.
The total GPUs are almost used all the time.

## Experiment 3: Changing the Worker Number Does not Hurt the Model Convergence

Some users may worry that changing the worker number will have a bad effect on
convergence. Using elastic AllReduce, the batch size varies with the worker
number. So, the learning rate also need to vary with the worker number to
remain the convergence. In ElasticDL, users can define a callback to adjust
learning rate according the worker number, like:

```python
def callbacks():
    def _schedule(epoch):
        LR = 0.001
        if epoch > 80:
            lr = LR * 0.01
        elif epoch > 60:
            lr = LR * 0.1
        elif epoch > 40:
            lr = LR * 0.2
        elif epoch > 35:
            lr = LR * 0.5
        else:
            lr = LR
        return lr * hvd.size()

    return [LearningRateScheduler(_schedule)]
```

In the experiment, we use ElasticDL and gang scheduling to train ResNet20 with
CIFAR-10 dataset.

The accuracy of the test dataset is the following:

![accuracy](./data/experiment_3.png)

In the experiment, we use gang scheduling with two workers or four workers
to train ResNet20. Each worker has an Nvidia Tesla P100 GPU.
The worker number of ElasticDL varies from 2 to 4. From the figure, there
is no obvious difference in the accuracy of training jobs.
