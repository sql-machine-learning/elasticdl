# ElasticDL: A Kubernetes-native Deep Learning Framework

[![Travis-CI Build Status](https://travis-ci.org/sql-machine-learning/elasticdl.svg?branch=develop)](https://travis-ci.org/sql-machine-learning/elasticdl)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

ElasticDL is a Kubernetes-native deep learning framework built on top of TensorFlow 2.0 that supports fault-tolerance and elastic scheduling.  

TensorFlow has its native distributed computing feature that is fault-recoverable. In the case that some processes fail, the distributed computing job would fail; however, we can restart the job and recover its status from the most recent checkpoint files.

ElasticDL, as an enhancement of TensorFlow's distributed training feature, supports fault-tolerance. In the case that some processes fail, the job would go on running. Therefore, ElasticDL doesn't need to checkpoint nor recover from checkpoints.

The feature of fault-tolerance makes ElasticDL works with the priority-based preemption of Kubernetes to achieve elastic scheduling.  When Kubernetes kills some processes of a job to free resource for new-coming jobs with higher priority, the current job doesn't fail but continues with less resource.

Elastic scheduling could significantly improve the overall utilization of a cluster. Suppose that a cluster has N GPUs, and a job is using one of them. Without elastic scheduling, a new job claiming N GPUs would have to wait for the first job to complete before starting. This pending time could be hours, days, or even weeks. During this very long time, the utilization of the cluster is 1/N. With elastic scheduling, the new job could start running immediately with N-1 GPUs, and Kubernetes might increase its GPU consumption by 1 after the first job completes.  In this case, the overall utilization is 100%.

The feature of elastic scheduling of ElasticDL comes from its Kubernetes-native design -- it doesn't rely on Kubernetes extensions like Kubeflow to run TensorFlow programs; instead, the master process of an ElasticDL job calls Kubernetes API to start workers and parameter servers; it also watches events like process/pod killing and reacts to such events to realize fault-tolerance.

In short, ElasticDL enhances TensorFlow with fault-tolerance and elastic scheduling in the case that you have a Kubernetes cluster. We provide a tutorial showing how to set up a Kubernetes cluster on Google Cloud and run ElasticDL jobs there.  We respect TensorFlow's native distributed computing feature, which doesn't require specific computing platforms like Kubernetes and allows TensorFlow running on any platform.

For development guide, please refer to [this document](elasticdl/README.md).

For running ElasticDL jobs in Google Kubernetes Engine, please check out [this tutorial](doc/elastic_scheduling.md).
