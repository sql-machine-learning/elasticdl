# ElasticDL: A Kubernetes-native Deep Learning Framework

[![Travis-CI Build Status](https://travis-ci.org/sql-machine-learning/elasticdl.svg?branch=develop)](https://travis-ci.org/sql-machine-learning/elasticdl)
[![Code Coverage](https://codecov.io/gh/sql-machine-learning/elasticdl/branch/develop/graph/badge.svg)](https://codecov.io/gh/sql-machine-learning/elasticdl)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI Status Badge](https://badge.fury.io/py/elasticdl.svg)](https://pypi.org/project/elasticdl/)

ElasticDL is a Kubernetes-native deep learning framework built on top of TensorFlow 2.0 that supports fault-tolerance and elastic scheduling.

|                          | TensorFlow 1.x graph mode | TensorFlow 2.x eager execution |
|--------------------------|---------------------------|--------------------------------|
| No change to the runtime | Uber Horovod              | ElasticDL (early stage)        |
| Changes the runtime      | TensorFlow ps-based distribution | TensorFlow distribution strategies |

**Note that ElasticDL is still under active development, and we have not extensively tested it in production environments. We open sourced this early-stage project with the hope of encouraging further work on fault-tolerance and elastic scheduling from the community.**

## Main Features

### Elastic Scheduling and Fault-Tolerance

Through Kubernetes-native design, ElasticDL enables fault-tolerance and works with the priority-based preemption of Kubernetes to achieve elastic scheduling for deep learning tasks.

### TensorFlow 2.0 Eager Execution

A distributed deep learning framework needs to know local gradients before the model update. Eager Execution allows ElasticDL to do it without hacking into the graph execution process.

### Minimalism Interface

Given a model defined with Keras API, train the model with a command line.
```bash
elasticdl train --model_def=mnist_functional_api.custom_model --training_data=/mnist/train --output=output
```

### Integration with SQLFlow

ElasticDL will be integrated seamlessly with SQLFlow to connect SQL to distributed deep learning tasks with ElasticDL.

```sql
SELECT * FROM employee LABEL income INTO my_elasticdl_model
```

## Quick Start

Please check out our [step-by-step tutorial](docs/tutorials/get_started.md) for running ElasticDL on local laptop, on-prem cluster, or on public cloud such as Google Kubernetes Engine.

## Background

TensorFlow has its native distributed computing feature that is fault-recoverable. In the case that some processes fail, the distributed computing job would fail; however, we can restart the job and recover its status from the most recent checkpoint files.

ElasticDL, as an enhancement of TensorFlow's distributed training feature, supports fault-tolerance. In the case that some processes fail, the job would go on running. Therefore, ElasticDL doesn't need to checkpoint nor recover from checkpoints.

The feature of fault-tolerance makes ElasticDL works with the priority-based preemption of Kubernetes to achieve elastic scheduling.  When Kubernetes kills some processes of a job to free resource for new-coming jobs with higher priority, the current job doesn't fail but continues with less resource.

Elastic scheduling could significantly improve the overall utilization of a cluster. Suppose that a cluster has N GPUs, and a job is using one of them. Without elastic scheduling, a new job claiming N GPUs would have to wait for the first job to complete before starting. This pending time could be hours, days, or even weeks. During this very long time, the utilization of the cluster is 1/N. With elastic scheduling, the new job could start running immediately with N-1 GPUs, and Kubernetes might increase its GPU consumption by 1 after the first job completes.  In this case, the overall utilization is 100%.

The feature of elastic scheduling of ElasticDL comes from its Kubernetes-native design -- it doesn't rely on Kubernetes extensions like Kubeflow to run TensorFlow programs; instead, the master process of an ElasticDL job calls Kubernetes API to start workers and parameter servers; it also watches events like process/pod killing and reacts to such events to realize fault-tolerance.

In short, ElasticDL enhances TensorFlow with fault-tolerance and elastic scheduling in the case that you have a Kubernetes cluster. We provide a tutorial showing how to set up a Kubernetes cluster on Google Cloud and run ElasticDL jobs there.  We respect TensorFlow's native distributed computing feature, which doesn't require specific computing platforms like Kubernetes and allows TensorFlow running on any platform.

## Development Guide

Please refer to [this document](elasticdl/README.md) for development guide.
