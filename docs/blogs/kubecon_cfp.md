# ElasticDL: Kubernetes-native Deep Learning System with Elastic Scheduling

## Description

In these years, Kubernetes becomes the fundamental infrastructure of machine
learning. Kubeflow project provides some operators to run distributed machine
learning jobs on Kubernetes. It uses gang scheduling policy. For example, a job
requires N pod, it will keep pending if there are N-1 or less available pods. And
during execution, the job will crash if any one of these N instances crashes. As
a result, the job pending time is long and the resouce utilization is insufficient.

ElasticDL implements elastic scheduling policy. A job doesn't need wait for all
the required resource ready and start promptly. A job process can be preempted
by a higher priority job, join freely once the resource become sufficient later
and the entire job keeps running. It can significantly shorten the job waiting
time and improve the resource utilization.

## Benefits to the Ecosystem

This talk introduces a system which implements a new policy to schedule deep
learning workloads on Kubernetes - elastic scheduling. Compared with the gang
scheduling policy that kubernetes operators from Kubeflow apply, it can improve
the resource utilization significantly. We add a master to launch, monitor and
manage the worker pods using Kubernetes API and distribute the sharded data
dynamically to them. It doesn't rely on the Kubernetes operators. In this way,
master can understand the internal details of deep learning process and model
structure, and then make more optimized scheduling decision.

For system developers and operators, this talk introduces a elastic scheduling
policy for deep learning jobs to improve the resource utilization.

For model developers, this talk introduces how to develop a distributed TensorFlow
program and submit a training job to Kubernetes. It provides a way to make quick
verfication and iteration on the model and then improve the development efficiency.

What's more, we will share the results of three benchmark experiments to show how
ElasticDL improve the resource utilization and development efficiency.

## Open Source Projects

[ElasticDL](https://github.com/sql-machine-learning/elasticdl)

## Speakers
