# ElasticDL - Kubernetes-native Deep Learning System with Elastic Scheduling

## Description

In these years, Kubernetes becomes the fundamental infrastructure of machine
learning. Kubeflow project provides some operators to run distributed machine
learning jobs on Kubernetes. It uses gang scheduling policy. For example, a job
requires N pod, it will keep pending if there are N-1 or less available pods. And
during execution, the job will crash if any one of these N instances crashes. As
a result, the job pending time is long and the resouce utilization is insufficient.

ElasticDL implements elastic scheduling policy. A job doesn't need wait for all
the required resource ready and start promptly. A job process can be preempted
by a higher priority job, join later while the resource become sufficient and the
entire job keeps running. It can significantly shorten the job waiting time and
improve the resource utilization.

## Benefits to the Ecosystem

To operators, this talk

To model deveopers, this talk

## Open Source Projects

[ElasticDL](https://github.com/sql-machine-learning/elasticdl)

## Speakers
