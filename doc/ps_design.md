# ElasticDL Parameter Server Design Doc

## Overview
Currently, there is one parameter server (PS) co-existed with the master. In order to support multiple PSs and PS fault tolerance, we need to separate PS from the master. Besides a KV store for model variables and embedding tables, PS should also support model variable update and sparse embedding table update using gradients. For a sparse embedding table update, it will update a number of embedding vectors in the embedding table.

The master will create *N* PSs, with *N* as the number of PSs specified by the user. Each model variable and embedding vector has a corresponding PS. Thus, every PS has a subset of model variables and embedding tables.

The master will monitor PS pods status similar to what it does for worker pods. In case a PS pod fails, the master will try to relaunch it. Since PS pods have higher priority than worker pods, if there are still some running worker pods, the relaunch will succeed by using either idled or preempted Kubernetes resources. If the relaunch fails, there are no worker pods left. The Elastic job has to wait for resources for the PS pod and worker pods.

Each worker has a local copy of the model variables. After the master relaunches a PS, the PS pod can recover model variables from workers. For embedding vectors, PS must create replicas to support fault tolerance. For each PS *P(i)*, it will store *M* replicas in the following *M* PSs *P(i+1 % N), P(i+2 % N), ... P(i+M % N)*. The relaunched PS can recover embedding vectors from one of its replicas. If there are more than *M* continuously-indexed PSs failing, at least one PS fails with all of its replica PSs. The ElasticDL job has to recover from a recent checkpoint.

## Parameter Server (PS)

## Interactions among Master, PS and Worker
### Master - PS
### Master - Worker
### Worker - PS

## Embedding Replicas in PS
