# ElasticDL Parameter Server Design Doc

## Overview
Currently there is one parameter server (PS) co-existed with the master. In order to support multiple PS nodes and PS fault tolerance, we need to seperate PS from the master. Besides a KV store for model variables and embedding tables, PS should also support model variable update and sparse embedding table update using gradients. For a sparse embedding table update, it will update a number of embedding vectors in the embedding table.

The master will create *N* PS pods, with *N* as the number of PS nodes. Each model variable and embedding vector has a corresponding PS node. Thus, every PS node has a subset of model variables and embedding tables.

The master will monitor the PS pods status similar as it does for the worker pods. In case a PS pod fails, the master will try to relaunch the PS pod. Since PS pods have higher priority than worker pods, if there are still some running worker pods, the relaunch will succeed by using either idled or preempted Kubernetes resouces. If the relaunch fails, there are no worker pods left. The Elastic job has to wait for resources for the PS pod and worker pods.

Each worker has a local copy of model variables. After the master relaunches a PS node, the PS pod can recover model variables from workers. For embedding vectors, PS must create replicas to support fault tolerance. For each PS node *Ni*, it will store *M* replicas in following *M* PS nodes *Ni+1, Ni+2, ... Ni+M*. We should use division remainder for PS node indices with the divided number as *N*. The relaunched PS can recover embedding vectors from one of its replicas. If there are more than *M* continously-indexed PS nodes failing, at least one PS node fails with all of its replica PS nodes. The ElasticDL job has to recover from a recent checkpoint.

## Parameter Server (PS)

## Interactions among Master, PS and Worker
### Master - PS
### Master - Worker
### Worker - PS

## Embedding Replicas in PS
