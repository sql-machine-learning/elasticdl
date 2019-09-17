# Parameter Server Design

## Overview

A typical parameter server(pserver) architecture contains three roles:

- master, calling Kubernetes API to start workers and optionally parameter servers, and scheduling tasks to workers
- pserver, maintaining the global model, and providing parameter pull/push/optimize/checkpoint service
- worker, computing gradients of model parameters using local data and pushing gradients to pservers


Please refer to:

![parameter_server](./images/parameter_server.png)

## Key concepts


### Parameter sharding


We prefer to shard the global model for some reasons:

- A model could be too large to fit in the memory of a process. For example, many recommending and ranking models take very high-dimensional (and sparse) inputs, thus require large embedding tables. In such cases, we'd have to have multiple parameter server instances to hold the global model.

- Even when a single process can hold the global model, we might still shard it onto multiple parameter server instances, who share the both communication and optimization workload.

There are several kinds of parameter to be handled separately:

- Very big embedding table: Embedding table is a collection of <item id, embedding vector> pairs. There is also a mapping from item id to pserver id, so the corresonding embedding vector will be stored at the certain pserver

- Big dense tensor: If the dense tensor parameter exceeds certain size, it will be sliced into several subtensors. The parameter name combining subtensor id will become a unique key, and the value is the subtensor(Please note that the slice tensor operation is zero-copy)

- Small dense tensor: The small dense tensor parameter will be stored at certain pserver wholely


The local model on workers and the global model on the parameter server(s) have the same size. In case that large embedding tables make the model size too big to fit in a single process's memory space, an intuitive solution is to save the model copies on an external storage service like Redis or Memcached.

However, such a solution creates a complex workflow which introduces too many cross-node communications.

We propose an alternative solution that doesn't rely on Redis or Memcachd. Instead, the global model, including the large embedding tables, is sharded across parameter servers, and the local model contains only part of the embedding table -- the small fraction that is required to compute recent minibatch(es).

### Parameter Initialization


Parameter initialization of a very big embedding table is lazy. For example, in online learning, there could be unkown item id in the training data. So, until worker send the unkown item id to pserver, will pserver initialize corresponding embedding vector and send back to worker. This is a `get_or_create` semantic.

Other parameters could be initialized before training.


## Workflow

### Assumption

We apply different solutions to different scenarios.


| scenario| hardware | update stragety | communication  stragety |
| :----  |:----  |:---|:---|
| Recommendation  | CPU|  Async |  Parameter server |
| CV | GPU | BSP| Allreduce|
| NLP | GPU | BSP| Allreduce |


Following is the assumption under recommendation scenario:

- only support asynchronous SGD
- only support worker failover, do not support pserver failover


**Note**

Allreduce failover and pserver failover will be discussed on other design docs.

### Master

- responsible for creating/deleting/scheduling pserver and worker
- define parameter sharding strategy, and generate unique key for each parameter(or sharded parameter)
- define the hash strategy from parameter key to pserver id 

Workflow:


1. create several pservers and workers according to user's configuration
2. generate parameter sharding strategy based on model definition provided by user
3. initialize parameter, and send to pservers
4. trigger pservers and workers and start training
5. monitor cluster, and increase or decrease worker according to priority


### Pserver

- provide kv store service, worker could push/pull <key, value> to the pserver
- provide optimize service, apply gradients to parameters and update back to kv store


Workflow:

1. wait for workers pushing <key, gradient> to pserver
2. get the key and query kv store to get corresponding parameter
3. call optimizer, apply <key, gradient> to <key, parameter>
4. update <key, parameter> back to kv store
5. save model checkpoint to disk periodially


**Note**

We could implement the kv store by ourselves, or we could use some already solution, such as redis.

### Worker

- define forward/backward computation proccess
- define dataloader module

Workflow:

1. before forward layer starting, send key to pserver, and pull <key, parameter> back from pserver
2. start forward layer computation
3. after backward layer finishing, generate <key, gradient>
4. push <key, gradient> to pserver


There is also another kind of worker who does an evalution job. It define its own evalution computation process and dataloader of validation data. It pulls <key, parameter> from pserver, but never push data to pserver.


**Note**

There could be some same item id in a minibatch data. So some gradient vector of embedding table will have the same item id. We need to sum these gradient before pushing to pserver.


### Delayed model updating

In order to reduce the communication overhead between workers and pservers, we propose a stragety called delayed model updating.

A worker runs a round of forward/backward computation using its local model, and keep the gradients in its local. After several rounds finishing, it push gradients to pservers.

**Note**

Since local model is only a part of global model, in some circumstance, worker still has to pull embedding vector parameter from pservers if there exits unknow item ids in a minibatch data.

In async mode, this will lead to relative newer embedding part parameters, but relative older other part parameters.



