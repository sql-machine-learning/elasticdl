# Parameter Server Design

## Overview and key concepts

### Overview

A typical parameter server architecture contains three roles:

- master
- pserver
- worker


Please refer to:

![parameter_server](./images/parameter-server.png)



### Parameter sharding

We usually shard big parameters of a neural network model to some nodes under a rule, in order to achieve network load banlance. There are several kinds of parameter to be handled seperately:

- Very big embedding table: Embedding table is a collection of <item id, embedding vector> pairs. There is also a mapping from item id to pserver id, so the corresonding embedding vector will be stored at the certain pserver

- Big dense tensor: If the dense tensor parameter exceeds certain size, it will be sliced into several subtensors. The parameter name combines subtensor id will become a unique key, and the value is the subtensor(Please note that the slice tensor operation is zero-copy)

- Small dense tensor: The small dense tensor parameter will be stored at certain pserver wholely

### Parameter Initialization


Parameter initialization of very big embedding table is lazy. For example, in online learing, there could be unkown item id in the training data. So, after worker send the unkown item id to pserver, pserver will initialize corresponding embedding vector and send back to worker. This is a get_or_create semantic.

Other parameters could be initialized before training.


## Implementation

### Assumption

- only support asynchronous SGD
- only support worker failover, do not support pserver

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


We could implement the kv store by ourselves, or we could use some already solution, such as redis.

### Worker

- define forward/backward computation proccess
- define dataloader module

Workflow:

1. before forward layer starting, send key to pserver, and pull <key, parameter> back from pserver
2. start forward layer computation
3. after backward layer finishing, generate <key, gradient>
4. push <key, gradient> to pserver

