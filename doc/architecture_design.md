# ElasticDL Architecture Design

## Overall Design

![component_architecture](/doc/images/component_architecture.jpg)

## Distributed Task Execution Framework

The master node plays the master role in two aspects.

1. It's the master of the cluster. Take charge of the lifecycle of the worker node. Start the worker pod, listen to the POD event and relaunch the killed worker if necessary.
2. It's the master of the model training process.
   1. Shard the training/evaluation data
   2. Generate the training/evaluation task from the sharded data
   3. Aggregate the gradients reported from the workers
   4. Update the model and save the checkpoint

## Distributed Training

## Dynamic Data Sharding

## Data IO Pipeline

## Open Questions
