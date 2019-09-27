# ElasticDL Architecture Design

## Component Architecture

![component_architecture](/doc/figures/component_architecture.jpg)

The master node plays the master role in two aspects.

1. It's the master of the cluster. It manages the lifecycle of the worker node. Start the worker pod, listen to the pod event and relaunch the killed worker if necessary.
2. It's the master of the model training process.
   * Shard the training/evaluation data
   * Generate the training/evaluation task from the sharded data
   * Aggregate the gradients reported from the workers
   * Update the model and save the checkpoint

## Distributed Training

![distributed_training_sequence](/doc/figures/distributed_training_sequence.jpg)

## Dynamic Data Sharding

The distributed execution of ElasticDL is data based, not graph based. Each worker holds the whole graph definition of the model. Different shards of data are dispatched to different workers. As a result, while the worker gets the task containing the data shard index, it would be important to read the data content of this shard efficiently from the data storage.

Elastic is a key feature of ElasticDL. A worker can join and left at any time and the entire job still keeps running.

At the start of an epoch, master node splits the entire data set into multiple shards and then generate a list of task. Each task corresponds to a shard of data. At this time, each shard doesn't have a owner. The worker pulls a task(aka a shard of data) at runtime and the master assign the task to the worker.

![dynamic_data_sharding](/doc/figures/dynamic_data_sharding.png)

## Data IO Pipeline

Data IO pipeline for elasticdl involve reading data from [RecordIO](https://github.com/elasticdl/recordio) file, making data generator for tf.data.Dataset and parsing features and label by dataset_fn user defined (see <em> Figure 2</em>).

<center>
    <img src="figures/data_io_pipeline.jpg" height="400" width="450">
    <br>
    <div style="
    display: inline-block;
    color: #999;
    padding: 2px;"><em>Figure 2 </em>. elasticdl data IO pipeline</div>
</center>

After worker is launched, the worker will send request to get task from master. Each task contains record index range [m, m+N) which can locate records in RecordIO file. DataReader read N records from RecordIO file by task index rane and yield each record to create a generator. Then worker will perform the following steps to comsume record data in generator:

1. Create a dataset by [tf.data.Dataset.from_generator](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator).
2. Convert dataset by dataset_fn user defined to generate features and label.  
3. Calculate gradients of trainable variables for train task and predictions of samples for eualuation task.
4. Send calculation result to master.
5. Send task execution status to master after completing all records for the task.
