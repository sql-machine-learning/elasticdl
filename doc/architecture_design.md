# ElasticDL Architecture Design

## Component Architecture

![component_architecture](/doc/figures/component_architecture.jpg)

ElasticDL uses the master-worker architecture. The master node plays the master role in two aspects.

1. It's the master of the cluster. It manages the lifecycle of the worker pod, starts the worker pod, listens to the pod event and relaunches the terminated worker pod if necessary.
2. It's the master of the model training process.
   * Partition the training/evaluation data into mutiple shards.
   * Generate the training/evaluation tasks from the data shards.
   * Aggregate the gradients reported from the workers.
   * Update the model variables and save the checkpoint if necessary.

## Distributed Training

![distributed_training_sequence](/doc/figures/distributed_training_sequence.jpg)

## Dynamic Data Sharding

**Elastic** is the key feature of ElasticDL. A worker can join and left at any time and the entire job still keeps running.

The distributed execution of ElasticDL is data based, not graph based. Each worker holds the whole graph definition of the model. Different shards of data are dispatched to different workers. Master doesn't care which worker reports the gradients, it just care how many gradients are reported for the model version. In this way, add or remove a worker won't interrupt the training process.

At the start of an epoch, master node partitions the entire data set into multiple shards and then generate a todo list of task. Each task corresponds to a shard of data.\
At the start point, each data shard doesn't have a owner.\
The worker pulls a task (aka. a shard of data) at runtime and the master assigns the task to this worker. And then move this task to doing list.\
After processing this task and reports the result, the worker will pull the next task.\
If the worker is preempted while processing the assigned task, master will recover the unifinished tasks and insert them back into todo list.

![dynamic_data_sharding](/doc/figures/dynamic_data_sharding.png)

The worker gets the task containing the data shard index (contains filename, startIndex, endIndex), it would be important to read the data content of this shard efficiently from the data storage. In order to reach the IO efficiency, We choose the [RecordIO](https://github.com/elasticdl/recordio) data format for the input data.

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
