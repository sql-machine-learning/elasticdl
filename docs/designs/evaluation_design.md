# Design for ElasticDL Evaluation during Training

The document describes the ElasticDL evaluation process design.

## The Existing Evaluation Design

The master starts to evaluate model every many steps defined by `eval_steps` or every many seconds defined by `eval_secs`. When an evaluation job starts, the master creates evaluation tasks and inserts them into todo queue at head. Then the worker pulls an evaluation task after completing the current task. Because evaluation tasks are located at the head of todo queue. The worker pulls model from parameter server (pserver) and calculates model prediction outputs with records in the evaluation task, then it reports the outputs to master. The evaluation sequence is shown as:

![evaluate_sequence](/docs/images/evaluate_sequence.svg) \
 <em>Figure 1</em>. ElasticDL Evaluate Sequence

To speed up training, the worker creates `tf.data.Dataset` by [tf.data]((https://www.tensorflow.org/guide/data_performance)) API to construct an input pipeline to load data. The flowchart of training loop and evaluation in worker is shown as:

![evaluate_flowchart](/docs/images/train_and_evaluate_flowchart.svg)\
<em>Figure 2</em>. Flowchart of Train and Evaluation in Worker

While the worker is performing train step N, the Dataset.prefetch thread is preparing the data for step N+1. If step N is the last batch in the task,  the generator in the worker for Dataset will keep pulling the continuous evaluation tasks until the task type is training. Then worker generates a train batch for step N+1 by Dataset.prefetch. 

There are two problems in the existing evaluation design:

* Problem 1: One worker will get all evaluation tasks in the flowchart shown as <em>Figure 2</em>.\
As shown in the 10th step of <em> Figure 1.</em>, the master generates evaluation tasks and insert them into the head of todo queue when an evaluation job starts. When one worker starts to process the last mini-batch in its task, it will get next task until the task type is training to prepare next mini-batch for Dataset.prefetch. The result is that the worker will get all evaluation tasks before it gets the next training task. Meanwhile, the master will remove all evaluation tasks from todo to doing list and other workers can not get any evaluation task. As a result, the evaluation speed is low.

* Problem 2: 
For current ElasticDL, workers may use the embedding vectors from different model versions for evaluation. For non-embedding variables, workers will use the same version from checkpoint.\
As shown in <em>Figure 1</em>. The worker 1 gets model from the pserver at the 15th step. But the worker 2 is slow, so it get the model later at 25th step. However, the pserver has updated the model parameters by gradients from the worker 2 before the worker 2 gets the model to evaluate. The model for evaluation is not consistent. The variable is the same, but the embedding vectors are always changing. So records assigned to different workers in the same validation dataset will be evaluated with different model versions. 


## Proposal to Evaluation Design
### Evaluate with Current Model in Pserver 
1. The pserver updates model variables if it receives gradients from the worker after an evaluation job starts.\
The solution is the same as existing design in ElasticDL which can not resolve Problem 2. To resolve Problem 1, the master can divide train and evaluation tasks into two todo lists at the 10th step in <em>Figure 1</em>. When evaluation starts, the master generates evaluation tasks and inserts those into the evaluation todo list not the train list. Meanwhile, the master needs to tell all pservers to change the model mode to evaluation.  The worker starts to pull evaluation tasks and inferences outputs when the mode of model it gets is evaluation. The pserver updates model variables by the mini-batch gradients from each worker at most once. Because the model mode is evaluating when the worker gets for next mini-batch. The master will change the model mode to train after receiving all record outputs of evaluation tasks. Then worker(s) continues to process training tasks.

![evaluate_flowchart_proposal](/docs/images/train_and_evaluate_flowchart_proposal.svg)\
<em>Figure 3</em>. Proposal Flowchart of Train and Evaluation

2. Pserver stops to update model variables when an evaluation job starts.\
As shown in <em>Figure 1</em>, pserver will not execute the 18th and 19th steps after an evaluation job starts at the 10th step. So, the model worker 2 get is the same as the worker 1. The solution can resolve Problem 2 and also can resolve Problem 1 by solution in <em>Figure 3.</em>. But, the training process must wait all evaluation tasks have been completed. 

Questions to discuss:
* None of training checkpoints will be evaluated in above two solutions which are different from tf.estimator. tf.estimator will load the last checkpoint saved by train to evaluate in a separated pod named evaluator. 
* Should master preserve the model to checkpoint after evaluation?

### Evaluate by Loading Checkpoint
Generally, deep learning frameworks preserve model as checkpoint every many seconds or steps. So pserver can load the model from checkpoint for evaluation. There are also two designs for pserver to load model. 

1. The master launches additional pods as evaluation pserver.\
When an evaluation job starts, the master will launch additional pods as evaluation pservers and evaluation pservers will load the model from checkpoint.The worker gets a model from evaluation pservers to inference model outputs for evaluation tasks and reports outputs to master.The master will release those pods after receiving all outputs of evaluation tasks from the worker. In this case, evaluation is separated from train. So, some workers can process evaluation tasks and other workers can process train tasks at the same time. But more additional pods are needed. What's more, the master can launch a new evaluation-only ElasticDL job to evaluate checkpoint model. The new job has independent master, workers and pservers separated from training job.

2. Existing pservers stop train task and load model from checkpoint to evaluate.\
When an evaluation job starts, pservers save the current model to the temporary file and load model from the last checkpoint. After evaluation, pservers will restore the model from temporary file preserved and continue to train. The design of pserver will be complicated in this case. What's more, some workers may be reporting gradients when the master change model to evaluation for pservers and pservers can not update model by those gradients at this moment. So, this method may be not good choice for high cost.


## Summary:
1. If the inconsistency with gradients of several mini-batches  between model variables and embedding vectors to evaluate is acceptable, the first solution in `Evaluate with Current Model in Pserver` can be adopted. Otherwise，the master should send stop signals to pservers to stop update model after an evaluation job starts.
2. The solution to evaluate by loading checkpoint to pservers is the same as the design of tf.estimator. But more additional pods are needed or more complicated design of pserver are needed. 
3. During training loop, we can allow some inconsistencies in stand-alone evaluation job but ensure correct and precise evaluation results when doing checkpoint evaluation job.


## Reference
### Introduction to tf.estimator Evaluation Process
tf.estimator will launch pods as pservers, workers and evaluator when a distributed train job is submitted with ParameterServerStrategy. The evaluator does not participate in training process. tf.estimator decides when to evaluate by [throttle_secs](https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec#throttle_secs). tf.estimator will check whether there are new checkpoints in checkpoint directory every throttle_secs after the last evaluation ends. The evaluator only has one instance and it will restore the whole model from the last checkpoint. So, the evaluation of tf.estimator is not distributed and not fault-tolerant.
