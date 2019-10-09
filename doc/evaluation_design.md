# Design for ElasticDL Evaluation

This document describes the design that ElasticDL evaluates the model defined in model_fn.

## The Existing Evaluation Design

The master starts to evaluate model every many steps defined by `eval_steps` or every many seconds defined by `eval_secs`. When a evaluation job starts, the master creates evaluation tasks and inserts them into todo queue at head. Then the worker pulls a evaluation task after completing the current task. Because evaluation tasks are located at the head of todo queue.\
The worker pulls model from parameter serve (pserver) and calculates outputs of records in the evaluation task, then it reports outputs to master. The evaluation sequence is shown as:

<center>
    <img src="figures/evaluate_sequence.svg"
    <br>
    <em>Figure 1 </em>. ElasticDL Evaluate Sequence</div>
</center>

In order to speed up training, the worker creates `tf.data.Dataset` by [tf.data]((https://www.tensorflow.org/guide/data_performance)) API to construct an input pipeline to load data. The flowchart of train loop and evluation in worker is shown as:
<center>
    <img src="figures/train_and_evaluate_flowchart.svg"
    <br>
    <em>Figure 2 </em>. Flowchart of Train and Evaluation in Worker</div>
</center>

While the worker is performing train step N, the Dataset.prefetch thread is preparing the data for step N+1. If step N is the last batch in the task, the generator in the worker for dataset will get next task until the task mode is train. Then worker generate a train batch for step N+1 by Dataset.prefetch. 

There are two problems in the existing evaluation design:

* Problem 1: One worker will get all evaluation tasks in the flowchart shown as <em>Figure 2</em>.\
As shown in the 10th step of <em> Figure 1.</em>, the master generates evaluation tasks and insert them into the head of todo queue when a evaluation jon starts. When one worker starts to process the last mini-batch in its task, it will get next task until the task mode is train to prepare next mini-batch for Dataset.prefetch. The result is that the worker will get all evaluation tasks before it gets next train task. Meanwhile, the master will remove all evaluation tasks from todo to doing list and other worker can not get any evaluation task. So, there is only one work to serially process evaluation tasks.

* Problem 2: The model version is different between workers to evaluate model on a same validation dataset.\
As shown in <em>Figure 1</em>. The worker 1 gets model from the pserver at 15th step and the worker 2 gets model later at 22nd step for the reason that the worker 2 is slower. However, the pserver has updated the model by gradients from the worker 2 before the worker2 gets the model to evaluate. So records assigned to different worker in a same validation dataset will be evluated with different model verion. 


## Evaluate with Current Model in pserver(s) 
1. The pserver updates model variables if it receives gradient from the worker after a evaluation job starts.\
The solution is same as exiting desigin in ElasticDL which can not resolve Problem 2. In order to resolve Problem 1, the master can divide train and evaluation tasks into two todo lists at 10th step in <em>Figure 1</em>. When evalution starts, the master generates evaluation tasks and inserts those into evluation todo list not train list. Meanwhile the master changes the model mode to evaluation. The worker starts to pull evaluation tasks and inferences outputs when the mode of model it gets is evluation. The pserver updates model variables by the mini-batch gradients from each worker at most once. Because the model mode is evluating when the worker get for next mini-batch. The master will change the model mode to train after receiving all record outputs of evluation tasks. Then worker(s) continues to process training tasks.
<center>
    <img src="figures/train_and_evaluate_flowchart_proposal.svg"
    <br>
    <em>Figure 3 </em>. Proposal Flowchart of Train and Evaluation </div>
</center>

2. Pserver stops to update model variables when a evaluation job starts.\
As shown in <em>Figure 1</em>, pserver will not execute the 17th and 18th steps after a evaluation job starts at the 10th step. So, the model worker 2 get is same as the worker 1. The solution can resolve Problem 2 and also can resolve Problem 1 by solution in <em>Figure 3.</em>. But, the train process must wait all evaluationt tasks have been completed. 

Questions to discuss:
* None of training checkpoints will be evaluted in above two solutions which is different from tf.estimator. tf.estimator will load the lastest checkpoint saved by train to evaluate in a separated pod named evaluator. 
* Should master preserve the model to checkpoint after evaluation?

## Evaluate by Loading Checkpoint to Pserver
Generally, deep learning framework preserves model as checkpoint every many seconds or steps. So pserver can load model from checkpoint for evaluation. There are also two designs for pserver to load model. 

1. The master launches additional pods as evalution pserver.\
When a evaluation job starts, the master will launch additional pods as evaluation pservers and evaluation pservers will load model from checkpoint.\
The worker gets model from evaluation pservers to inference outputs for evluation tasks and reports outputs to master.\
The master will release those pods after receiving all outputs of evalution tasks from worker.\
In this case, evaluation is separated from train. So, some workers can process evaluation tasks and other workers can process train tasks at same time. But more additional pods are needed.

2. Existing pservers stop train task and load model from checkpoint to evaluate.\
When a evaluation job starts, pservers preserve the current model to temporary file and load model from the lastest checkpoint. After evaluation, pservers will restore model from temporary file preserved and continue to train. \
The design of pserver will be complicated in this case. What's more,some workers may be reporting gradients when the master change model to evaluation for psersers and pservers can not update model by those gradients at this moment.


## Introduction of tf.estimator to Evaluate
tf.estimator will launch pods as pservers, workers and evalutor when a distributed train job is submitted with ParameterServerStrategy. The evaluator does not participate in train. tf.estimator decides when to evaluate by [throttle_secs](https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec#throttle_secs). tf.estimator will check whether there are new checkpoints in checkpoint directory every throttle_secs after the last evaluation ends. The evaluator will restore whole model from the lastest checkpoint to evaluate. So, evaluaion of tf.estimator is not distributed and not fault-tolerant.

## Summary:
1. If difference with gradients of several mini-batches of model between workers to evaluate is acceptable, the first solution in `Evaluate with Current Model in Pserver` can be adopted. Otherwise，the master should send stop signal to pservers to stop update model after a evaluation job starts.
2. The solution to evaluate by loading checkpoint to pservers is same as the design of tf.estimator. But more additional pods are needed or more complicated design of pserver are needed.

