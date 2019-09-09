# Design Doc: Asynchronous SGD

## Motivation
Parameter server (PS) based distributed training uses data parallelism to speed up training. 

We have implemented synchronous SGD in ElasticDL. When PS accumulates `grads_to_wait` gradients from workers, PS averages these gradients and updates the model with the averaged gradients. PS also maintains `model_version` which equals to the number of model updates. Each worker has a local copy of the model. Before a minibatch training step starts, if there is a new `model_version` on PS, the worker will get the new model from PS to replace the local model. After computing the gradients with a minibatch, the worker reports the gradients to PS together with the local model version. When PS receives gradients from a worker, it only accepts the gradients with a model version same as the current PS `model_version`.



This Synchronous SGD ensures model consistency in the price of wasted and blocked computation. 

* Wasted computation: when a worker reports gradients with an outdated model version to PS, PS will reject these gradients. The worker have to get the current model from PS, reuse the minibatch data to train the model again.
* Blocked computation: PS has to use a lock for model update with gradients and model read by workers to ensure model consistency.

Asynchronous SGD can avoid the wasted and blocked computation mentioned above with a relaxed model consistency.

* PS will accept all gradients from workers.
* PS does not use locks and supports concurrent model reads and updates.


## Asynchronous SGD
Let us recall how workers train the model in synchronous SGD. Below is the pseudocode:

```
for minibatch in training_data:
    accepted = False
    while not accepted:
        local_model，model_version = get_model_from_ps()
        gradients = compute_gradient(local_model, minibatch)
        accepted = report_gradient_to_ps(gradients, model_version)  
```

In asynchronous SGD, each worker is training the model in nearly the same way as synchronous SGD. The only difference is that the worker does not need to retrain any minibatch data as PS accepts all gradients. 

```
for minibatch in training_data:
    local_model, model_version = get_model_from_ps()
    gradients = compute_gradient(local_model, minibatch)
    report_gradient_to_ps(gradients, model_version)  
```

PS does not need locks in `GetModel` and `ReportGradient` GRPC services for asynchronous SGD. 

```
def GetModel():
    pb_model = Model()
    for variable in pb_model:
        assign_value(variable, current_variable_value_in_PS)
    return pb_model, PS_model_version
    
def ReportGradient(gradients, version):
    grad_var = zip(gradients, model_variables)
    optimizer.apply_gradient(grad_var)
    PS_model_version.atomic_add(1)
```

### Relaxed Model Consistency
PS can processes multiple GRPC calls `GetModel` and `ReportGradients` concurrently. Thus, there are two kinds of relaxed model consistency.

1. In `GetModel`, during the variable assign loop, there may be `ReportGradient` GRPC service running and updating the variables. Thus, variables in `local_model` in workers may contain values from different model versions. `model_version` from `get_model_from_ps` is just a proximate model version.
2. There may be multiple `ReportGradient` running concurrently. Different model variables may apply these gradients in different orders. 

Also, the concurrent updates to variables in `ReportGradient` may cause some gradients are not applied, as the updates can be overwritten by other concurrent running updates. TensorFlow optimizers have an argument [`use_locking`](https://github.com/tensorflow/tensorflow/blob/ff441191277b7e758deb48e45249fee9e880f2c8/tensorflow/python/training/optimizer.py#L319). If [`use_locking`](https://github.com/tensorflow/tensorflow/blob/ff441191277b7e758deb48e45249fee9e880f2c8/tensorflow/python/training/optimizer.py#L319) is `True`, TensorFlow will use a [lock](https://github.com/tensorflow/tensorflow/blob/11e22c01eb801ff24200afcdce8a03a7cdd2ed3f/tensorflow/core/kernels/training_ops.cc#L528) to prevent concurrent updates to variables.

### Staleness in Asynchronous SGD
In `ReportGradient`, the argument `version` may be smaller than `PS_model_version`. 
Staleness value is the difference between `PS_model_version` and `version`:

```
staleness = PS_model_version - version
```

According to some [researches](https://arxiv.org/abs/1810.03264), this staleness affects the training convergence, and large staleness may result in poor training accuracy. The deeper the model, the more impact of the staleness. Some optimizers such as [SGD](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/SGD) and [Adagrad](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/Adagrad) are more robust to staleness, some optimizers such as other with momentum are very bad with staleness.

[Staleness-aware asychronous SGD](https://arxiv.org/abs/1511.05950) proposes a method to modulate learning rate by the staleness. If the staleness if not 0, this method modulates the learning rate used in the optimizer as:

```
if staleness > 0:
    learning_rate_used = learning_rate / staleness
else:
    learning_rate_used = learning_rate
```

### Stale Synchronous Parallel (SSP)
In the pesudocode for the asynchronous SGD worker, the worker pulls model from PS in every minibatch step. [Stale synchronous parallel (SSP) method](https://dl.acm.org/citation.cfm?id=2999748) uses the strategy that the fastest worker can exceed the slowest one within a predefined staleness threshold. SSP can reduce the number of `get_model_from_ps` calls. The worker training process is:

```
staleness_threshold = predefined_staleness_threshold
local_model, model_version = get_model_from_ps()
local_update_count = 0
for minibatch in training_data:
    gradients = compute_gradient(local_model, minibatch)
    report_gradient_to_ps(gradients, model_version)
    local_update_count += 1
    if local_update_count >= staleness_threshold:
        local_model, model_version = get_model_from_ps()
        local_update_count = 0
    else:
        apply_gradient(local_model, gradients)
```
Althrough the original SSP method uses this strategy in synchronized SGD, we can also adopt SSP strategy in asynchronized SGD to reduce `get_model_from_ps` calls.
Note that in ElasticDL, local models only have non-embedding variables. So in `apply_gradient(local_model, gradients)`, ElasticDL workers only update non-embedding variables.

## Support Asynchronous SGD in ElasticDL

### Change in PS
1. No need to use locks in `GetModel` and `_update_model` in [server.py](../python/master/servicer.py).
2. No need to accumulate gradients in `ReportGradient` in [server.py](../python/master/servicer.py). `ReportGradient` calls `_update_model` directly.
3. Users decide if disabling concurrent variable update by set `use_locking` argument in the optimizer.
4. To support [Staleness-aware asychronous SGD](https://arxiv.org/abs/1511.05950), PS need to modulate the learning rate in the optimizer with the staleness value. PS may have multiple threads running concurrently for model updates with a same optimizer instance. Thus, we cannot modify the learning rate in the optimizer instance. We may modify the learning rate as a callable method, and use a thread local storage `threading.local()` to store the staleness. The callable method uses the staleness value to modulate the learning rate. The optimizer will call this callable method [when it reads the learning rate hyperparameter](https://github.com/tensorflow/tensorflow/blob/e4262fb2fbf1cb33aaea79ff81754d1e92e99af1/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L530).

### Change in Worker
1. No need to retrain with the minibatch data.
2. To support SSP strategy, the worker pulls the model from PS in every `staleness_threshold` minibatch step. Also, the worker needs to update the local model with the computed gradients. model pull/updates do not include embedding variables, as we directly access the embedding vectors in the embedding service.

### Add Arguments for `elasticdl.train`
1. `--use_async, default=False, help="True for asynchronous SGD, False for synchronous SGD"`
2. `--lr_staleness_modulation, default=False, help="If True, master will modulate learning rate with staleness in asynchronous SGD"`
3. `--get_model_frequency, default=1, help="worker will get_model from PS every this steps."`
