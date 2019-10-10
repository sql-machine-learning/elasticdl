# ElasticDL Parameter Server Design Doc
## Parameter Server


## Interactions among Master, PS and Worker

The following events involve interactions among the master, workers and PS:

* The master starts PS.
* Initialization of parameters in PS.
* Relaunch of PS.
* Workers get model parameters from PS.
* Workers push gradients to PS.
* PS reports submodel version to the master
* The master tells PS to save checkpoint.


### The master starts PS
When an ElasticDL task starts, `master.main` is responsible for starting PS. After starting PS, `master.main` starts the master and workers, and tells them the endpoints of PS.

### Initialization of parameters in PS
PS does not have any model variable and mdoel meta info after starting. Model meta info includes dimension of embedding layers, initialization methods of embedding vectors, initialization methods of slot variables in optimizer.

There are two ways for PS to get model variables and model meta info, one is to read from a checkpoint file, one is to obtain them from workers.

When `master.main` starts PS, `master.main` decides how to initialize PS. If `master.main` passes an argument specifying the checkpoint file name to PS, PS reads from the checkpoint. Otherwise, PS does nothing but waiting for the first `get_model` from worker. In the reponse of `get_model` call, PS tells the worker to initialize model, and report model variables and meta info to the PS.

Please Note that the worker only initializes model variables. ElasticDL adopts lazy initialization for embedding vectors. Please refer to "[Workers get model parameters from PS](#Workers-get-model-parameters-from-PS)" section.

### Relaunch of PS
In case a PS pod fails, the master will try to relaunch one PS and it should recover model variables and embedding tables.

For model variables, PS can recover from workers.

For embedding tables, the `master.main` tells PS through in starting command that PS should recover from replica. If there is no replica, PS has to recover from checkpoint.

### Workers get model parameters from PS
Before each forward-pass, workers need to get all model parameters from PS. Currently, workers call function `get_model()` to get parameters.

When workers want to get model parameters from PS, PS may not possess all the embedding vectors needed because ElasticDL adopts lazy initialization for embedding vectors, i.e. iniatializing embedding vectors when they are needed in workers. Thus, if a worker wants to pull some embedding vectors that are not existing in PS, PS will create and initialize these embedding vectors and return their value to the worker.

### Push Gradients
After backward-pass, workers push gradients to PS.

### PS reports submodel version to the master
The master needs to know the model version to decide when to save checkpoint and when to evaluate model. PS regularly reports the version of the submodel it possessed to the master. 

Please note different pserver has different submodel version. The master choose the maximum of these submodel versions as the current model version.

### The master tells PS to save checkpoint
When the master decides to save checkpoint, the master tells all the pservers to save checkpoint. Every pserver saves the submodel it possessed into a separate file.


## Embedding Replicas in PS