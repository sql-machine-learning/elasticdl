# Design Overview for ElasticDL on SQLFlow

## Overview

This is a design doc on integration with [SQLFlow](https://github.com/sql-machine-learning/sqlflow).

### User Interface

#### Training Job Submission

```sql
SELECT
    c1, c2, c3, c4, c5 as class
FROM training_data
TRAIN ElasticDLKerasClassifier
WITH
    model.optimizer = "optimizer",
    model.loss = "loss",
    model.eval_metrics_fn = "eval_metrics_fn",
    model.num_classes = 3,
    model.dataset_fn = "dataset_fn",
    train.shuffle = 120,
    train.epoch = 2,
    train.grads_to_wait = 2,
    train.tensorboard_log_dir = "",
    train.checkpoint_steps = 0,
    train.checkpoint_dir = "",
    train.keep_checkpoint_max = 0,
    eval.steps = 0,
    eval.start_delay_secs = 100,
    eval.throttle_secs = 0,
    eval.checkpoint_filename_for_init = "",
    engine.master_resource_request = "cpu=400m,memory=1024Mi",
    engine.master_resource_limit = "cpu=1,memory=2048Mi",
    engine.worker_resource_request = "cpu=400m,memory=2048Mi",
    engine.worker_resource_limit = "cpu=1,memory=3072Mi",
    engine.num_workers = 2,
    engine.volume = "",
    engine.image_pull_policy = "Never",
    engine.restart_policy = "Never",
    engine.extra_pypi_index = "",
    engine.namespace = "default",
    engine.minibatch_size = 64,
    engine.master_pod_priority = "",
    engine.cluster_spec = "",
    engine.num_minibatches_per_task = 2,
    engine.docker_image_repository = "",
    engine.envs = ""
COLUMN
  c1,
  c2,
  c3,
  c4
LABEL class
INTO trained_elasticdl_keras_classifier;
```

#### Prediction Job Submission

```sql
SELECT
    c1, c2, c3, c4
FROM prediction_data
PREDICT prediction_results_table
WITH
    model.num_classes = 10,
    model.dataset_fn = "dataset_fn",
    predict.checkpoint_filename_for_init = "modelv1.chkpt",
    engine.master_resource_request = "cpu=400m,memory=1024Mi",
    engine.master_resource_limit = "cpu=1,memory=2048Mi",
    engine.worker_resource_request = "cpu=400m,memory=2048Mi",
    engine.worker_resource_limit = "cpu=1,memory=3072Mi",
    engine.num_workers = 2,
    engine.volume = "",
    engine.image_pull_policy = "Never",
    engine.restart_policy = "Never",
    engine.extra_pypi_index = "",
    engine.namespace = "default",
    engine.minibatch_size = 64,
    engine.master_pod_priority = "",
    engine.cluster_spec = "",
    engine.num_minibatches_per_task = 2,
    engine.docker_image_repository = "",
    engine.envs = ""
USING trained_elasticdl_keras_classifier;
```


### Implementation

#### Mapping Extended SQL

The components of the extended SQL defined by SQLFlow are mapped to a ``elasticDLFiller`` struct that looks like the following:

```
type elasticDLFiller struct {
	IsTraining bool
	TrainInputTable    string
	EvalInputTable     string
	PredictInputTable  string
	PredictOutputTable string
	PredictInputModel  string
	OutputShape        int
	InputShape         int
	ModelDir           string
	LabelColName        string
	FeaturesList        string
	TrainClause   *resolvedTrainClause
	PredictClause *resolvedPredictClause
}
```

This ``elasticDLFiller`` struct will be used to fill a template pre-defined to generate the model definition components required
for ElasticDL, such as the model definition using `tf.keras` API, loss, optimizer, `dataset_fn`, etc.

For example, the `dataset_fn` is generated using the `FeaturesList`, `LabelColName`, `InputShape`, `IsTraining`, and `TrainClause` in the ``elasticDLFiller`` struct:

```python
def dataset_fn(dataset, mode, metadata):
    def _parse_data(record):

        def _get_features_without_labels(
            record, label_col_ind, features_shape
        ):
            features = [
                record[:label_col_ind],
                record[label_col_ind + 1 :],
            ]
            features = tf.concat(features, -1)
            return tf.reshape(features, features_shape)

        record = tf.strings.to_number(record, tf.float32)
        features_shape = ({{.InputShape}}, 1)
        labels_shape = (1,)
        {{if .IsTraining}}
        label_col_name = "{{.LabelColName}}"
        if mode != Mode.PREDICTION:
            if label_col_name not in metadata.column_names:
                raise ValueError(
                    "Missing the label column '%s' in the retrieved "
                    "table." % label_col_name
                )
            label_col_ind = metadata.column_names.index(label_col_name)
            labels = tf.reshape(record[label_col_ind], labels_shape)
            return (
                _get_features_without_labels(
                    record, label_col_ind, features_shape
                ),
                labels,
            )
        {{end}}
        return tf.reshape(record, features_shape)

    dataset = dataset.map(_parse_data)

    {{if .IsTraining}}
    if mode != Mode.PREDICTION and "{{.TrainClause.EnableShuffle}}" == "true":
        dataset = dataset.shuffle(buffer_size={{.TrainClause.ShuffleBufferSize}})
    {{end}}

    return dataset
```

Some fields used to generate the above `dataset_fn` are obtained directly from the extended SQL statement. For example, ``FeaturesList`` is obtained
from `SELECT FROM` clause. `LabelColName` is obtained from `LABEL` clause. `TrainClause.ShuffleBufferSize` is obtained from
`train.shuffle` in the `WITH` clause. There are also fields that are obtained indirectly. For example, `InputShape` is inferred from `FeaturesList`.

Note that in the template we currently we hard-coded the types for each column to be ``tf.float32`` in the generated `dataset_fn`. We should infer
this information from the database instead. We also hard-coded other components in the model definition such as ``loss`` and ``optimizer``, these
components should be derived from the model zoo instead.

#### Generate ElasticDL Command

Once we generated the components for the model definition, we can then generate the ElasticDL command to submit the job. 
Below is an example:

```sh
elasticdl train \
--image_base=elasticdl:ci \
--model_zoo=<model-zoo> \
--model_def=<path-to-generated-model-def> \
--loss=<loss-function-name> \
--eval_metrics_fn=<eval-metrics-function-name> \
--training_data=<training-input-table> \
--validation_data=<validation-input-table> \
--num_epochs=2 \
--master_resource_request="cpu=400m,memory=1024Mi" \
--master_resource_limit="cpu=1,memory=2048Mi" \
--worker_resource_request="cpu=400m,memory=2048Mi" \
--worker_resource_limit="cpu=1,memory=3072Mi" \
--minibatch_size=64 \
--num_minibatches_per_task=10 \
--num_workers=2 \
--checkpoint_steps=10 \
--evaluation_steps=15 \
--grads_to_wait=2 \
--job_name=test-iris \
--log_level=INFO \
--image_pull_policy=Never \
--output=<model-output> \
--envs=<env-vars> \
--data_reader_params=<data-reader-params>
```

In the command, ``--model_def`` is the path to the model definition file we generated earlier. Additional arguments
related to model definition such as ``--loss`` and ``--eval_metrics_fn`` are obtained from parameters
with name starting with ``model.``.

The rest of the arguments are derived from the extended SQL, for example:

* ``--model_zoo`` is obtained from `TRAIN` clause.
* ``--training_data`` is obtained from `FROM` clause.
* ``--num_epochs`` is obtained from `train.shuffle` in `WITH` clause.

ElasticDL engine specific arguments such as ``--grads_to_wait`` and ``--num_workers`` are obtained from parameters
with name starting with ``engine.``.

In order to integrate with different databases we support, we pass additional information to the ElasticDL command.

For example, we pass necessary environment variables such as access ID and key for ODPS account to ``--envs``. In addition,
we pass the list of column names that we want to read from ODPS via ``--data_reader_params``.

### Future Work

* Support ``tf.feature_columns`` API via ``COLUMN`` clause.
* Support evaluation job. Evaluation on separate evaluation table is not supported yet in SQLFlow. Please check out
[#675](https://github.com/sql-machine-learning/sqlflow/issues/675) and [#675](https://github.com/sql-machine-learning/sqlflow/issues/674) for details.
* Switch to use intermediate representation for ElasticDL codegen. For details, please see [#1075](https://github.com/sql-machine-learning/sqlflow/issues/1075).
* Support on synchronous call on high level API. For details, please see [#1285](https://github.com/sql-machine-learning/elasticdl/issues/1285).
* Unify model zoos between SQLFlow and ElasticDL and support submitting an ElasticDL job for a model defined in model zoo.
Please see [#22](https://github.com/sql-machine-learning/models/issues/22) and [#1063](https://github.com/sql-machine-learning/sqlflow/issues/1063) for details.
* Currently the only database ElasticDL supports is ODPS. However, we should expose necessary abstractions so ElasticDL
 can fully leverage SQLFlow's functionality to read/write
from different SQL databases.
* Support prediction job and add integration tests on Travis CI.
* Currently we have hard-coded the types for each column to be ``tf.float32`` in the generated `dataset_fn`. We should infer
this information from the database instead.
