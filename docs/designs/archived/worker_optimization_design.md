# ElasticDL Worker Training Performance Optimization 
This document describes the design for performance optimization for ElasticDL worker training process.

## Current worker training pipeline (@July 24, 2019)
```
while True:
    task = get_task()
    if no task, break
    while True:
        minibatch = get_batch()
        if no batch, break
        features, labels = input_fn(minibatch)
        get_model()
        if task is training:
            loss = compute_loss(features, labels)
            report_gradient(loss)
        elif task is evaluation:
            metrics = compute_metrics(features, labels)
            report_evaluation_metrics(metrics)
        elif task is prediction:
            predict = compute_predictfeatures)
            report_prediction_outputs(predict) 
    report_task_result()
```

There are 4 types of ElasticDL jobs:

1. Training-only
2. Training-with-evaluation
2. Evaluation-only
3. Prediction-only

## Test setting
Here, we will concentrate on the performance of the training-only case.


All tests are done with CIFAR10 training data. The timing-related arguments used in the test are:

```
  --num_epochs=1
  --master_resource_request="cpu=2,memory=2048Mi,ephemeral-storage=5000Mi"
  --worker_resource_request="cpu=4,memory=2048Mi,ephemeral-storage=5000Mi,gpu=1"
  --minibatch_size=128
  --num_minibatches_per_task=32
  --num_workers=1
  --grads_to_wait=1
  --model_def=cifar10_functional_api.cifar10_functional_api.custom_model
```

## Current perf for training-only job
The total training time and some break-down timing (in seconds) are below:

Total Time | get_batch | input_fn | compute_loss | get_model | report_gradient
---|---|---|---|---|---
51.0 | 6.7 | 17.9 | 14.8 | 5.0 | 5.0

(6.7 + 17.9) / 51.0 = 48% time is spent on training data read and preprocessing (`get_batch` + `input_fn`).

14.8 / 51.0 = 29% time is on model compute (`compute_loss`).

(5.0 + 5.0) / 51.0 = 20% time is on PS related ops (`get_model` + `report_gradient`)

## Optimization Strategy

### Data Read and Preprocessing
Data Read and preprocessing (`get_batch` and `input_fn`) are serialized with the model compute and PS-related ops within one thread. They can be processed asynchronously  and in parallel, so that their processing time is overlapping with the other parts of the training pipeline.

We can use TensorFlow Dataset for its asynchronization and C++ level multithreading.

One way is to generate a dataset for each task after `get_task`. But the dataset will get stuck whenever `get_task` is needed, and there is overhead in creating a new dataset.

A better way is to use a shared dataset, which will call `get_task` to get new training data when it runs out the data.
The user needs to provide `dataset_fn` instead of `input_fn`. `dataset_fn` would take a RecordIO dataset as input, decode and preprocessing the data as needed.


For example, instead of:

```
def input_fn(records):
    feature_description = {
        "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }
    image_list = []
    label_list = []
    for r in records:
        # deserialization
        r = tf.io.parse_single_example(r, feature_description)
        label = r["label"].numpy()
        image = r["image"].numpy()
        # processing data
        image = image.astype(np.float32)
        image /= 255
        label = label.astype(np.int32)
        image_list.append(image)
        label_list.append(label)

    # batching
    batch_size = len(image_list)
    images = np.concatenate(image_list, axis=0)
    images = np.reshape(images, (batch_size, 32, 32, 3))
    images = tf.convert_to_tensor(value=images)
    labels = np.array(label_list)
    return ({"image": images}, labels)
```

Define `dataset_fn` instead:

```
def dataset_fn(dataset):
    def _parse_data(record):
        feature_description = {
            "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }
        r = tf.io.parse_single_example(record, feature_description)
        label = r["label"]
        image = r["image"]
        return image, label

    dataset = dataset.map(_parse_data)
    dataset = dataset.map(
        lambda x, y: (
            {"image": tf.math.divide(tf.cast(x, tf.float32), 255.0)},
            y,
        )
    )
    return dataset
```

### Implementation

For training-only, we can use a shared dataset as below:

```       
def data_generator(self):
    task = get_task()
    if no task, break
    for r in recordio_data(task):
        yield r

# Create recordio dataset
recordio_dataset = tf.data.Dataset.from_generator(
    data_generator, (tf.string), (tf.TensorShape([]))
    
# apply dataset_fn
dataset = dataset_fn(recordio_dataset)

# batching
dataset = dataset.batch(minibatch)

# training loop
for d in dataset:
    features = d[0]
    labels = d[1]
    get_model()
    loss = compute_loss(features, labels)
    report_gradient(loss)
    if a task finishs:
        report_task_result()
```

To support the full functionalities of the worker (training-only, training-with-evaluation, evaluation-only, predict-only), we need to keep track of the pending tasks to get:

1. the current task type(training, evaluation, or predict).
2. when a task finishes, so we can `report_task_result`.


### Roadmap for this implementaion
1. Add prerequisite: 
    * The master knows the type of the job from user provided arguments. Add an extra argument for the job type in worker initialization. 
    * Add `dataset_fn` in model_zoo examples.
2. Support training-only with a shared dataset.
    * If the job type is not training-only, keep use the legacy code.
3. Add evaluation-only, predict-only support.
4. Support training-with-evaluation.
5. remove the legacy code, including `input_fn`. 


### Model Compute
The TensorFlow eager mode comes at the expense of performance compared with the graph mode. [`tf.function`](https://www.tensorflow.org/beta/tutorials/eager/tf_function) can be used to accelerate the model compute parts(`compute_loss`, `compute_metrics`, `compute_predict`).

#### Implementation
Define `compute_loss`, `compute_metrics`, `compute_predict` as `tf.function`-decorated functions.

### Expected perf after the optimization
The timing below is from a prototype testing.

Total Time | `compute_loss` | `get_model` | `report_gradient`
---|---|---|---
23.8 | 4.2 | 5.4 | 7.6





