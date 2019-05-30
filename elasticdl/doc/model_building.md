# ElasticDL Model Building
To run an ElasticDL job, the user provides a model file in the job submission, such as [`mnist_functional_api.py`](../python/examples/mnist_functional_api.py) used in this [ElasticDL job submission sample](elastic_scheduling.md#submit-the-first-job-with-low-priority). 

This model file contains a [model](#model) built in Keras and other components required by ElasticDL model training, including [input\_fn](#input_fn), [feature\_columns](#feature_columns), [label\_columns](#label_columns), [loss](#loss), and [optimizer](#optimizer). 

## Model File Components
### model
`model` is a Keras model built using either Tensorflow Keras [functional API](https://www.tensorflow.org/guide/keras#functional_api) or [model subclassing](https://www.tensorflow.org/guide/keras#model_subclassing). 

The following example shows `model` using functional API:

```
inputs = tf.keras.Input(shape=(28, 28, 1), name='img')
x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x, training=True)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.25)(x, training=True)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(10)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
```

Another example using model subclassing:

```
class MnistModel(tf.keras.Model):
    def __init__(self, channel_last=True):
        super(MnistModel, self).__init__(name='mnist_model')
        if channel_last:
            self._reshape = tf.keras.layers.Reshape((28, 28, 1))
        else:
            self._reshape = tf.keras.layers.Reshape((1, 28, 28))
        self._conv1 = tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation='relu')
        self._conv2 = tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation='relu')
        self._batch_norm = tf.keras.layers.BatchNormalization()
        self._maxpooling = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2))
        self._dropout = tf.keras.layers.Dropout(0.25)
        self._flatten = tf.keras.layers.Flatten()
        self._dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self._reshape(inputs)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._batch_norm(x, training=training)
        x = self._maxpooling(x)
        if training:
            x = self._dropout(x, training=training)
        x = self._flatten(x)
        x = self._dense(x)
        return x

model = MnistModel()
```
### input_fn

```
input_fn(records)
```
`input_fn` is a function that takes a batch of training data `records` as input, preprocesses `records` as needed, and returns the batched `model_inputs` and `labels` as a pair. `model_inputs` is a dictionary of tensors, which will be used as [model](#model) input by applying the schema specified by [`feature_columns`](#feature_columns). `labels` will be used as an input argument in [loss](#loss).

Example:

```
def input_fn(records):
    image_list = []
    label_list = []
    # deserialize
    for r in records:
        get_np_val = (lambda data: data.numpy() if isinstance(data, EagerTensor) else data)
        label = get_np_val(r['label'])
        image = np.frombuffer(get_np_val(r['image']), dtype="uint8")
        image = np.resize(image, new_shape=(28, 28))
        image = image.astype(np.float32)
        image /= 255
        label = label.astype(np.int32)
        image_list.append(image)
        label_list.append(label)

    # batching
    batch_size = len(image_list)
    images = np.concatenate(image_list, axis=0)
    images = np.reshape(images, (batch_size, 28, 28))
    labels = np.array(label_list)
    return ({'image': images}, labels)
```

### feature_columns
```
feature_columns()
```

`feature_columns` is a function that returns a list of [`tf.feature_column.numeric_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column). The number of `tf.feature_column.numeric_column` in the list equals the number of [`model`](#model)'s inputs. `feature_columns` defines the schema between `model_input` from [`input_fn`]($input_fn) and [model](#model)'s inputs.
The required arguments in `tf.feature_column.numeric_column` are `key`, `dtype`, and `shape`.

For example, the following `feature_columns` example specifies that `model` has one input, which is `model_inputs['image']`, with data type as `tf.dtypes.float32` and shape as `[1, 28, 28]`.

```
def feature_columns():
    return [tf.feature_column.numeric_column(key="image",
        dtype=tf.dtypes.float32, shape=[1, 28, 28])]
```

### label_columns
```
label_columns()
```

`label_columns` is a functions that returns a list of [`tf.feature_column.numeric_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column), which represents the schema for `labels` from [`input_fn`]($input_fn).

Example:

```
def label_columns():
    return [tf.feature_column.numeric_column(key="label",
        dtype=tf.dtypes.int64, shape=[1])]
```

### loss
```
loss(output, labels)
```
`loss` is the loss function used in ElasticDL training.

Arguments:

- output:  [model](#model)'s output.

- labels: `labels` from [`input_fn`](#input_fn).

Example:

```
def loss(output, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels))
```

### optimizer
```
optimizer()
```
`optimizer` is a function returns a [`tf.train.Optimizer`](https://www.tensorflow.org/api_docs/python/tf/train/Optimizer).

Example:

```
def optimizer(lr=0.1):
    return tf.train.GradientDescentOptimizer(lr)
```

## Model Building Examples
### [MNIST model using Keras functional API](../python/examples/mnist_functional_api.py)
### [MNIST model using Keras model subclassing](../python/examples/mnist_sublcass.py)
### [CIFAR-10 model using Keras functional API](../python/examples/cifar10_functional_api.py)
### [CIFAR-10 model using Keras model subclassing](../python/examples/cifar10_sublcass.py)