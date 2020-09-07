# Support Customizing the Training Loop Using AllReduce

This document describes the design for supporting customizing
the training loop when users use AllReduce distribution strategy.

## Motivation

Now, users need to define the forward computation, loss function,
optimizer and dataset function in ElasticDL. ElasticDL provides
the training loop with those definitions. ElasticDL only supports
Keras API to define the forward computation. It maybe not flexible for
users to define complex models in CV or NLP. Sometimes, users need to
customize the training loop to control the model iteration. For example,
For example, given a pre-trained network, users might only want to
optimize a new set of output weights.

If ElasticDL supports customizing the training loop, users can use different
deep learning library (e.g. TensorFlow, Pytorch) to define their training loop.
ElasticDL only need to provide dynamic data partitioning for dataset and
elastic AllReduce to merge gradients across workers.

## The Training Loop of AllReduce

The training loop of AllReduce contains two steps:

- Dataset to read data from disk and convert the data to tensor.
- Calculate gradients and update the model.

To support elastic training, the worker should read data from disk
to create dataset according to the data task assigned by the master.
So, ElasticDL can create a dataset for users and users only need to
get batch data from the dataset to do forward and backward computation.

```python
def train(dataset):
    for batch_data in dataset:
        train_step(batch_data)
```

Using AllReduce distribution strategy, users need to merge gradients
across workers before updating the local model during each step.

```Python
def train_step(batch_data):
    loss = forward(batch_data)
    gradients = backward(loss)
    gradients = allreduce(gradients)
    upate_model(gradients)
```

To support elastic training, the `allreduce` should be able to
complete gradient combination even if the number of worker changes.

## Elastic AllReduce to Merge Gradients in ElasticDL

ElasticDL AllReduce is based on Horovod. Using Horovod, users only need to
use Horovod APIs to wrap optimizer or tape to execute AllReduce in their
training loop.

```python
tape = hvd.DistributedGradientTape(tape)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(
    zip(grads, model.trainable_variables)
)
```

We can develop a decorator to wrap the gradient combination process
using AllReduce. The decorator can support fault-tolerance and
elastic training. If AllReduce fails, the decorator will query the
master for a new AllReduce ring and retry to combine gradients
across alive workers. And the decorator also queries the master
for a new ring periodically in case that the master add new workers.

```python

class ElasticAllReduceManager(object):
    def elastic_run(self, func):
        def wrapper(*args, **kwargs):
            # In case where the master add new workers
            if ITER_STEP % STEPS_TO_CHECK_RING == 0:
                init_horovod_if_needed()

            for i in range(MAX_ALLREDUCE_RETRY_COUNT):
                # Retry to allreduce if failed.
                try:
                    func(*args, **kwargs)
                    report_batch_finished()
                except HorovodInternalError:
                    init_horovod_if_needed()

                ITER_STEP += 1
        return wrapper

    def broadcast(self):
        broadcast_model()
        broadcast_optimizer()

    def set_broadcast_object(self, broadcast_obj):
        ...

```

## Examples of Different Libraries Using ElasticDL AllReduce

Users only need to make the following changes to their training loop:

- Use the decorator of elastic training to wrap the code of gradient
combination using Horovod.
- Set objects to broadcast like model and optimizer. Because, one alive
worker should broadcast its model and optimizer to other workers if
AllReduce fails.

### TensorFlow 1.x

Using Tensorflow 1.x, we use `tf.Session` to execute the forward and backward
computation. The training function definition likes the following
code snippets.

```python
def allreduce(session, train_op):
    """Users should wrap the backward computation using ElasticDL
    """
    sess.run(train_op, feed_dict=feed_dict)


def train(dataset, elastic_manager):
    dataset_iter = dataset.make_one_shot_iterator()
    features, labels = dataset_iter.get_next()

    loss = forward(features, labels)

    global_step = tf.train.get_or_create_global_step()

    lr = tf.Variable(base_lr * hvd.size())
    optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = hvd.DistributedOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.Session(config=config) as session:
        # ElasticDL provides ElasticBroadcastObject to set broadcast objects
        elastic_manager.set_session(session)

        session.run(initializer)
        step = 0
        while True:
            # Users should wrap their forward and backward computation
            # using ElasticDL decorator
            elastic_allreduce = elastic_manager.elastic_run(allreduce)

            elastic_allreduce(session, train_op)
            loss = sess.run(loss, feed_dict=feed_dict)
            if step % 20 == 0:
                logging.info("loss = {}".format(loss))
            step += 1
```

### TensorFlow 2.x

```python
def allreduce(model, optimizer, features, labels):
     with tf.GradientTape() as tape:
        outputs = model.call(features, training=True)

        loss = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=outputs, labels=labels
            )
        )
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss, model.trainable_variables)
    # Take care of the order of grads and vars if worker modifies
    # `_non_embed_vars` during training.
    optimizer.apply_gradients(
        zip(grads, model.trainable_variables)
    )

def train(dataset, elastic_manager):
    """ ElasticDL will call the function to execute the training loop

    Arguments:
        dataset: tf.data.Dataset which initialized by ElasticDL
    """
    inputs = tf.keras.Input(shape=(28, 28), name="image")
    outputs = Conv(10)(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

    optimizer = tf.optimizers.SGD(lr)

    # Set object to broadcast
    elastic_manager.set_model(model)
    elastic_manager.set_optimizer(optimizer)

    for step, (features, labels) in enumerate(dataset):
        elastic_allreduce = elastic_manager.elastic_run(elastic_manager)
        elastic_allreduce(model, optimizer, features, labels)
        if step % 20 == 0:
            logging.info("Step = {}, loss = {}".format(step, loss))

```

### PyTorch

```python
import torch
import horovod.torch as hvd

def allreduce(optimizer):
    """Users should wrap the backward computation using ElasticDL
    """
    optimizer.step()

def train(dataset, elastic_manager):
    """ ElasticDL will call the function to execute the training loop

    Arguments:
        dataset: tf.data.Dataset which initialized by ElasticDL. We can
        use eager execution to fetch batch data from the dataset for PyTorch.
    """
    model = ...

    optimizer = optim.SGD(model.parameters(), lr * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)

    # Set object to broadcast
    elastic_manager.set_model(model)
    elastic_manager.set_optimizer(optimizer)

    for features, labels in dataset:
        optimizer.zero_grad()
        output = model(features)
        loss = F.nll_loss(output, labels)
        loss.backward()
        elastic_allreduce = elastic_manager.elastic_run(allreduce)
        elastic_allreduce(optimizer)

    if step % 20 == 0:
        logging.info("Step = {}, loss = {}".format(step, loss))
```
