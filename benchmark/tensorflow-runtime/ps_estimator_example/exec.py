from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import os, sys, json, logging, time

strategy = tf.distribute.experimental.ParameterServerStrategy()
tfds.disable_progress_bar()
logging.basicConfig(stream=sys.stdout)

BUFFER_SIZE = 10000
BATCH_SIZE = 64
LEARNING_RATE = 0.1
NUM_SAMPLES = 60000
NUM_WORKERS = int(os.environ["NUM_WORKERS"])
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
NUM_EPOCH = 10


def input_fn(mode, input_context=None):
  datasets, info = tfds.load(name='mnist',
                                with_info=True,
                                as_supervised=True)
  mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                   datasets['test'])

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  if input_context:
    mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)

  return mnist_dataset.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(NUM_EPOCH)


def model_fn(features, labels, mode):
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  logits = model(features, training=False)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'logits': logits}
    return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

  optimizer = tf.compat.v1.train.GradientDescentOptimizer(
      learning_rate=LEARNING_RATE)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
  loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=optimizer.minimize(
          loss, tf.compat.v1.train.get_or_create_global_step()))

config = tf.estimator.RunConfig(train_distribute=strategy)

os.mkdir('pstest')
classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='./pstest', config=config)
start = time.time()
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)
end = time.time()
print("time consumption %.2f" % (end-start))