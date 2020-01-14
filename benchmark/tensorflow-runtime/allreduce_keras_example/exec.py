# allreduce training example
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import os, time

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
tfds.disable_progress_bar()

tf.compat.v1.disable_eager_execution()

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_SAMPLES = 60000
NUM_WORKERS = int(os.environ["NUM_WORKERS"])
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
NUM_EPOCH = 10


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

datasets, info = tfds.load(name='mnist',
                           with_info=True,
                           as_supervised=True)

train_datasets_unbatched = datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE).repeat()

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
      metrics=['accuracy'])
  return model

train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
start = time.time()
with strategy.scope():
  multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x=train_datasets, epochs=NUM_EPOCH, steps_per_epoch=NUM_SAMPLES // GLOBAL_BATCH_SIZE)
end = time.time()
print("time consumption %.2f" % (end-start))