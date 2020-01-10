# allreduce training example
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import os

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
tfds.disable_progress_bar()
tf.compat.v1.disable_eager_execution()

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# 将 MNIST 数据从 (0, 255] 缩放到 (0., 1.]
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
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

NUM_WORKERS = int(os.environ["NUM_WORKERS"])
print(NUM_WORKERS)
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
with strategy.scope():
  multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x=train_datasets, epochs=10, steps_per_epoch=60000/GLOBAL_BATCH_SIZE) # 60000 is the size of the training set