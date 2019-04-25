# ElasticDL Client: Submit EDL job to minikube from laptop.

## Prepare Environment

To submit EDL job to minikube, make sure minikube is started locally.

```bash
sudo minikube status
```

Start a minikube dashboard may help you check the pod status easily.

```bash
sudo minikube dashboard
```


## Install ElasticDL client package
```bash
curl -o elasticdl-0.0.1-py3-none-any.whl http://dl-alipay-hz1.cn-hangzhou.oss.aliyun-inc.com/elasticdl-0.0.1-py3-none-any.whl
pip install elasticdl-0.0.1-py3-none-any.whl
```


## Write keras model and use EDL client to submit job

```bash
import tensorflow as tf
import elasticdl as edl

class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

    @staticmethod
    def input_shapes():
        return (1, 1)

    @staticmethod
    def input_names():
        return ['x']

    @staticmethod
    def loss(outputs, labels):
        return tf.reduce_mean(tf.square(outputs - labels['y']))

    @staticmethod
    def input_fn(records):
        x_list = []
        y_list = []
        # deserialize
        for r in records:
            parsed = np.frombuffer(r, dtype='float32')
            x_list.append([parsed[0]])
            y_list.append([parsed[1]])
        # batching
        batch_size = len(x_list)
        xs = np.concatenate(x_list, axis=0)
        xs = np.reshape(xs, (batch_size, 1))
        ys = np.concatenate(y_list, axis=0)
        ys = np.reshape(xs, (batch_size, 1))
        return {'x': xs, 'y': ys}

    @staticmethod
    def optimizer(lr=0.1):
        return tf.train.GradientDescentOptimizer(lr)

if __name__ == '__main__':
    edl.run(TestModel, train_data_dir='/data/mnist/train')
```

## Check the pod status

```bash
kubectl get pods
kubectl logs ${pod_name}
```
