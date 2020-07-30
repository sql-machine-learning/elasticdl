# Supporting PyTorch

## Introduction

ElasticDL is an open-source distributed deep learning programming framework based
on TensorFlow and Kubernetes. Considering that PyTorch is more widely used in academia,
this project will support PyTorch.

## How PyTorch is Used

The training of most neural networks can be simplified to this process:
1. Define the network: Define the `Class` of the network, declare the instance of
the network net=Net().
2. Define the optimizer: `optimizer=optim.xxx(net.parameters()，lr=xxx)`
3. Define the loss function: `compute_loss=nn.MSELoss()`
4. training loop:
    (a) Clear the gradient information in the optimizer: `optimizer.zero_grad()`
    (b) Forward: `output=net(input)`
    (c) Calculate the loss: `loss=compute_loss(target,output)`
    (d) Backward: `loss.backward()`
    (e) Update parameters: `optimizer.step()`

### Define a Class

If you want to build a network, you need to define a `Class` first which inherits
`nn.Module`. `nn` is a very useful toolbox, `import torch.nn as nn` is needed.
For example, there are mainly two functions written in this class called `Net`,
one is the initialized `__init__` function, and the other is the `forward` function.

```python
class Net(nn.Module):
       def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
 
    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),2)
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        return x
net = Net()
```

`__init__` is the definition of the convolutional layer, and `super()` must be
executed first to initialize the parent class `nn.Module`.`forward` is the real
execution of the data flow. For example, in the above code, the input `x` passes
through the defined `conv1` first and then passes through the activation function
`F.relu()`.In the beginning, you should `import torch.nn.functional` as `F`.
`F.relu()` is an official function. If you define `relu` as `myrelu` in `__init__`,
then your first sentence will be `x=F.max_pool2d(myrelu(self.conv1(x)),2)`.
After a series of flows, return `x` to the outside.

### Input with DataLoader

PyTorch has an abstract `Dataset` class. This[tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
walks through a nice example of creating a custom `FacialLandmarkDataset` class as
a subclass of `Dataset`.

PyTorch’s [TensorDataset](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset)
is a `Dataset` wrapping tensors.
`torch.utils.data.DataLoader` is an iterator that provides all these features.

- Batching the data
- Shuffling the data
- Load the data in parallel using multiprocessing workers.

```python
# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
```

### Training

Next, we input `train_data` into neural network and get output by `forward()`, and
finally calculate the error. The code below omits the part of calculating the accuracy.
If you want to take a closer look at the accuracy code, please go to me see all the
code on github.

```python
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
```

If you want the `output` of the neural network to be similar to the ground truth
you expect, that is to keep reducing the difference between the two. This difference
is defined by you, which is the object function or loss function. If the `loss`
function approaches 0, then the goal is naturally achieved.

## How to make ElasticDL work with PyTorch

The master process of ElasticDL uses asynchronous or synchronous SGD methods to
coordinate workers for `training`. When using asynchronous SGD method, the master
will start a high-performance parameter server for each worker to use. When using
synchronous SGD, ElasticDL uses Kubernetes-native's fault-tolerable AllReduce implementation.

ElasticDL holds [master-worker architecture](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/overall.md#architecture).
The master node plays the master role in two aspects.

1. It's the master of the cluster.
2. It's the master of the model training/evaluation/prediction process.

### 1. Simple and Standardized Model Method

ElasticDL requests users to provide several functions, including `forward`,
`loss`, `optimizer` and `feed`. [Here](https://github.com/sql-machine-learning/elasticdl/blob/develop/model_zoo/mnist/mnist_subclass.py)
is a MNIST model written in TensorFlow Keras API. The `feed` customizes the conversion
process of training data to PyTorch model input.

In PyTorch, we follow the same interface design, and the following is a detailed
example.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output
net = Net()
```

In addition to defining models, users also need to specify `feed`, `loss`, `optimizer`
functions.

```python
def loss(labels, predictions):
    labels = tf.reshape(labels, [-1])
    func = nn.CrossEntropyLoss()
    return func(predictions, labels)
    )

def optimizer(lr=0.1):
    return torch.optim.Adam(cnn.parameters(), lr)


# how to make PyTorch DataLoad works with ElasticDL master, need improvment
def feed(dataset, mode, _):
    def _parse_data(record):
        if mode == Mode.PREDICTION:
            feature_description = {
                "image": Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
            }
        else:
            feature_description = {
                "image": Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True),
                "label": Data.DataLoader(dataset=train_label, batch_size=BATCH_SIZE, shuffle=True),
            }
        r = tf.io.parse_single_example(record, feature_description)
        features = {
            "image": tf.math.divide(tf.cast(r["image"], tf.float32), 255.0)
        }
        if mode == Mode.PREDICTION:
            return features
        else:
            return features, tf.cast(r["label"], tf.int32)

    dataset = dataset.map(_parse_data)

    if mode == Mode.TRAINING:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset
```


### 2. Load Data from Task

ElasticDL introduces a master process for each job. By calling the Kubernetes API,
the master process understands the cluster situation. The data is distributed by
the master.[dynamic_data_sharding.md](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/dynamic_data_sharding.md)

1. A worker get a task from the master.
2. A worker reads real data according to the offset in the task
`feed` customizes the conversion process of training data
to PyTorch model input.

`TODO: Make DataLoader works with task, more details will be added.`

There is a tutorial about [feed](https://github.com/sql-machine-learning/elasticdl/blob/dc4b2901d651cea08cdb2825a6829c97294e4652/model_zoo/mnist/mnist_subclass.py#L64)
in TensorFlow.

### 3. Transmission of Gradient Information

A task received by an ElasticDL worker usually includes multiple minibatches.
For each task, the worker opens the corresponding file or table, and then:
1. Get a mini-batch training data.
2. Call the user-defined `forward` function with the local model as a parameter
to calculate the cost. If the model is large, some parameters may come from the
parameter server.
3. The worker performs backward calculations to obtain the gradient.
4. In synchronous SGD, the worker calls `AllReduce` to implement FTlib to synchronize
gradients and update the model. In asynchronous SGD, the worker uploads gradients
to the `parameter server` from time to time, and also obtains global model parameters
from the parameter server from time to time.

```python
while (True):
    task = get_task()
    dataset = create_dataset(task)
    for minibatch in dataset:
        pull_parameters()
        forward()
        backward()
        push_gradients()
```

#### Gradient Information Acquisition

The advanced API in PyTorch such as `torch.optim` is not available,we had to
update the value of each parameter by name, and manually zero the gradient of
each parameter.
`torch.no_grad()` context is necessary because we don't want to record these
operations in the next gradient calculation.

#### Aggregating Gradients under Parameter Server Strategy

This [document](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/parameter_server.md)
describes the design of a distributed parameter server for ElasticDL.

`PSClient` provides several util functions, like `push_gradients` and `pull_dense_parameters`,
we could directly use them.

```python
with torch.no_grad():
    grads = [param.grad.numpy() for param in model.parameters()]
    self.ps_client.push_gradients(grads)
```

#### Model Parameter Access from Worker

In the parameter server strategy, the workers pull the latest parameters from
the PS before forwarding and push gradients to the PS after backward.
Each PS pod has a RPC server to provide RPC services. Workers use RPC services
to pull model parameters. `pull_variable` service is to pull all non-embedding
parameters. `pull_embedding_vector` service is to pull embedding vectors
specified by an embedding layer name and a list of discrete IDs.

```python
service PServer{
    rpc pull_variable(PullModelRequest) returns (PullModelResponse);
    rpc pull_embedding_vector(PullEmbeddingVectorRequest) returns (Tensor);
}
```

#### Model Parameter Update

A worker computes gradients in each training iteration, which contain gradients
for non-embedding parameters and some embedding vectors if applicable. The worker
partitions these gradients using their corresponding parameter names or discrete
IDs for embedding vectors. Then the worker sends gradient partitions to their
corresponding PS pods by RPC calls `push_gradient`.

```python
service PServer{
    rpc push_gradient(PushGradientRequest) returns (PushGradientResponse);
}
```

When a PS pod receives gradients in `push_gradient`, it uses a PyTorch optimizer
to apply gradients to non-embedding parameters.