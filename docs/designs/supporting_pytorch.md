# Design doc about Supporting PyTorch

## introduction
ElasticDL is an open-source distributed deep learning programming framework based on TensorFlow 2.x and Kubernetes. At the Google Developer Day event in the fall of 2019, the ElasticDL team from Ant Financial showed off the first open-source version of ElasticDL.The primary design intent of ElasticDL is to simplify distributed programming. It allows users to provide only models described with TensorFlow 2.0 API, without requiring users to write distributed training process code. As long as the user's model definition can be adjusted locally, the model can be trained with large-scale data in a distributed environment, thereby improving R&D efficiency.

However, this project only supports tensorflow. Considering that PyTorch is more widely used in academia, this project will improve and support PyTorch.

## Dataloader
PyTorch has an abstract Dataset class. A Dataset can be anything that has a `__len__` function (called by Python’s standard `len` function) and a `__getitem__` function as a way of indexing into it. This[tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) walks through a nice example of creating a custom `FacialLandmarkDataset` class as a subclass of `Dataset`.

PyTorch’s [TensorDataset](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset) is a Dataset wrapping tensors. By defining a length and way of indexing, this also gives us a way to iterate, index, and slice along the first dimension of a tensor. This will make it easier to access both the independent and dependent variables in the same line as we train.

DataLoader is a tool used to package your data. So you have to load your own (NumPy array or other) data format into Tensor, and then put it into this wrapper. Using DataLoader help you iterate data efficiently.

As we set `BATCH_SIZE = 5`, 5 data are exported for learning in a step. The result is as follows.

```
Epoch:  0 | Step:  0 | batch x:  tensor([5., 6., 2., 1., 8.]) | batch y:  tensor([ 6.,  5.,  9., 10.,  3.])
Epoch:  0 | Step:  1 | batch x:  tensor([ 7.,  3.,  9.,  4., 10.]) | batch y:  tensor([4., 8., 2., 7., 1.])
Epoch:  1 | Step:  0 | batch x:  tensor([ 8., 10.,  5.,  7.,  9.]) | batch y:  tensor([3., 1., 6., 4., 2.])
Epoch:  1 | Step:  1 | batch x:  tensor([2., 1., 4., 6., 3.]) | batch y:  tensor([ 9., 10.,  7.,  5.,  8.])
Epoch:  2 | Step:  0 | batch x:  tensor([10.,  7.,  3.,  5.,  9.]) | batch y:  tensor([1., 4., 8., 6., 2.])
Epoch:  2 | Step:  1 | batch x:  tensor([1., 6., 4., 2., 8.]) | batch y:  tensor([10.,  5.,  7.,  9.,  3.])
```
Another advantage of DataLoader is that the data size can be adjusted automatically according to the batch size.
As we set `BATCH_SIZE = 7`, the result is as follows. The remaining data in this epoch will be returned at `step = 1`.
```
Epoch:  0 | Step:  0 | batch x:  tensor([1., 7., 2., 8., 9., 6., 3.]) | batch y:  tensor([10.,  4.,  9.,  3.,  2.,  5.,  8.])
Epoch:  0 | Step:  1 | batch x:  tensor([ 5., 10.,  4.]) | batch y:  tensor([6., 1., 7.])
Epoch:  1 | Step:  0 | batch x:  tensor([1., 6., 7., 5., 2., 9., 8.]) | batch y:  tensor([10.,  5.,  4.,  6.,  9.,  2.,  3.])
Epoch:  1 | Step:  1 | batch x:  tensor([10.,  3.,  4.]) | batch y:  tensor([1., 8., 7.])
Epoch:  2 | Step:  0 | batch x:  tensor([ 4.,  1.,  9.,  5.,  7.,  8., 10.]) | batch y:  tensor([ 7., 10.,  2.,  6.,  4.,  3.,  1.])
Epoch:  2 | Step:  1 | batch x:  tensor([3., 2., 6.]) | batch y:  tensor([8., 9., 5.])
```

```python
import torch
import torch.utils.data as Data

BATCH_SIZE = 5
x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)
def show():
    for epoch in range(3):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
            batch_x, '| batch y: ', batch_y)
            # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
            #       batch_x.numpy(), '| batch y: ', batch_y.numpy())
if __name__ == '__main__':
    show()
```
### MNIST Example

This dataset is in numpy array format, and has been stored using pickle, a python-specific format for serializing data. Each image is 28 x 28, and is being stored as a flattened row of length 784 (=28x28). Let’s take a look at one; we need to reshape it to 2d first.
```python
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()
```

PyTorch uses `torch.tensor`, rather than `numpy` arrays.
```python
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
```
Or we can convert our data like this.
```python
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
```



## Training Loop 
ElasticDL holds master-worker architecture. Workers get model parameters from parameter server (PS), compute gradients using different training data, and send computed gradients to PS. PS iteratively updates these model parameters using gradients sent by workers.
In our training loop, the gradient information of the neural network needs to be calculated and transmitted between the master and the worker.
The advanced API in PyTorch such as `torch.optim` is not available,we had to update the value of each parameter by name, and manually zero the gradient of each parameter, as follows:
```python
with torch.no_grad():
    weights -= weights.grad * lr
    bias -= bias.grad * lr
    weights.grad.zero_()
    bias.grad.zero_()
```
`torch.no_grad()` context is necessary because we don't want to record these operations in the next gradient calculation.
To go further, we can use `model.parameters()` and `model.zero_grad()` (defined by PyTorch for `nn.Module`) to make these steps more concise, and there will be no errors of forgetting some parameters, especially when we build a complex model:
```python
with torch.no_grad():
    for param in model.parameters(): param -= param.grad * lr
    model.zero_grad()
```


## Work with PS_client
This [document](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/parameter_server.md) describes the design of a distributed parameter server for ElasticDL.
```python
class PSClient(object):
    def __init__(self, ps_stubs):
        self.ps_stubs = ps_stubs
        self.ps_num = len(self.ps_stubs)
        self.parameter_to_ps = {}
        self.ps_to_parameter = {}

    def pull_embedding_vectors(self, layer_name, embedding_ids):
        return new_embeddings

    def partition_dense_parameters(self, param_names):
        # Partition dense parameters to PS
        # ps_id = string_to_id(param_name)
        
    def push_dense_parameters(self, parameters, ps_id, version):
        # Push dense parameters to PS
        # Args:parameters: a list of Tensors
        #     ps_id: PS id
        #     version: model version

    def push_gradients(self, grads, edl_grads, learning_rate, model_versions,):
        # Push gradients to PS. There two kinds of gradients:
        # - gradients of normal layers
        # - sparse gradients of ElasticDL embedding layers
        # 1. handle grads
        # 2. handle sparse grads of elasticdl embedding layers
        # Sum up the values of the duplicated indices in the
        # gradients. It can reduce the gradient payload of the
        # dense embedding.
        # 3. push gradients to PS
```

### Model Parameter Access from Worker
Each PS pod has a RPC servicer to provide RPC services. Workers use RPC services to pull model parameters. `pull_variable` service is to pull all non-embedding parameters. `pull_embedding_vector` service is to pull embedding vectors specified by an embedding layer name and a list of discrete IDs.
```python
service PServer{
    rpc pull_variable(PullModelRequest) returns (PullModelResponse);
    rpc pull_embedding_vector(PullEmbeddingVectorRequest) returns (Tensor);
}
```
### Model Parameter Update
A worker computes gradients in each training iteration, which contain gradients for non-embedding parameters and some embedding vectors if applicable. The worker partitions these gradients using their corresponding parameter names or discrete IDs for embedding vectors. Then the worker sends gradient partitions to their corresponding PS pods by RPC calls `push_gradient`.
```python
service PServer{
    rpc push_gradient(PushGradientRequest) returns (PushGradientResponse);
}
```
When a PS pod receives gradients in `push_gradient`, it uses a PyTorch optimizer to apply gradients to non-embedding parameters.


## Code
```python
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters

loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
learning_rate = 1e-4

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss

        # print(epoch, loss.item())
        # optimizer.zero_grad()           # clear gradients for this training step
        # loss.backward()                 # backpropagation, compute gradients
        # optimizer.step()                # apply gradients

        cnn.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in cnn.parameters():
                param -= learning_rate * param.grad

        if step % 50 == 0:
            test_output, last_layer = (test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
```