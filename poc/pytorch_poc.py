import torch
import torch.nn as nn
import numpy as np
import itertools
import threading
import copy

class DataSource(object):
    def get_data(self):
        x = np.random.rand()
        return np.array([x, x * 2 + 1], dtype='float')


class ParameterServer(object):
    def __init__(self, params):
        self._lock = threading.Lock()

        # Note that unlike in TensorFlow, a parameter in PyTorch doesn't have
        # an identifier. Here we rely on the face that the order of the 
        # parameters won't change during training. In real framework, we will
        # need a way to match parameters in ParameterServer and its clients.
        self._params = copy.deepcopy(params)

        # In real framework, the type of optimizer and learning rate need to be
        # passed in from command flags and need to be consistent with the local
        # optimizer.
        self._optim = torch.optim.SGD(self._params, lr=0.1)

    def push(self, grad):
        with self._lock:
            assert len(self._params) == len(grad)
            for p, g in zip(self._params, grad):
                # Note that weight is a 1 x 1 tensor and bias is a scalar. 
                if p.size() == torch.Size([1, 1]):
                    p._grad = torch.tensor([[g]])
                else:
                    p._grad = torch.tensor([g])
            self._optim.step()

    def pull(self):
        with self._lock:
            return [p.data.item() for p in self._params]

class Worker(threading.Thread):
    def __init__(self, name, ds, ps, model, prog):
        self._name = name
        self._ds = ds
        self._ps = ps
        self._prog = prog(name, model)
        threading.Thread.__init__(self, name = name)

    def run(self):
        for i in range(100):
            if i % 2 == 0:
                w = self._ps.pull()
                self._prog.update_param(w)
            x, y = self._ds.get_data()
            grad = self._prog.forward(x, y)
            self._ps.push(grad)

class Program(object):
    def __init__(self, name, model):
        self._name = name
        self._model = model
        self._params = list(self._model.parameters())
        self._loss = nn.MSELoss()
        self._optim = torch.optim.SGD(self._params, lr=0.1)  

    def update_param(self, vals):
        assert len(vals) == len(self._params)
        for p, v in zip(self._params, vals):
            p.data.copy_(torch.tensor([v]))

    def forward(self, x, y):
        inputs = torch.tensor([x])
        targets = torch.tensor([y])

        # Forward pass
        outputs = self._model(inputs)
        loss = self._loss(outputs, targets)
        
        # Backward and optimize
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        for p in self._params:
            print('%s w: %f g: %f' % (self._name, p.item(), p._grad.item()))

        return [p._grad.item() for p in self._params]

def main():
    model = nn.Linear(1, 1)
    ps = ParameterServer(list(model.parameters()))

    worker1 = Worker('worker1', DataSource(), ps, model, Program)        
    worker2 = Worker('worker2', DataSource(), ps, model, Program)        

    worker1.start()
    worker2.start()

    worker1.join()
    worker2.join()

if __name__ == '__main__':
    main()