import torch
import torch.nn as nn
import numpy as np
import threading


# The following is supposed to be the content of a user program.

# ElasticFlow users write a deep learning training program that can
# run locally and distributedly simply by define a PyTorch nn.Module
# class -- just like what they do in usual PyTorch programming work --
# but no need to write the main loop nor the tediously complex
# parallel computing logic like calling the torch.distribute package.
# ElasticFlow will handle the distributed computing logic, argurablly
# better than torch.distribute could do.
class UserDefinedModule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._linear = nn.Linear(1, 1)

    def forward(self, x):
        return self._linear(x)

    def loss(self, x, y):
        return nn.MSELoss()(self.forward(x), y)

    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

 # The following is supposed to be part of the ElasticFlow framework.


class DataSource(object):
    def get_data(self):
        x = np.random.rand()
        return np.array([x, x * 2 + 1], dtype='float')


class ParameterServer(object):
    def __init__(self, module_cls):
        self._lock = threading.Lock()
        self._model = module_cls()
        self._optmr = self._model.optimizer()

    # Note that unlike in TensorFlow, a parameter in PyTorch doesn't have
    # an identifier. Here we rely on the fact that the order of the
    # parameters won't change during training. In real framework, we will
    # need a way to match parameters in ParameterServer and its clients.
    def push(self, grad):
        with self._lock:
            params = list(self._model.parameters())
            assert len(params) == len(grad)
            for p, g in zip(params, grad):
                # Note that weight is a 1 x 1 tensor and bias is a scalar.
                if p.size() == torch.Size([1, 1]):
                    p._grad = torch.tensor([[g]])
                else:
                    p._grad = torch.tensor([g])
            self._optmr.step()

    def pull(self):
        with self._lock:
            return [p.data.item() for p in self._model.parameters()]


class Worker(threading.Thread):
    def __init__(self, name, ds, ps, module_cls):
        self._name = name
        self._ds = ds
        self._ps = ps
        self._model = module_cls()
        self._optmr = self._model.optimizer()
        threading.Thread.__init__(self, name=name)

    def update_param(self, vals):
        params = list(self._model.parameters())
        assert len(params) == len(vals)
        for p, v in zip(params, vals):
            p.data.copy_(torch.tensor([v]))

    def run(self):
        for i in range(100):
            if i % 2 == 0:
                w = self._ps.pull()
                self.update_param(w)

            # Forward and backward pass
            self._optmr.zero_grad()
            x, y = self._ds.get_data()
            loss = self._model.loss(torch.tensor([x]), torch.tensor([y]))
            loss.backward()
            self._optmr.step()

            # Collect local gradients
            grad = [p._grad.item() for p in self._model.parameters()]
            self._ps.push(grad)

            for p in self._model.parameters():
                print('%s w: %f g: %f' %
                      (self._name, p.item(), p._grad.item()))


def main():
    ps = ParameterServer(UserDefinedModule)

    worker1 = Worker('worker1', DataSource(), ps, UserDefinedModule)
    worker2 = Worker('worker2', DataSource(), ps, UserDefinedModule)

    worker1.start()
    worker2.start()

    worker1.join()
    worker2.join()

    for p in ps._model.parameters():
        print('%f' % (p.item()))


if __name__ == '__main__':
    main()
