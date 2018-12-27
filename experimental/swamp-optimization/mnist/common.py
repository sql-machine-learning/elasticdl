import torch
from torchvision import datasets, transforms

def prepare_data_loader(is_train, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=is_train,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size,
        shuffle=shuffle)

def bool_parser(v):
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
