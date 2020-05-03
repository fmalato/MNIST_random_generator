import torch
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

from net import LeNet5, LeNet5_FC
from dummy_net import DummyNet, DummyFCN
from train_CPU import test, test_FCN

data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((28, 28)),
                      transforms.ToTensor()]))

batch_size = 16

data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=8)

net = DummyNet()
net_FCN = DummyFCN(batch_size)
# Same checkpoint for both networks
checkpoint = 'checkpoints/10_epochs_tanh_FC/10_epochs_tanh_FC.pth'
# functional.py calls a methods that asks for some weights that are not initialized. Maybe it's caused by
# unfitting data from the custom state dict?
# (When loading the custom state dict with load_state_dict() there are some keys in the model that are not in
#  the state)

# Test for regular classifier
accuracies = []
net.load_state_dict(torch.load(checkpoint))

print('Testing regular classifier netowrk:')
test(net, data_test_loader, 'cpu', nn.CrossEntropyLoss(), data_test, accuracies)
print('\n')

# Test for Fully Convolutional classifier
accuracies_fcn = []
print('Testing fully convolutional network:')

test_FCN(net_FCN, data_test_loader, 'cpu', nn.CrossEntropyLoss(), data_test, accuracies_fcn)
