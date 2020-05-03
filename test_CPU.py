import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchsummary import summary

from net import LeNet5, LeNet5_FC
from dummy_net import DummyNet, DummyFCN
from train_CPU import test, test_FCN, train_FCN

data_train = MNIST('./data/mnist',
                       download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

batch_size = 16

data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=8)

net = LeNet5()
net_FCN = LeNet5_FC()
# Same checkpoint for both networks
checkpoint = 'checkpoints/10_epochs_LeNet5_32x32/10_epochs_LeNet5_32x32.pth'

# functional.py calls a methods that asks for some weights that are not initialized. Maybe it's caused by
# unfitting data from the custom state dict?
# (When loading the custom state dict with load_state_dict() there are some keys in the model that are not in
#  the state)

# Test for regular classifier
accuracies = []
net.load_state_dict(torch.load(checkpoint))

print('Testing regular classifier netowrk:')
test(net, data_test_loader, 'cpu', nn.CrossEntropyLoss(), data_test, accuracies)

# Test for Fully Convolutional classifier
losses_fcn = []
accuracies_fcn = []
net_FCN.load_custom_state_dict(torch.load(checkpoint))

print('\nTesting fully convolutional network:')
# TODO: Should it be re-trained due to the reshaping of the parameters in the Linear layers?
"""train_FCN(net, data_train_loader, optim.SGD(net.parameters(), lr=2e-3, momentum=0.9),
          'cpu', nn.CrossEntropyLoss(), losses_fcn, 1)"""
test_FCN(net_FCN, data_test_loader, 'cpu', nn.CrossEntropyLoss(), data_test, accuracies_fcn)
