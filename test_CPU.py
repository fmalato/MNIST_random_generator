import torch
import torchvision.transforms as transforms

from generator import MNISTDataset
from net import LeNet5, LeNet5_FC, ShittyFCN
from train_CPU import test

data_test = MNISTDataset('MNIST_seg_dataset/test_set/annotations.csv',
                         'MNIST_seg_dataset/test_set/imgs/',
                          transform=transforms.Compose([
                              transforms.Resize((28, 28)),
                              transforms.ToTensor()]))

net = LeNet5_FC()
checkpoint = 'checkpoints/16_epochs_tanh_FC/16_epochs_tanh_FC.pth'
net.load_custom_state_dict(torch.load(checkpoint))
# functional.py calls a methods that asks for some weights that are not initialized. Maybe it's caused by
# unfitting data from the custom state dict?
# (When loading the custom state dict with load_state_dict() there are some keys in the model that are not in
#  the state)
test(net)
