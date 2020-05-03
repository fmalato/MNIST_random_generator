import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from net import LeNet5, LeNet5_FC
from dummy_net import DummyNet, DummyFCN
from generator import MNISTDataset

from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torchsummary import summary


def train(net, data_train_loader, optimizer, device, criterion, losses, epoch):

    net.train()
    loss_list, batch_list = [], []

    for i, (images, labels) in enumerate(data_train_loader):

        optimizer.zero_grad()

        output = net(images.to(device))
        loss = criterion(output, labels.to(device))

        loss_list.append(loss.detach().item())
        batch_list.append(i+1)

        print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().item()))
        losses.append(loss.detach().item())

        loss.backward()
        optimizer.step()


def test(net_to_test, data_test_loader, device, criterion, data_test, accuracies):
    net_to_test.eval()
    total_correct = 0
    avg_loss = 0.0

    for i, (images, labels) in enumerate(data_test_loader):
        output = net_to_test(images.to(device))
        avg_loss += criterion(output, labels.to(device)).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.to(device).view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().item(), float(total_correct) / len(data_test)))
    accuracies.append(float(total_correct) / len(data_test))


def test_FCN(net_to_test, data_test_loader, device, criterion, data_test, accuracies):
    net_to_test.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        out = net_to_test(images.to(device))
        outputs = []
        for el in out:
            res = el.data.tolist()
            res = [x[0][0] for x in res]
            outputs.append(res)
        avg_loss += criterion(torch.FloatTensor(outputs), labels.to(device)).sum()
        pred = out.detach().max(1)[1]
        total_correct += pred.eq(labels.to(device).view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().item(), float(total_correct) / len(data_test)))
    accuracies.append(float(total_correct) / len(data_test))


def train_and_test(net, data_train_loader, data_test_loader, optimizer, device, criterion, epoch, data_test, losses,
                   accuracies):
    train(net, data_train_loader, optimizer, device, criterion, losses, epoch)
    test(net, data_test_loader, device, criterion, data_test, accuracies)


def main():
    # Starting code:
    # https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py

    data_train = MNIST('./data/mnist',
                       download=True,
                       transform=transforms.Compose([
                           transforms.Resize((28, 28)),
                           transforms.ToTensor()]))
    data_test = MNIST('./data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((28, 28)),
                          transforms.ToTensor()]))

    data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=32, num_workers=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # net = LeNet5()
    # net = LeNet5_FC()
    net = DummyNet()
    #net = DummyFCN()

    summary(net, input_size=(1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    # parameters given from fully convolutional networks paper
    optimizer = optim.SGD(net.parameters(), lr=2e-3, momentum=0.9)

    losses = []
    accuracies = []

    num_epochs = 10
    if '{x}_epochs_tanh_FC'.format(x=num_epochs) not in os.listdir('checkpoints/'):
        os.mkdir('checkpoints/{x}_epochs_tanh_FC'.format(x=num_epochs))

    for e in range(1, num_epochs):
        train_and_test(net, data_train_loader, data_test_loader, optimizer, device, criterion, e, data_test,
                       losses, accuracies)
    torch.save(net.state_dict(), 'checkpoints/{x}_epochs_tanh_FC/{x}_epochs_tanh_FC.pth'.format(x=num_epochs))
    with open('checkpoints/{x}_epochs_tanh_FC/loss.txt'.format(x=num_epochs), 'w+') as f:
        for loss in losses:
            f.write(str(loss) + '\n')
        f.close()
    with open('checkpoints/{x}_epochs_tanh_FC/accuracy.txt'.format(x=num_epochs), 'w+') as f:
        f.write('0.0\n')
        for a in accuracies:
            f.write(str(a) + '\n')
        f.close()


if __name__ == '__main__':
    main()
