from torch import nn

from collections import OrderedDict

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.Tanh()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.Tanh()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.Tanh())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.Tanh()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

class LeNet5_FC(nn.Module):

    def __init__(self):
        super(LeNet5_FC, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.Tanh()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.Tanh()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.Tanh())
        ]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('f6', nn.Conv2d(120, 84, kernel_size=(1, 1))),
            ('relu6', nn.Tanh()),
            ('f7', nn.Conv2d(84, 10, kernel_size=(1, 1))),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))


    def forward(self, img):
        output = self.convnet(img)
        #output = output.view(img.size(0), -1)
        output = self.convnet2(output)
        return output
