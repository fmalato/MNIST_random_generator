from torch import nn

from collections import OrderedDict


class DummyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1)),
            ('relu1', nn.ReLU()),
            ('mp2', nn.MaxPool2d(kernel_size=(1, 1), stride=2)),
            ('relu2', nn.ReLU()),
            ('c3', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)),
            ('relu3', nn.ReLU()),
            ('mp4', nn.MaxPool2d(kernel_size=(1, 1), stride=2)),
            ('relu4', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('fc5', nn.Linear(64*6*6, 128)),
            ('relu5', nn.ReLU()),
            ('fc6', nn.Linear(128, 128)),
            ('relu6', nn.ReLU()),
            ('e7', nn.Linear(128, 10)),
            ('sig7', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class DummyFCN(nn.Module):

    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(1, 1), stride=2)),
            ('relu2', nn.ReLU()),
            ('c3', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(1, 1), stride=2)),
            ('relu4', nn.ReLU())
        ]))
        # Convolutionalized
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Conv2d(64 * 6 * 6, 128, kernel_size=(1, 1))),
            ('relu6', nn.ReLU()),
            ('f7', nn.Conv2d(128, 128, kernel_size=(1, 1))),
            ('relu7', nn.ReLU()),
            ('fc8', nn.Conv2d(128, 10, kernel_size=(1, 1))),
            ('sig8', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view([self.batch_size, 64*6*6, 1, 1])
        output = self.fc(output)
        return output

    def load_custom_state_dict(self, state_dict):
        self_state = self.state_dict()
        layer = 0
        for name, param in state_dict.items():
            if name not in self_state:
                continue
            else:
                param = param.data
                # -------- Reshaping parameters --------
                new_shape = list(list(self.state_dict().items())[layer][1].shape)
                param = param.view(new_shape)
                # --------------------------------------
            self_state[name].copy_(param)
            layer += 1
