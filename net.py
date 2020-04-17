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

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Conv2d(120, 84, kernel_size=(1, 1))),
            ('relu6', nn.Tanh()),
            ('f7', nn.Conv2d(84, 120, kernel_size=(1, 1))),
            ('relu7', nn.Tanh()),
            ('fc8', nn.Conv2d(120, 10, kernel_size=(1, 1))),
            ('sig8', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

    def load_custom_state_dict(self, state_dict):
        self_state = self.state_dict()
        layer = 0
        for name, param in state_dict.items():
            if name not in self_state:
                continue
            else:
                # Temporary solution: the last layer from the old model is the output layer, while the layer with
                # the same name in the new model is not. Hence, the layer in the new model has way more parameters
                # and therefore the old ones cannot be fit there.
                if layer < len(state_dict) - 2:
                    param = param.data
                    # -------- Reshaping parameters --------
                    new_shape = list(list(self.state_dict().items())[layer][1].shape)
                    param = param.view(new_shape)
                    # --------------------------------------
            self_state[name].copy_(param)
            layer += 1


class ShittyFCN(nn.Module):

    def __init__(self):
        super().__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 64, kernel_size=(5, 5))),
            ('relu1', nn.Tanh()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(64, 64, kernel_size=(5, 5))),
            ('relu3', nn.Tanh()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('relu5', nn.Tanh()),
            # Convolutionalized
            ('f6', nn.Conv2d(64, 128, kernel_size=(1, 1))),
            ('relu6', nn.Tanh()),
            ('f7', nn.Conv2d(128, 128, kernel_size=(1, 1))),
            ('relu7', nn.Tanh()),
            ('fc8', nn.Conv2d(128, 10, kernel_size=(1, 1))),
            ('sig8', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        return output
