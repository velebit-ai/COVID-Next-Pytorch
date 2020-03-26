import torch
from torch import nn
from torchvision import models

from .layers import Trainable


class ResNext50(nn.Module):
    def __init__(self, n_classes):
        super(ResNext50, self).__init__()
        self.n_classes = n_classes

        # Layers
        backbone = models.resnext50_32x4d(pretrained=True)
        # Remove softmax layer at the end
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.model = Trainable(backbone,
                               name="backbone",
                               trainable=True)
        self.logits = Trainable(nn.Linear(2048, n_classes),
                                name="logits",
                                trainable=True)

    def forward(self, input):
        net = self.model(input)
        net = torch.squeeze(net)
        return self.logits(net)

    def probability(self, logits):
        return nn.functional.softmax(logits, dim=-1)


class SqueezeNet(nn.Module):
    def __init__(self, n_classes):
        super(SqueezeNet, self).__init__()
        self.n_classes = n_classes

        # Layers
        backbone = models.squeezenet1_1(pretrained=True)
        # Remove softmax layer at the end
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.model = Trainable(backbone,
                               name="backbone",
                               trainable=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.logits = Trainable(nn.Linear(512, n_classes),
                                name="logits",
                                trainable=True)

    def forward(self, input):
        net = self.model(input)
        net = self.avg_pool(net)
        net = torch.squeeze(net)
        return self.logits(net)

    def probability(self, logits):
        return nn.functional.softmax(logits, dim=-1)


class COVIDNet(nn.Module):
    def __init__(self, n_classes):
        super(COVIDNet, self).__init__()
        self.n_classes = n_classes
        # TODO

    def forward(self, input):
        raise NotImplementedError()

    def probability(self, logits):
        raise NotImplementedError()
