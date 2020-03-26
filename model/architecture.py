import torch
from torch import nn
from torchvision import models

from .layers import Trainable


class ResNext50(nn.Module):
    def __init__(self, n_classes):
        super(ResNext50, self).__init__()
        self.n_classes = n_classes
        self.trainable = False

        # Layers
        backbone = models.resnext50_32x4d(pretrained=True)
        self.block0 = Trainable(nn.Sequential(
                                    backbone.conv1,
                                    backbone.bn1,
                                    backbone.relu,
                                    backbone.maxpool),
                                trainable=self.trainable,
                                name="conv1")
        self.block1 = Trainable(backbone.layer1,
                                trainable=self.trainable,
                                name="block1")
        self.block2 = Trainable(backbone.layer2,
                                trainable=self.trainable,
                                name="block2")
        self.block3 = Trainable(backbone.layer3,
                                trainable=self.trainable,
                                name="block3")
        self.block4 = Trainable(backbone.layer4,
                                trainable=True,
                                name="block4")
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.logits = Trainable(nn.Linear(2048, n_classes),
                                name="logits",
                                trainable=True)

    def forward(self, input):
        net = input
        for layer in [self.block0, self.block1, self.block2, self.block3,
                      self.block4]:
            net = layer(net)
        net = self.avg_pool(net)
        net = torch.squeeze(net)
        return self.logits(net)

    def probability(self, logits):
        return nn.functional.softmax(logits, dim=-1)
