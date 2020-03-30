import torch
from torch import nn
from torchvision import models

from .layers import Trainable, ConvBn2d


class COVIDNext50(nn.Module):
    def __init__(self, n_classes):
        super(COVIDNext50, self).__init__()
        self.n_classes = n_classes
        trainable = True

        # Layers
        backbone = models.resnext50_32x4d(pretrained=True)
        self.block0 = Trainable(nn.Sequential(
                                    backbone.conv1,
                                    backbone.bn1,
                                    backbone.relu,
                                    backbone.maxpool),
                                trainable=trainable,
                                name="conv1")
        self.block1 = Trainable(backbone.layer1,
                                trainable=trainable,
                                name="block1")
        self.block2 = Trainable(backbone.layer2,
                                trainable=trainable,
                                name="block2")
        self.block3 = Trainable(backbone.layer3,
                                trainable=trainable,
                                name="block3")
        self.block4 = Trainable(backbone.layer4,
                                trainable=trainable,
                                name="block4")
        self.backbone_end = Trainable(nn.Sequential(
                                        ConvBn2d(2048, 512, 3),
                                        ConvBn2d(512, 1024, 1),
                                        ConvBn2d(1024, 512, 3)),
                                      name="back",
                                      trainable=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.logits = Trainable(nn.Linear(512, n_classes),
                                name="logits",
                                trainable=True)

    def forward(self, input):
        net = input
        for layer in [self.block0, self.block1, self.block2, self.block3,
                      self.block4]:
            net = layer(net)
        net = self.backbone_end(net)
        net = self.avg_pool(net)
        net = torch.squeeze(net)
        return self.logits(net)

    def probability(self, logits):
        return nn.functional.softmax(logits, dim=-1)
