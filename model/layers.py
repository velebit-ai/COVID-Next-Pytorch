from torch import nn


class Trainable(nn.Module):
    """
    Wraps an arbitrary module with a Trainable module. The Trainable module
    is used as a wrapper for freezing and thawing module layers.
    """
    def __init__(self, module, name, trainable=True):
        super().__init__()
        self.module = module
        self.name = name
        self.trainable_switch(trainable)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def trainable_switch(self, trainable):
        """
        Makes module layers trainable or not.

        :param trainable: bool, False to freeze the layers, True to unfreeze
         them.
        """
        for p in self.parameters():
            p.requires_grad = trainable


def ConvBn2d(in_dim, out_dim, kernel_size,
             activation=nn.LeakyReLU(0.1, inplace=True)):
    """
    Wraps Conv2D, Batch Normalization 2D, and an arbitrary activation layers
     with a nn.Sequential layer.

    :param in_dim: int, Input feature map dimension
    :param out_dim: int, Output feature map dimension
    :param kernel_size: int or tuple, Convolution kernel size
    :return: nn.Sequential structure containing above listed network layers
    """
    padding = kernel_size // 2
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                  padding=padding, bias=False),
        nn.BatchNorm2d(out_dim),
        activation)
    return net
