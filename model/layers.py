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
