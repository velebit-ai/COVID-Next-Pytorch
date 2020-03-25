from torch import nn


class COVIDNet(nn.Module):
    """
    Image only classification head
    """
    def __init__(self, num_classes, trainable, tfs_preprocess=False):
        """
        :param num_classes: int, Number of supported classes
        :param trainable: bool, Layer trainable flag. Use this flag to
            make layers trainanble (True) or non-traininable (False).
        :param tfs_preprocess: bool, Whether to embed preprocessing ops
            into the model. Most commonly used during the Pytorch -> TF Serving
            conversion.
        """
        self.num_classes = num_classes
        self.tfs_preprocess = tfs_preprocess
        self.trainable = trainable

    def forward(self, input):
        """
        Implements the network forward pass

        :param input: Pytorch tensor, Image Tensor
        :return: tuple of Pytorch tensors, logits and base network features

        """
        raise NotImplementedError()