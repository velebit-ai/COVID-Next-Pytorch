from torchvision import transforms
import torch


class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image, dtype=torch.float32) 


def train_transforms():
    trans_list = [ToTensor()]
    return transforms.Compose(trans_list)


def val_transforms():
    trans_list = [ToTensor()]
    return transforms.Compose(trans_list)
