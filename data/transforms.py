from torchvision import transforms
import torch


class ToTensor(object):
    """
    Convert to Pytorch float32 tensor and transpose from HWC to CHW format.
    """
    def __call__(self, image):
        tensor = torch.from_numpy(image).to(torch.float32)
        tensor = torch.transpose(tensor, 0, 1)
        return torch.transpose(tensor, 0, 2)


def train_transforms():
    trans_list = [ToTensor()]
    return transforms.Compose(trans_list)


def val_transforms():
    trans_list = [ToTensor()]
    return transforms.Compose(trans_list)
