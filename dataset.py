from torch.utils.data import Dataset
import torch
import numpy as np


class COVIDxNumpy(Dataset):
    def __init__(self, imgs_npy, labels_npy, transforms):
        self.input = np.load(imgs_npy)
        self.labels = np.load(labels_npy)
        self.transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.input[idx]
        img_tensor = self.transforms(img)

        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor
