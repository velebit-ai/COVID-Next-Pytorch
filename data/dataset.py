import os

from torch.utils.data import Dataset
import torch
from PIL import Image


class COVIDxFolder(Dataset):
    def __init__(self, img_dir, labels_file, transforms):
        self.mapping = {
            'normal': 0,
            'bacteria': 1,
            'viral': 2,
            'COVID-19': 3
        }
        self.img_pths, self.labels = self._prepare_data(img_dir, labels_file)
        self.transforms = transforms

    def _prepare_data(self, img_dir, labels_file):
        with open(labels_file, 'r') as f:
            labels_raw = f.readlines()

        labels, img_pths = [], []
        for i in range(len(labels_raw)):
            data = labels_raw[i].split()
            img_name = data[1]
            img_pth = os.path.join(img_dir, img_name)
            img_pths.append(img_pth)
            labels.append(self.mapping[data[2]])

        return img_pths, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_pths[idx]).convert("RGB")
        img_tensor = self.transforms(img)

        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor
