"""
Minimal prediction example
"""

import torch
from PIL import Image

from model.architecture import COVIDNext50
from data.transforms import val_transforms


category_map = {
    0: 'normal',
    1: 'bacteria',
    2: 'viral',
    3: 'COVID-19'
}

model = COVIDNext50(n_classes=len(category_map))

ckpt_pth = '<model_checkpoint_path>.pth'
weights = torch.load(ckpt_pth)['state_dict']
model.load_state_dict(weights)
model.eval()

transforms = val_transforms(width=224, height=224)

img_pth = 'assets/covid_example.jpg'
img = Image.open(img_pth).convert("RGB")
img_tensor = transforms(img).unsqueeze(0)

with torch.no_grad():
    logits = model(img_tensor)
    cat_id = int(torch.argmax(logits))
print("Prediction for {} is: {}".format(img_pth, category_map[cat_id]))
