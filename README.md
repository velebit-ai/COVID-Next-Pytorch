# COVID-Next ---> Pytorch upgrade of the COVID-Net

Inspired by the recent paper [COVID-Net: A Tailored Deep Convolutional Neural Network Design forDetection of COVID-19 Cases from Chest Radiography Images](https://arxiv.org/pdf/2003.09871.pdf) and its Tensorflow [implementation](https://github.com/lindawangg/COVID-Net), we are now open sourcing the upgraded Pytorch version called COVID-Next.

COVID-Next features an architecture that builds upon the famous ResNext50 architecture, which has around **10x** less parameters than the original COVID-Net, and achieves comparable performance.

Tensorflow and Pytorch are two major deep learning frameworks and our motivation was to give the Pytorch research community the same starting ground Tensorflow already has when it comes to AI COVID-19 research. As the authors from the paper have already mentioned, this model still doesn't offer production ready performance, but with more data and better models, we hope productization will soon become reality.

## Requirements

As always, we recommend [virtual environments](https://docs.python.org/3/tutorial/venv.html) where you install all requirements separately from your system ones. This step is optional :)

To install all requirements, simply run `pip3 install -r requirements.txt`.
Code was tested with Python 3.6.9.

## Pretrained model

Download the pretrained COVID-Next model from [here](TODO).

## Training

Training configuration is currently modified through the `config.py` module. Check it out before starting training.

`python3 train.py` command will run model training.

### Note

IO will probably be a bottleneck during training because most of the images are large and a lot of time is wasted on loading images into memory. To avoid this issue, we suggest downscaling images beforehand to input size used by the model.

You can also try to increase the `config.n_threads` to alleviate this issue but beware that increasing the number of threads will result in increased memory usage.

### Dataset

We have used the dataset referenced in the paper which you can find
[here](https://drive.google.com/open?id=1L0_mojCvH9K7r3D2I4mj2jGf_lMKm-m9). **Be aware this dataset was downloaded on March 26 2020 and could be deprecated by now as the authors mention**. 
Please refer the [original repo](https://github.com/lindawangg/COVID-Net) for the newest version of the dataset. Automating the dataset generation process will be one of our future tasks for this repo.

----

Chest radiography images distribution (as of March 26 2020)
|  Type | Normal | Bacterial| Non-COVID19 Viral | COVID-19 Viral | Total |
|:-----:|:------:|:--------:|:-----------------:|:--------------:|:-----:|
| train |  1349  |   2540   |       1355        |        66      |  5310 |
|  test |   234  |    246   |        149        |        10      |   639 |

## Evaluating

Results of the COVID-Next model on the test dataset (as of March 26 2020).
| Accuracy | F1 Macro | Precision Macro | Recall Macro |
|:--------:|:--------:|:---------------:|:------------:|
| 82.32%   | 84.38%   | 86.04%          | 83.71%       |

### Minimal prediction example

You can find the minimal prediction example in `minimal_prediction.py`.
The example demonstrates how to load the model and use it to predict the disease type on the image.

## Upgrades

- [x] Image augmentations
- [x] Pretrained model
- [x] Minimal prediction example
- [ ] Automated dataset generation
- [ ] Tensorboard Logging
- [ ] Smart sampling