# COVID-Next &rarr; Pytorch upgrade of the COVID-Net

Inspired by the recent paper [COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images](https://arxiv.org/pdf/2003.09871.pdf) and its Tensorflow [implementation](https://github.com/lindawangg/COVID-Net), we are now open sourcing the upgraded Pytorch version called COVID-Next.

COVID-Next features an architecture that builds upon the famous ResNext50 architecture, which has around **5x** less parameters than the original COVID-Net, and achieves comparable performance.

Tensorflow and Pytorch are two major deep learning frameworks and our motivation was to give the Pytorch research community the same starting ground Tensorflow already has when it comes to AI COVID-19 research. As the authors from the paper have already mentioned, this **model still doesn't offer production ready performance**. The key issue that needs to be resolved is the number of COVID-19 images as the number of such images is currently **not diverse and large enough** to provide representative prediction results end-users could expect in the production system.

## Requirements

As always, we recommend [virtual environments](https://docs.python.org/3/tutorial/venv.html) where you install all requirements separately from your system ones. This step is optional :)

To install all requirements, simply run `pip3 install -r requirements.txt`.
Code was tested with Python 3.6.9.

## Pretrained model

Download the pretrained COVID-Next model from [here](https://drive.google.com/open?id=1G8vQKBObt52b4qe5cQdoQkdPxjZK3ucI).

## Training

Training configuration is currently modified through the `config.py` module. Check it out before starting training.

`python3 train.py` command will run model training.

### Dataset

We have created a script that automates the dataset generation from the two sources referenced in the original repo. To generate the dataset, follow these steps:

1. Download the datasets listed below:
    * COVID ChestXray [dataset](https://github.com/ieee8023/covid-chestxray-dataset.git). Be aware this repository is constantly adding new images.
    * Kaggle RSNA pneumonia [dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
2. Run the `generate_dataset.py` script. Run `python3 generate_dataset.py -h` to see supported arguments.

The script will create a new folder with `train` and `test` subfolders where images are located, along with the two metadata files for both train and test subsets.


### Note

IO will probably be a bottleneck during training because most of the images are large and a lot of time is wasted on loading images into memory. To avoid this issue, we suggest downscaling images beforehand to input size used by the model.

You can also try to increase the `config.n_threads` to alleviate this issue but beware that increasing the number of threads will result in increased memory usage.

## Results

The following results were obtained on the dataset used in the original repo as of March 20 2020.
|                   | Accuracy | F1 Macro | Precision Macro | Recall Macro |
|:-----------------:|:--------:|:--------:|:---------------:|:--------------:|
| COVID-Net (Large) | 91.90%   | 91.39%   | 91.4%           | 91.33%       |
| **COVID-Next**    | 94.76%   |     92.98%     |       96.40%          |       90.33%      |

### Minimal prediction example

You can find the minimal prediction example in `minimal_prediction.py`.
The example demonstrates how to load the model and use it to predict the disease type on the image.

## Upgrades

* [x] Training image augmentations
* [x] Pretrained model
* [x] Minimal prediction example
* [x] Loss weights
* [x] Automated dataset generation
* [ ] Define train, validation, and test data splits for more proper model evaluation.
* [ ] Tensorboard Logging
* [ ] Smart sampling
