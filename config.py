# General
name = "COVIDNext50"
gpu = True
batch_size = 64
n_threads = 20
random_seed = 1337

# Model
# Model weights path
weights = None

# Optimizer
lr = 1e-3
weight_decay = 5e-4
lr_reduce_factor = 0.7
lr_reduce_patience = 5

# Data
train_imgs = "/data/ssd/datasets/covid/COVIDx/imgs/train"
train_labels = "/data/ssd/datasets/covid/COVIDx/imgs/train_split.txt"

val_imgs = "/data/ssd/datasets/covid/COVIDx/imgs/test"
val_labels = "/data/ssd/datasets/covid/COVIDx/imgs/test_split.txt"

width = 224
height = 224
n_classes = 4

# Training
epochs = 300
log_steps = 200
eval_steps = 400
ckpts_dir = "/data/ssd/users/bfreskura/docker-home/covid19/COVID-Net-Pytorch/experiments/ckpts"
