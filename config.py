# General
name = "SqueezeNet"
gpu = True
batch_size = 64
n_threads = 4
random_seed = 1337

# Model
weights = "experiments/ckpts/SqueezeNet_F1_68.71_step_1200.pth"
# "

# Optimizer
lr = 5e-5
weight_decay = 5e-4
lr_reduce_factor = 0.7
lr_reduce_patience = 5

# Data
train_x = "/data/ssd/datasets/covid/COVIDx/npy_files/x_train.npy"
train_y = "/data/ssd/datasets/covid/COVIDx/npy_files/y_train.npy"

val_x = "/data/ssd/datasets/covid/COVIDx/npy_files/x_test.npy"
val_y = "/data/ssd/datasets/covid/COVIDx/npy_files/y_test.npy"

n_classes = 4

# Training
epochs = 300
log_steps = 200
eval_steps = 400
ckpts_dir = "/data/ssd/users/bfreskura/docker-home/covid19/COVID-Net-Pytorch/experiments/ckpts"
