# General
name = "COVIDNext50_NewData"
gpu = True
batch_size = 64
n_threads = 20
random_seed = 1337

# Model
# Model weights path
weights = "./experiments/ckpts/<model.pth>"

# Optimizer
lr = 1e-4
weight_decay = 1e-3
lr_reduce_factor = 0.7
lr_reduce_patience = 5

# Data
train_imgs = "/data/ssd/datasets/covid/COVIDxV2/data/train"
train_labels = "/data/ssd/datasets/covid/COVIDxV2/data/train_COVIDx.txt"

val_imgs = "/data/ssd/datasets/covid/COVIDxV2/data/test"
val_labels = "/data/ssd/datasets/covid/COVIDxV2/data/test_COVIDx.txt"

# Categories mapping
mapping = {
    'normal': 0,
    'pneumonia': 1,
    'COVID-19': 2
}
# Loss weigths order follows the order in the category mapping dict
loss_weights = [0.05, 0.05, 1.0]

width = 256
height = 256
n_classes = len(mapping)

# Training
epochs = 300
log_steps = 200
eval_steps = 400
ckpts_dir = "./experiments/ckpts"
