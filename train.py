import logging
import os

import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss

from data.dataset import COVIDxNumpy
from data import transforms
from torch.utils.data import DataLoader
from model import architecture
import util
import config


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_model(model, config):
    if isinstance(model, torch.nn.DataParallel):
        # Save without the DataParallel module
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()

    state = {
        "state_dict": model_dict,
        "global_step": config['global_step'],
        "clf_report": config['clf_report']
    }
    f1_macro = config['clf_report']['macro avg']['f1-score'] * 100
    name = "{}_F1_{:.2f}_step_{}.pth".format(config['name'],
                                             f1_macro,
                                             config['global_step'])
    model_path = os.path.join(config['save_dir'], name)
    torch.save(state, model_path)
    log.info("Saved model to {}".format(model_path))


def validate(data_loader, model, best_score, global_step, cfg):
    model.eval()
    gts, predictions = [], []

    log.info("Validation started...")
    for data in data_loader:
        imgs, labels = data
        imgs = util.to_device(imgs, gpu=cfg.gpu)

        with torch.no_grad():
            logits = model(imgs)
            probs = model.module.probability(logits)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

        labels = labels.cpu().detach().numpy()

        predictions.extend(preds)
        gts.extend(labels)

    predictions = np.array(predictions, dtype=np.int32)
    gts = np.array(gts, dtype=np.int32)
    acc, f1, prec, rec = util.clf_metrics(predictions=predictions,
                                          targets=gts,
                                          average="macro")
    report = classification_report(gts, predictions, output_dict=True)

    log.info("VALIDATION | Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | "
             "Recall {:.4f}".format(acc, f1, prec, rec))

    if f1 > best_score:
        save_config = {
                    'name': config.name,
                    'save_dir': config.ckpts_dir,
                    'global_step': global_step,
                    'clf_report': report
                }
        save_model(model=model, config=save_config)
        best_score = f1
    log.info("Validation end")

    model.train()
    return best_score


def main():
    if config.gpu and not torch.cuda.is_available():
        raise ValueError("GPU not supported or enabled on this system.")
    use_gpu = config.gpu

    log.info("Loading train dataset")
    train_dataset = COVIDxNumpy(config.train_x, config.train_y,
                                transforms.train_transforms())
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=config.n_threads,
                              pin_memory=use_gpu)
    log.info("Number of training examples {}".format(len(train_dataset)))

    log.info("Loading val dataset")
    val_dataset = COVIDxNumpy(config.val_x, config.val_y,
                              transforms.val_transforms())
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.n_threads,
                            pin_memory=use_gpu)
    log.info("Number of validation examples {}".format(len(val_dataset)))

    if config.weights:
        state = torch.load(config.weights)
        log.info("Loaded model weights from: {}".format(config.weights))
    else:
        state = None

    state_dict = state["state_dict"] if state else None
    model = architecture.SqueezeNet(n_classes=config.n_classes)
    if state_dict:
        model = util.load_model_weights(model=model, state_dict=state_dict)

    if use_gpu:
        model.cuda()
        model = torch.nn.DataParallel(model)
    optim_layers = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer and lr scheduler
    optimizer = Adam(optim_layers,
                     lr=config.lr,
                     weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  factor=config.lr_reduce_factor,
                                  patience=config.lr_reduce_patience,
                                  mode='max',
                                  min_lr=1e-7)

    # Load the last global_step from the checkpoint if existing
    global_step = 0 if state is None else state['global_step'] + 1

    loss_fn = CrossEntropyLoss(reduction='mean')

    # Reset the best metric score
    best_score = -1
    for epoch in range(config.epochs):
        log.info("Started epoch {}/{}".format(epoch + 1,
                                              config.epochs))
        for data in train_loader:
            imgs, labels = data
            imgs = util.to_device(imgs, gpu=use_gpu)
            labels = util.to_device(labels, gpu=use_gpu)

            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % config.log_steps == 0 and global_step > 0:
                probs = model.module.probability(logits)
                preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
                labels = labels.cpu().detach().numpy()
                acc, f1, _, _ = util.clf_metrics(preds, labels)

                log.info("Step {} | TRAINING batch: Loss {:.4f} | F1 {:.4f} | "
                         "Accuracy {:.4f}".format(global_step, loss.item(),
                                                  f1, acc))

            if global_step % config.eval_steps == 0 and global_step > 0:
                best_score = validate(val_loader,
                                      model,
                                      best_score=best_score,
                                      global_step=global_step,
                                      cfg=config)
                scheduler.step(best_score)
            global_step += 1


if __name__ == '__main__':
    seed = config.random_seed
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    main()
