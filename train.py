import logging
import os

import torch
from torch.optim import AdamW
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report

import transforms
from dataset import COVIDX_numpy
from torch.utils.data import DataLoader
from model import COVIDNet
import util


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = dict()


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
        name = "COVIDNet_F1_{:.4f}_step_{}".format(config['clf_report']['f1'],
                                                   config['global_step'])
        model_path = os.path.join(config['save_dir'], name)
        torch.save(state, model_path)
        log.info("Saved model to {}".format(model_path))


def validate(data_loader, model, best_score, global_step, cfg):
    gts, predictions = [], []
    for data in data_loader:
        imgs, labels = data
        imgs = util.to_device(imgs, gpu=cfg.gpu)

        with torch.no_grad():
            logits = model(input=imgs)

        probs = model.probs(logits)
        preds = torch.argmax(probs, dim=1)

        labels = labels.cpu().detach().numpy()

        predictions.extend(preds)
        gts.extend(labels)

    predictions = np.array(predictions, dtype=np.int32)
    gts = np.array(gts, dtype=np.int32)
    f1, prec, rec = util.calc_metrics(predictions=predictions,
                                      targets=predictions,
                                      average="macro")
    report = classification_report(gts, predictions, output_dict=True)

    if f1 > best_score:
        save_config = {
                    'save_dir': config.save_dir,
                    'global_step': global_step,
                    'clf_report': report
                }
        save_model(model=model, config=save_config)
        best_score = f1

    return best_score


def main():
    if config.gpu and not torch.cuda.is_available():
        raise IOError("GPU not supported or enabled on this system.")
    use_gpu = config.gpu

    log.info("Loading train dataset")

    train_dataset = COVIDX_numpy(None, None, transforms.train_transforms())
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=config.n_threads,
                              pin_memory=use_gpu)

    val_dataset = COVIDX_numpy(None, None, transforms.train_transforms())
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.n_threads,
                            pin_memory=use_gpu)

    if config.weights:
        state = torch.load(config.weights)
        log.info("Loaded model weights from: {}".format(config.weights))
    else:
        state = None

    state_dict = state["state_dict"] if state else None

    model = COVIDNet()
    if state_dict:
        model = util.load_model_weights(model=model, state_dict=state_dict)

    if use_gpu:
        model.cuda()
        model = torch.nn.DataParallel(model)

    optim_layers = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer and lr scheduler
    optimizer = AdamW(optim_layers,
                      lr=config['lr'],
                      weight_decay=config['weight_decay'])

    scheduler = ReduceLROnPlateau(optimizer=optimizer)

    # Load the last global_step from the checkpoint if existing
    global_step = 0 if state is None else state['global_step']

    loss_fn = CrossEntropyLoss(reduction='mean')

    # Reset the best metric
    best_metric = -1
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

            if global_step % config.log_steps == 0:
                log.info("Batch loss {:.4f}".format(loss.item()))

            if global_step % config.eval_steps == 0:
                best_score = validate(val_loader,
                                      model,
                                      best_score=best_metric,
                                      global_step=global_step,
                                      cfg=config)
                scheduler.step(best_score)
            global_step += 1

        log.info("Finished epoch {}/{}".format(epoch + 1, config.epochs))


if __name__ == '__main__':
    seed = config.random_seed
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    main()
