import logging
from sklearn.metrics import f1_score, precision_score, recall_score, \
     accuracy_score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model_weights(model, state_dict, verbose=True):
    """
    Loads the model weights from the state dictionary. Function will only load
    the weights which have matching key names and dimensions in the state
    dictionary.

    :param state_dict: Pytorch model state dictionary
    :param verbose: bool, If True, the function will print the
        weight keys of parametares that can and cannot be loaded from the
        checkpoint state dictionary.
    :return: The model with loaded weights
    """
    new_state_dict = model.state_dict()
    non_loadable, loadable = set(), set()

    for k, v in state_dict.items():
        if k not in new_state_dict:
            non_loadable.add(k)
            continue

        if v.shape != new_state_dict[k].shape:
            non_loadable.add(k)
            continue

        new_state_dict[k] = v
        loadable.add(k)

    if verbose:
        log.info("### Checkpoint weights that WILL be loaded: ###")
        {log.info(k) for k in loadable}

        log.info("### Checkpoint weights that CANNOT be loaded: ###")
        {log.info(k) for k in non_loadable}

    model.load_state_dict(new_state_dict)
    return model


def to_device(tensor, gpu=False):
    """
    Places a Pytorch Tensor object on a GPU or CPU device.

    :param tensor: Pytorch Tensor object
    :param gpu: bool, Flag which specifies GPU placement
    :return: Tensor object
    """
    return tensor.cuda() if gpu else tensor.cpu()


def clf_metrics(predictions, targets, average='macro'):
    f1 = f1_score(targets, predictions, average=average)
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    acc = accuracy_score(targets, predictions)

    return acc, f1, precision, recall


def get_learning_rate(optimizer):
    """
    Retrieves the current learning rate. If the optimizer doesn't have
    trainable variables, it will raise an error.
    :param optimizer: Optimizer object
    :return: float, Current learning rate
    """
    if len(optimizer.param_groups) > 0:
        return optimizer.param_groups[0]['lr']
    else:
        raise ValueError('No trainable parameters.')