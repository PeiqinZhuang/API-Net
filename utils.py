import torch
import shutil
import os
import torchvision
from torch.utils.data import default_collate


def save_checkpoint(save_path, state, is_best, saved_file):
    torch.save(state, saved_file)
    if is_best:
        shutil.copyfile(saved_file, os.path.join(save_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def accuracy_test(scores, targets):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = torch.topk(scores, 1)
    correct = ind.eq(targets.view(-1))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def accuracy_test_binary(scores, targets):
    batch_size = targets.size(0)
    _, ind = torch.topk(scores, 1)
    if ind == 0 and targets.view(-1) == 0:
        correct = 1
    elif ind != 0 and targets != 0:
        correct = 1
    else:
        correct = 0

    return correct * (100.0 / batch_size)


def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.zeros_(m.weight)


def my_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    return default_collate(batch)
