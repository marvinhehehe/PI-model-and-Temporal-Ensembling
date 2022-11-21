import logging
import numpy as np
import torch
import os
import random
import copy
from torch import nn
import torchvision.transforms as transforms


def get_logger(name):
    """Initializes multi-GPU-friendly python command line logger."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    logger = logging.getLogger(name)

    return logger


def ramp_up(epoch, max_epochs, max_val, mult):
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, max_val, n_labeled, n_samples, mult=-5):
    max_val = max_val * (float(n_labeled) / n_samples)
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return ramp_up(epoch, max_epochs, max_val, mult)


def get_lr_lambda(total_epoch, step_per_epoch, max_epochs, last_epochs):
    def lr_lambda(step):
        step = step + 1
        if step <= max_epochs * step_per_epoch:
            return ramp_up(step, max_epochs * step_per_epoch, 1.0, mult=-5)
        elif step >= (total_epoch - last_epochs) * step_per_epoch:
            return ramp_up(total_epoch * step_per_epoch - step, last_epochs * step_per_epoch, 1.0, mult=-12.5)
        else:
            return 1.0

    return lr_lambda


def trans_view(input_tensor):
    return input_tensor.view(-1, input_tensor.size(-1))


def shuffle(batch):
    tmp_batch = copy.deepcopy(batch)
    input_ids, attn_mask = trans_view(tmp_batch.input_ids), trans_view(tmp_batch.attention_mask)
    bsz, seq_len = input_ids.shape
    position_ids = torch.arange(512).expand((bsz, -1))[:, :seq_len]
    # shuffle position_ids
    shuffled_pid = []
    for bsz_id in range(bsz):
        sample_pid = position_ids[bsz_id]
        sample_mask = attn_mask[bsz_id]
        num_tokens = sample_mask.sum().int().item()
        indexes = list(range(1, num_tokens - 1))
        random.shuffle(indexes)
        rest_indexes = list(range(num_tokens, seq_len))
        total_indexes = [0] + indexes + [num_tokens - 1] + rest_indexes
        shuffled_pid.append(torch.index_select(sample_pid, 0, torch.tensor(total_indexes)))
    tmp_batch.position_ids = torch.stack(shuffled_pid, 0).to(input_ids.device)
    return tmp_batch


def cutoff(batch, rate):
    tmp_batch = copy.deepcopy(batch)
    input_ids, attn_mask = trans_view(tmp_batch.input_ids), trans_view(tmp_batch.attention_mask)
    bsz, seq_len = input_ids.shape
    cutoff_pid = []
    for bsz_id in range(bsz):
        num_tokens = attn_mask[bsz_id].sum().int().item()
        num_cutoff_indexes = int(num_tokens * rate)
        if num_cutoff_indexes < 0 or num_cutoff_indexes > num_tokens:
            raise ValueError(
                f"number of cutoff dimensions should be in (0, {num_tokens}), but got {num_cutoff_indexes}")
        indexes = list(range(num_tokens))
        random.shuffle(indexes)
        cutoff_indexes = indexes[:num_cutoff_indexes]
        cutoff_pid.append(torch.tensor(
            [attn_mask[bsz_id][i] if ((input_ids[bsz_id][i] == 101) or (i not in cutoff_indexes)) else 0 for i in
             range(seq_len)]))
    tmp_batch.attention_mask = torch.stack(cutoff_pid, 0).to(input_ids.device)
    return tmp_batch


aug_type = {
    "cifar-10": nn.Sequential(
        transforms.Pad(padding=2),
        transforms.RandomCrop(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    ),
    "svhn": nn.Sequential(
        transforms.Pad(padding=2),
        transforms.RandomCrop(size=(32, 32)),
        transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    )}


def img_aug(batch, task_name):
    tmp_batch = copy.deepcopy(batch)
    tmp_batch.imgs = aug_type[task_name](tmp_batch.imgs)
    return tmp_batch


def save_checkpoints(filename, ckpt):
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, "wb") as f:
        torch.save(ckpt, f)


def load_checkpoints(filename, device):
    obj = torch.load(filename, map_location=torch.device(device))
    return obj
