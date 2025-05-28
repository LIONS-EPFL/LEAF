import sys

import wandb
from time import sleep
import os
import torch
import numpy as np


def init_wandb(project_name, model_name, config, **wandb_kwargs):
    os.environ['WANDB__SERVICE_WAIT'] = '300'
    while True:
        try:
            wandb_run = wandb.init(
                project=project_name, name=model_name, save_code=True,
                config=config, **wandb_kwargs,
                )
            break
        except Exception as e:
            print('wandb connection error', file=sys.stderr)
            print(f'error: {e}', file=sys.stderr)
            sleep(1)
            print('retrying..', file=sys.stderr)
    return wandb_run

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def print_statistics(arr):
    # make sure its 1-d
    assert len(arr.shape) == 1
    print(
        f"[mean] {arr.mean():.4f} [median] {np.median(arr):.4f} [min] {arr.min():.4f} [max] "
        f"{arr.max():.4f} [std] {arr.std():.4f} [n] {len(arr)}\n"
    )


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def project_perturbation(perturbation, eps, norm):
    if norm in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif norm in [2, 2.0, 'l2', 'L2', '2']:
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    else:
        raise NotImplementedError(f'Norm {norm} not supported')


def normalize_grad(grad, p):
    if p in ['inf', 'linf', 'Linf']:
        return grad.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        bs = grad.shape[0]
        grad_flat = grad.view(bs, -1)
        grad_normalized = F.normalize(grad_flat, p=2, dim=1)
        return grad_normalized.view_as(grad)