
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import numpy as np

def build_optimizer(optimizer_name, model, initial_lr=1e-4, weight_decay=0.04):
    """
    Build an optimizer for the model based on the user input.

    Args:
        optimizer_name (str): Name of the optimizer (sgd, adam, or adamw).
        model (nn.Module): Model for which to create the optimizer.
        initial_lr (float, optional): Initial learning rate. Default: 1e-4.
        weight_decay (float, optional): Weight decay factor. Default: 0.04.
    """
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        for param_group in optimizer.param_groups:
            param_group['initial_wd'] = param_group['weight_decay']
    else:
        raise ValueError("Invalid optimizer name")

    return optimizer

  


def build_scheduler(scheduler_name, optimizer, T_max, num_epochs, train_loader, lr_min=1e-6, wd_min=0.04, wd_max=0.4):
    """
    Build a learning rate scheduler for the optimizer based on user choice.

    Args:
        scheduler_name (str): Name of the scheduler (cosineannealing or custom).
        optimizer (Optimizer): Optimizer to apply the scheduler on.
        T_max (int): Maximum number of iterations for the scheduler.
        num_epochs (int): Number of training epochs.
        train_loader (DataLoader): DataLoader for the training set.
        lr_min (float, optional): Minimum learning rate. Default: 1e-6.
        wd_min (float, optional): Minimum weight decay. Default: 0.04.
        wd_max (float, optional): Maximum weight decay. Default: 0.4.
    """
    if scheduler_name == "cosineannealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == "custom":
        scheduler = CosineDecayLRWithWeightDecay(optimizer, T_max=num_epochs * len(train_loader), lr_min=lr_min, wd_min=wd_min, wd_max=wd_max)
    else:
        raise ValueError("Invalid scheduler name")

    return scheduler


class CosineDecayLRWithWeightDecay(_LRScheduler):
    """
    Custom learning rate scheduler that decays the learning rate and weight decay
    using a cosine function.

    Args:
        optimizer (Optimizer): Optimizer to apply the scheduler on.
        T_max (int): Maximum number of iterations.
        lr_min (float): Minimum learning rate.
        wd_min (float): Minimum weight decay.
        wd_max (float): Maximum weight decay.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, lr_min, wd_min, wd_max, last_epoch=-1):
        self.T_max = T_max
        self.lr_min = lr_min
        self.wd_min = wd_min
        self.wd_max = wd_max
        super(CosineDecayLRWithWeightDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for the current epoch using the cosine function.
        """
        t_cur = self.last_epoch
        lr_scale = self.lr_min + 0.5 * (1 - self.lr_min) * (1 + np.cos(np.pi * t_cur / self.T_max))
        wd_scale = self.wd_min + 0.5 * (self.wd_max - self.wd_min) * (1 + np.cos(np.pi * t_cur / self.T_max))

        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = param_group['initial_wd'] * wd_scale

        return [base_lr * lr_scale for base_lr in self.base_lrs]

def build_criterion():
    return torch.nn.CrossEntropyLoss()
