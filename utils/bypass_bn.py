import torch
import torch.nn as nn
import torch.nn.modules.batchnorm as _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)