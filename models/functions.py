import torch


def mean_max(x):
    return torch.mean(x, dim=1), torch.max(x, dim=1)[0]
