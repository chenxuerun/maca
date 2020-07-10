import torch
import torch.nn as nn

def cross_entropy(x, l):
    return - l * torch.log(x) - (1 - l) * torch.log(1 - x)

def mean_square(x, l):
    return (x - l) ** 2 / 2