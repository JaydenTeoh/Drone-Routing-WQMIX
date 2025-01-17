import torch.nn as nn
import numpy as np

ActivationFunctions = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "Softmax": nn.Softmax,
    "Softmax2d": nn.Softmax2d,
    "Elu": nn.ELU,
}

NormalizeFunctions = {
    "LayerNorm": nn.LayerNorm,
    "GroupNorm": nn.GroupNorm,
    "BatchNorm": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "InstanceNorm2d": nn.InstanceNorm2d
}

InitializeFunctions = {
    "orthogonal": nn.init.orthogonal_
}
