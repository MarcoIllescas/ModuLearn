import torch

from torch.nn import nn
from abc import ABC, abstractmethod

class BaseNN(nn.Module, ABC):
    def __init__(self, name):
        super(BaseNN, self).__init__()
        self.name = name

    @abstractmethod
    def forward(self, x):
        pass