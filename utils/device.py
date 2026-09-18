import torch
from functools import cache

@cache #Depurador para cache
def get_device():
    #indicar el dispositivo a usar
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

    