import torch.nn as nn

from models.base import BaseNN

#          (herencia)
class MLP(BaseNN): #Desacoplamiento 
    def __init__(self):
        super(MLP, self).__init__("MLP")
        self.network = nn.Sequential(
            nn.Flatten(), #Capa de aplanamiento
            nn.Linear(28 * 28, 1000), # (caracteristicas de entrada, caracteristicas de salida)
            nn.Sigmoid(),
            nn.Linear(1000, 10), #logits
        )

    def forward(self, x):
        return self.network(x)

    