import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNN
from torchvision.models import (vgg11, VGG11_Weights)

class VGG11(BaseNN):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__(name="vgg11")
        self.network = vgg11(VGG11_Weights.DEFAULT)

        for parameter in self.network.features.parameters():
            parameter.requires_grad = False

        in_features = self.network.classifier.in_features
        self.network.classifier[-1] = nn.Linear(in_features, num_classes)

        self.register_buffer("mnist_std", torch.tensor([0.3081].view(1, 1, 1, 1)))
        self.register_buffer("mnist_mean", torch.tensor([0.1307].view(1, 1, 1, 1)))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225].view(1, 3, 1, 1)))
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406].view(1, 3, 1, 1)))

    def forward(self, x):
        x = x * self.mnist_std + self.mnist_mean
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.imagenet_mean) / self.imagenet_std

        return self.network(x)