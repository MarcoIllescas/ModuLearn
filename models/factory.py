from models.cnn import CNN
from models.mlp import MLP
from models.vgg import VGG11

def create_model(model_name, num_classes=10):
    if model_name == "cnn":
        return CNN()
    if model_name == "mlp":
        return MLP()
    if model_name == "vgg11":
        return VGG11(num_classes=num_classes)
    raise ValueError("Unknown model")