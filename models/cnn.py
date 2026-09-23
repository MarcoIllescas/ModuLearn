import torch.nn as nn

from models.base import BaseNN

class CNN(BaseNN):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3, # Window size
                stride=1, # Step size
                padding=1 # Adding 0 around the image
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Feature map size: 28x28 -> 14x14
            nn.Dropout2d(0.1),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3, # Window size
                stride=1, # Step size
                padding=1 # Adding 0 around the image
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Feature map size: 14x14 -> 7x7
            nn.Dropout2d(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x # Return logits, raw output of the model (MLP)