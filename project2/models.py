import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear, ReLU, LeakyReLU, Sequential, Flatten, Conv2d, MaxPool2d

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.convolutional_layers = Sequential(
            Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.fully_connected_layers = Sequential(
            Flatten(),
            Linear(in_features=16 * 4, out_features=64),
            LeakyReLU(),
            Linear(in_features=64, out_features=64),
            LeakyReLU(),
            Linear(in_features=64, out_features=32),
            LeakyReLU(),
            Linear(in_features=32, out_features=n_actions),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.convolutional_layers(x)
        x = self.fully_connected_layers(x)
        return x 