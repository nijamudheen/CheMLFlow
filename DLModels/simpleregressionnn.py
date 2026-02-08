import torch.nn as nn
from DLModels.tropicallayer import TropicalLayer

class SimpleRegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, use_tropical=False):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        ]

        if use_tropical:
            layers.append(TropicalLayer(hidden_dim // 2, 1))
        else:
            layers.append(nn.Linear(hidden_dim // 2, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)
