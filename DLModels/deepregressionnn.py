import torch.nn as nn
from DLModels.tropicallayer import TropicalLayer

class DeepRegressionNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.2,
        use_tropical=False,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = h_dim

        if use_tropical:
            layers.append(TropicalLayer(prev_dim, 1))
        else:
            layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)
