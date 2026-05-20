import torch.nn as nn
from DLModels.tropicallayer import TropicalLayer

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, use_tropical=False):
        super().__init__()

        layers = []

        # The paper introduces a tropical neural network where the 
        # first layer is a tropical embedding layer[cite: 13, 55].
        if use_tropical:
            # TropicalLayer acts as an embedding to transform tropical 
            # vectors to Euclidean space[cite: 12, 28].
            layers.append(TropicalLayer(input_dim, hidden_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Following layers are the same as classical ones[cite: 13, 30].
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        layers.append(nn.ReLU())
        
        # Final output layer
        layers.append(nn.Linear(hidden_dim // 2, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # The paper notes that tropical embedding preserves invariance 
        # along the tropical projective torus[cite: 57, 404].
        return self.net(x).squeeze(1)
