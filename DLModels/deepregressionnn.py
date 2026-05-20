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
        
        # Following the research, the tropical layer is utilized as the 
        # first hidden layer to perform tropical embedding[cite: 13, 55].
        if use_tropical:
            layers.append(TropicalLayer(input_dim, hidden_dims[0]))
            # Note: Classical layers like BatchNorm follow the embedding[cite: 13, 55].
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[0]
            # Start loop from the second hidden dimension
            current_hidden_dims = hidden_dims[1:]
        else:
            prev_dim = input_dim
            current_hidden_dims = hidden_dims

        # Subsequent layers are the same as classical ones[cite: 13, 30].
        for h_dim in current_hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = h_dim

        # Final linear output head for regression mapping
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)
