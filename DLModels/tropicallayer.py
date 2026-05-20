import torch
import torch.nn as nn

# Define a custom tropical layer for neural networks
# This layer computes a form of negative tropical distance between input and learnable weights
class TropicalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TropicalLayer, self).__init__()

        self.in_features = in_features      # Number of input features
        self.out_features = out_features    # Number of output features

        # Learnable weight matrix of shape [out_features, in_features]
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Initialize weights with a normal distribution
        nn.init.normal_(self.weight)

    def forward(self, x):
        """
        Returns the negative tropical distance between input x and the learned weights.
        Tropical distance here is defined as: min(x - w) - max(x - w)
        """

        # Compute pairwise differences: [B, 1, in] - [out, in] → [B, out, in]
        result_addition = x.unsqueeze(1) - self.weight

        # For each output dimension, compute tropical distance:
        # result = min_i(x_i - w_i) - max_i(x_i - w_i) for each output unit
        return torch.min(result_addition, dim=-1).values - torch.max(result_addition, dim=-1).values
