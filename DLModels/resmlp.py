import torch.nn as nn

# Define a residual MLP (Multi-Layer Perceptron) model for regression
class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_blocks=4, dropout=0.2):
        super().__init__()

        # Project input to the hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Create a list of residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),       # First linear layer in block
                nn.BatchNorm1d(hidden_dim),              # Batch normalization
                nn.ReLU(),                               # ReLU activation
                nn.Dropout(dropout),                     # Dropout for regularization
                nn.Linear(hidden_dim, hidden_dim),       # Second linear layer in block
                nn.BatchNorm1d(hidden_dim),              # Another batch norm
                nn.ReLU(),                               # ReLU again
            )
            for _ in range(n_blocks)                    # Repeat block n_blocks times
        ])

        # Output head for final prediction
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),      # Hidden layer: hidden_dim → hidden_dim // 2
            nn.ReLU(),                                   # ReLU activation
            nn.Dropout(dropout),                         # Dropout
            nn.Linear(hidden_dim // 2, 1)                # Final output layer: → 1 regression value
        )

    def forward(self, x):
        x = self.input_proj(x)                           # Initial projection to hidden_dim

        # Apply each residual block with skip connection
        for block in self.blocks:
            out = block(x)                               # Pass through the block
            x = x + out                                   # Residual connection (skip connection)

        return self.head(x).squeeze(1)                   # Final output, squeezed to shape [batch]
