import torch.nn as nn

# Define a GRU-based regression model using sequence data
class GRURegressor(nn.Module):
    def __init__(self,
                 seq_len,                  # Length of the input sequences
                 input_size=1,             # Dimensionality of each input element (default 1 for univariate)
                 hidden_size=512,          # Number of features in the GRU hidden state
                 num_layers=2,             # Number of stacked GRU layers
                 bidirectional=True,       # Whether to use a bidirectional GRU
                 dropout=0.2):             # Dropout rate between GRU layers (if num_layers > 1)
        super(GRURegressor, self).__init__()

        self.seq_len = seq_len            # Save sequence length

        # Define the GRU layer
        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers=num_layers,
                          batch_first=True,      # Input/output shape: [batch, seq_len, input_size]
                          bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0.0)  # Dropout only if num_layers > 1

        # Factor to account for bidirectional GRU doubling the hidden size
        factor = 2 if bidirectional else 1

        # Define the output head: maps GRU output to a single regression value
        self.head = nn.Sequential(
            nn.Linear(hidden_size * factor, 64),  # First linear layer: GRU output → 64
            nn.ReLU(),                            # ReLU activation
            nn.Dropout(dropout),                  # Dropout for regularization
            nn.Linear(64, 1)                      # Final linear layer: 64 → 1 output value
        )

    def forward(self, x):
        # x is shape [batch, seq_len]; each feature is treated as one time step
        x = x.unsqueeze(-1)                # Add input_size=1 dimension → [batch, seq_len, 1]

        gru_out, _ = self.gru(x)           # GRU forward pass → output of all time steps
                                           # gru_out shape: [batch, seq_len, hidden_size * factor]

        last = gru_out[:, -1, :]           # Take output from the last time step only

        return self.head(last).squeeze(1)  # Pass through head and remove singleton dim → [batch]