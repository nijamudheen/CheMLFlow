import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck=64):
        super().__init__()

        # Encoder: compresses input down to the bottleneck representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),     # First layer: input → 512
            nn.Linear(512, 128), nn.ReLU(),           # Intermediate layer: 512 → 128
            nn.Linear(128, bottleneck)                # Bottleneck layer: 128 → bottleneck
            # Optional ReLU after bottleneck could be added depending on use case
        )

        # Decoder: reconstructs input from bottleneck representation
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(),    # Mirror of encoder: bottleneck → 128
            nn.Linear(128, 512), nn.ReLU(),           # Mirror: 128 → 512
            nn.Linear(512, input_dim)                 # Output layer: 512 → original input dimension
        )

    # Forward pass: encode → decode
    def forward(self, x):
        z = self.encoder(x)          # Encode input to latent representation z
        return self.decoder(z)       # Reconstruct input from z


class AERegressor(nn.Module):
    def __init__(self, pretrained_encoder, bottleneck=64, dropout=0.1):
        super().__init__()

        self.encoder = pretrained_encoder  # Use an existing (possibly pretrained) encoder

        # Regression head: maps encoded features (bottleneck) to a single output
        self.head = nn.Sequential(
            nn.Linear(bottleneck, 128),     # Bottleneck → hidden layer
            nn.ReLU(),                      # ReLU activation
            nn.Dropout(dropout),            # Dropout for regularization
            nn.Linear(128, 1)               # Output layer: hidden → scalar prediction
        )

    # Forward method for prediction
    def forward(self, x):
        # Optionally freeze encoder by wrapping it in torch.no_grad()
        # with torch.no_grad():
        z = self.encoder(x)                 # Extract latent features from encoder
        return self.head(z).squeeze(1)     # Predict and squeeze output to shape [batch]
