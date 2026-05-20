import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd

class TropicalLayer( nn.Module ):
    def __init__( self, in_features, out_features ):
        super( TropicalLayer, self ).__init__ ()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter( torch.randn( out_features, in_features ))
        nn.init.normal_( self. weight )

    def forward ( self, x):
        """ Returns negative tropical distance between x and self . weight . """
        result_addition = x.unsqueeze(1) - self.weight # [B, 1 , in] - [out , in] -> [B, out , in]
        return torch.min( result_addition, dim = -1).values - torch.max( result_addition, dim = -1).values # [B, out , in] -> [B, out ]


class SimpleRegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleRegressionNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


class SimpleRegressionTNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleRegressionTNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            TropicalLayer(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


class DeepRegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(DeepRegressionNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepRegressionTNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(DeepRegressionTNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            prev_dim = h_dim
        layers.append(TropicalLayer(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    
class GRURegressor(nn.Module):
    def __init__(self,
                 seq_len,
                 input_size=1,
                 hidden_size=512,
                 num_layers=2,
                 bidirectional=True,
                 dropout=0.2):
        super(GRURegressor, self).__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0.0)
        factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x is shape [batch, seq_len]; treat each feature as one step
        x = x.unsqueeze(-1)            # → [batch, seq_len, 1]
        gru_out, _ = self.gru(x)       # → [batch, seq_len, hidden_size * factor]
        last = gru_out[:, -1, :]       # pick the last time step
        return self.head(last).squeeze(1)

class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_blocks=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            out = block(x)
            x = x + out        # residual connection
        return self.head(x).squeeze(1)

class TabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Linear(1, embed_dim)  # each scalar → vector
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dropout=dropout, dim_feedforward=embed_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # self.head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim//2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim//2, 1)
        # )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, 1)
        )


    def forward(self, x):
        # x: [batch, D] → [batch, D, 1] → embed to [batch, D, E]
        emb = self.token_emb(x.unsqueeze(-1))
        # transformer expects [seq_len, batch, embed]
        t = emb.permute(1, 0, 2)
        out = self.transformer(t)
        # out: [D, batch, E] → pool across D
        pooled = out.mean(dim=0)
        return self.head(pooled).squeeze(1)
    
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),           # original
            nn.Linear(512, 128), nn.ReLU(),                   # new intermediate layer
            nn.Linear(128, bottleneck) #, nn.ReLU()             # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(),            # mirror of encoder
            nn.Linear(128, 512), nn.ReLU(),                   # mirror
            nn.Linear(512, input_dim)                        # output
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class AERegressor(nn.Module):
    def __init__(self, pretrained_encoder, bottleneck=64, dropout=0.1):
        super().__init__()
        self.encoder = pretrained_encoder
        self.head = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    # def forward(self, x):
    #     with torch.no_grad():   # optionally freeze encoder
    #         z = self.encoder(x)
    #     return self.head(z).squeeze(1)
    
    def forward(self, x):
        # with torch.no_grad():   # optionally freeze encoder
        z = self.encoder(x)
        return self.head(z).squeeze(1)


class DLRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 model_class,
                 model_kwargs=None,
                 epochs=100,
                 batch_size=32,
                 learning_rate=1e-3,
                 verbose=False):
        # we no longer need input_dim here, it lives in model_kwargs
        self.model_class  = model_class
        self.model_kwargs = model_kwargs or {}
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.learning_rate= learning_rate
        self.verbose      = verbose
        self.scaler       = StandardScaler()
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model        = None

    def fit(self, X, y):
        # scale
        Xs = self.scaler.fit_transform(X)
        Ys = np.array(y).reshape(-1, 1)

        # decide if it's a sequence‐model
        is_sequence = issubclass(self.model_class, GRURegressor)

        # build tensors
        Xt = torch.tensor(Xs, dtype=torch.float32)
        if is_sequence:
            Xt = Xt.unsqueeze(2)    # [N, D] → [N, D, 1]
        Yt = torch.tensor(Ys, dtype=torch.float32).to(self.device)

        # data loader
        ds     = TensorDataset(Xt.to(self.device), Yt)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # instantiate
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        if is_sequence and hasattr(self.model, 'gru'):
            self.model.gru.flatten_parameters()

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # training loop
        self.model.train()
        for epoch in range(1, self.epochs+1):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                preds = self.model(bx)                     # [batch] or [batch,1]
                loss  = criterion(preds.unsqueeze(1), by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs} — loss: {total_loss/len(loader):.4f}")
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32)
        if issubclass(self.model_class, GRURegressor):
            Xt = Xt.unsqueeze(2)
        Xt = Xt.to(self.device)

        self.model.eval()
        preds_list = []
        with torch.no_grad():
            for i in range(0, len(Xt), self.batch_size):
                batch = Xt[i:i+self.batch_size]
                p = self.model(batch)
                preds_list.append(p.cpu().numpy())
        return np.concatenate(preds_list).flatten()
