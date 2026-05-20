"""
Adaptive Nonlinear Vector Autoregression (Adaptive NVAR) architectures.

These models predict the residual `Δx_t = x_t − x_{t-1}` from a delay-embedded
context `H_lin = [x_{t-k}, …, x_{t-1}]`. They are pure `nn.Module`s; training,
warmup, and autoregressive rollout live in `MLModels.training.timeseries_nvar`.

Two variants are provided:

  * ``AdaptiveNVARModel`` — a learnable MLP feature block.
  * ``AdaptiveConnectomeNVARModel`` — replaces the MLP with a fixed (or
    randomized) connectome adjacency, scaled by a learnable scalar α.

Both feed the concatenation `[H_lin, H_feature]` through a linear readout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Adaptive NVAR with a learnable MLP feature block
# ---------------------------------------------------------------------------


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveNVARModel(nn.Module):
    """Adaptive NVAR: H_total = [H_lin, MLP(H_lin)] -> linear readout to Δx."""

    def __init__(self, dk: int, m: int, d: int, hidden_dim: int) -> None:
        super().__init__()
        self.dk = int(dk)
        self.m = int(m)
        self.d = int(d)
        self.hidden_dim = int(hidden_dim)
        self.mlp = FeatureMLP(self.dk, self.hidden_dim, self.m)
        self.readout = nn.Linear(self.dk + self.m, self.d, bias=False)

    def forward(self, H_lin: torch.Tensor) -> torch.Tensor:
        H_nn = self.mlp(H_lin)
        H_total = torch.cat([H_lin, H_nn], dim=-1)
        return self.readout(H_total)


# ---------------------------------------------------------------------------
# Adaptive NVAR with a fixed connectome feature block
# ---------------------------------------------------------------------------


class ConnectomeFeatureLayer(nn.Module):
    """Replace the MLP with `tanh(α · A · W·H_lin + b)`.

    A is a fixed adjacency (loaded by the trainer); α is a learnable scalar
    parameterized via `log_alpha` so it stays positive.
    """

    def __init__(
        self,
        input_dim: int,
        n_connectome: int,
        connectome_adj: np.ndarray,
        input_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, n_connectome, bias=True)
        nn.init.uniform_(self.input_proj.weight, -input_scale, input_scale)
        nn.init.zeros_(self.input_proj.bias)

        A = np.asarray(connectome_adj, dtype=np.float32)
        if A.shape != (n_connectome, n_connectome):
            raise ValueError(
                f"connectome_adj has shape {A.shape}, expected "
                f"{(n_connectome, n_connectome)}"
            )

        # Buffers move with .to(device) but are not optimized.
        self.register_buffer(
            "A_connectome",
            torch.tensor(A, dtype=torch.float32),
        )
        self.log_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n_connectome, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    def forward(self, H_lin: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(H_lin)
        Az = torch.matmul(z, self.A_connectome.T)
        return torch.tanh(self.alpha * Az + self.bias)


class AdaptiveConnectomeNVARModel(nn.Module):
    """Adaptive NVAR with a connectome-shaped feature block."""

    def __init__(
        self,
        dk: int,
        d: int,
        n_connectome: int,
        connectome_adj: np.ndarray,
        input_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.dk = int(dk)
        self.d = int(d)
        self.n_connectome = int(n_connectome)
        self.feature_layer = ConnectomeFeatureLayer(
            input_dim=self.dk,
            n_connectome=self.n_connectome,
            connectome_adj=connectome_adj,
            input_scale=input_scale,
        )
        self.readout = nn.Linear(self.dk + self.n_connectome, self.d, bias=False)

    def forward(self, H_lin: torch.Tensor) -> torch.Tensor:
        H_conn = self.feature_layer(H_lin)
        H_total = torch.cat([H_lin, H_conn], dim=-1)
        return self.readout(H_total)


# ---------------------------------------------------------------------------
# Helpers used by both the training loop and rollout
# ---------------------------------------------------------------------------


def construct_H_lin(X: torch.Tensor, k: int) -> torch.Tensor:
    """Build delay-embedded matrix H_lin from a [T, d] series.

    Returns a [T-k, d*k] tensor. Row i contains [x_i, x_{i+1}, …, x_{i+k-1}],
    so that the prediction target Y_i = x_{i+k} − x_{i+k-1} aligns naturally.
    """
    if X.dim() != 2:
        raise ValueError(f"Expected [T, d] tensor, got shape {tuple(X.shape)}.")
    T = int(X.shape[0])
    if k < 1:
        raise ValueError("k must be >= 1")
    if T <= k:
        raise ValueError(f"Series of length {T} is too short for delay-embedding k={k}.")
    return torch.cat([X[i : T - k + i] for i in range(k)], dim=1)


def init_weights_stable(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
