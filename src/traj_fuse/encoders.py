"""Continuous-time trajectory and mask-safe molecular/panel encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def compute_dimensionless_time(
    visit_deltas: Tensor, landmark_lags: Tensor, landmark_time: float
) -> Tensor:
    """Return (B, L, 2) dimensionless [tau_lag, log1p(delta_t)] coordinates."""
    scale = max(float(landmark_time), 1e-6)
    tau_lag = (landmark_lags / scale).clamp(0.0, 1.0)
    tau_delta = torch.log1p(visit_deltas.clamp_min(0.0))
    return torch.stack([tau_lag, tau_delta], dim=-1)


class Time2Vec(nn.Module):
    """Dimensionless continuous-time embedding: [linear(tau), sin(omega * tau + phi)]."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        if d_model < 4:
            raise ValueError("Time2Vec requires d_model >= 4.")
        self.linear = nn.Linear(2, 2)
        self.periodic = nn.Linear(2, d_model - 2)

    def forward(self, tau: Tensor) -> Tensor:
        return torch.cat([self.linear(tau), torch.sin(self.periodic(tau))], dim=-1)


def _split_head(d_in: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_in), nn.SiLU(), nn.Linear(d_in, d_out))


class IrregularTrajectoryEncoder(nn.Module):
    """Bidirectional pre-landmark Transformer with -inf-masked landmark-lag pooling."""

    def __init__(
        self,
        d_traj: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_shared: int = 4,
        d_priv: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_traj = d_traj
        self.time2vec = Time2Vec(d_model)
        self.token_proj = nn.Linear(2 * d_traj, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.pool_h = nn.Linear(d_model, d_model)
        self.pool_t = nn.Linear(2, d_model, bias=False)
        self.pool_v = nn.Linear(d_model, 1, bias=False)
        self.shared_head = _split_head(d_model, d_shared)
        self.priv_head = _split_head(d_model, d_priv)

    def forward(
        self,
        traj_values: Tensor,
        traj_obs_mask: Tensor,
        visit_deltas: Tensor,
        landmark_lags: Tensor,
        visit_mask: Tensor,
        landmark_time: float,
    ) -> tuple[Tensor, Tensor]:
        tau = compute_dimensionless_time(visit_deltas, landmark_lags, landmark_time)
        tokens = self.token_proj(torch.cat([traj_values, traj_obs_mask], dim=-1))
        tokens = tokens + self.time2vec(tau)
        h = self.transformer(tokens, src_key_padding_mask=~visit_mask)
        h = h * visit_mask.unsqueeze(-1).to(h.dtype)
        scores = self.pool_v(torch.tanh(self.pool_h(h) + self.pool_t(tau))).squeeze(-1)
        scores = scores.masked_fill(~visit_mask, -torch.inf)
        alpha = torch.softmax(scores, dim=1)
        h_traj = torch.sum(alpha.unsqueeze(-1) * h, dim=1)
        return self.shared_head(h_traj), self.priv_head(h_traj)


class _ResidualBlock(nn.Module):
    def __init__(self, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
        )

    def forward(self, x: Tensor) -> Tensor:
        delta: Tensor = self.net(x)
        return x + delta


class MolecularEncoder(nn.Module):
    """Mask-safe residual MLP encoder (zero dropout on binary input masks)."""

    def __init__(
        self,
        d_mol: int,
        d_hidden: int = 64,
        d_shared: int = 4,
        d_priv: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(2 * d_mol, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            _ResidualBlock(d_hidden, dropout),
            _ResidualBlock(d_hidden, dropout),
        )
        self.shared_head = _split_head(d_hidden, d_shared)
        self.priv_head = _split_head(d_hidden, d_priv)

    def forward(self, panel_values: Tensor, panel_obs_mask: Tensor) -> tuple[Tensor, Tensor]:
        h = self.blocks(self.stem(torch.cat([panel_values, panel_obs_mask], dim=-1)))
        return self.shared_head(h), self.priv_head(h)
