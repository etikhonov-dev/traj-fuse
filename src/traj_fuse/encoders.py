"""Continuous-time trajectory and mask-safe molecular/panel encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as fn


def compute_dimensionless_time(
    visit_deltas: Tensor, landmark_lags: Tensor, landmark_time: float
) -> Tensor:
    """Return (B, L, 2) dimensionless [tau_lag, log1p(delta_t)] coordinates."""
    tau_lag = (landmark_lags / max(float(landmark_time), 1e-6)).clamp(0.0, 1.0)
    return torch.stack([tau_lag, torch.log1p(visit_deltas.clamp_min(0.0))], dim=-1)


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


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        b, seq_len, d = x.shape
        q, k, v = (
            self.qkv(self.n1(x))
            .view(b, seq_len, 3, self.n_heads, d // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        attn = fn.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.drop.p if self.training else 0.0
        )
        x = x + self.drop(self.proj(attn.transpose(1, 2).reshape(b, seq_len, d)))
        ff_out: Tensor = self.drop(self.ffn(self.n2(x)))
        return x + ff_out


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
        self.time2vec = Time2Vec(d_model)
        self.token_proj = nn.Linear(2 * d_traj, d_model)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
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
        h = self.token_proj(torch.cat([traj_values, traj_obs_mask], dim=-1)) + self.time2vec(tau)
        b, seq_len = visit_mask.shape
        attn_mask = visit_mask[:, None, None, :].expand(b, 1, seq_len, seq_len)
        for blk in self.blocks:
            h = blk(h, attn_mask)
        h = h * visit_mask.unsqueeze(-1).to(h.dtype)
        scores = self.pool_v(torch.tanh(self.pool_h(h) + self.pool_t(tau))).squeeze(-1)
        alpha = torch.softmax(scores.masked_fill(~visit_mask, -torch.inf), dim=1)
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
        self.stem = nn.Sequential(nn.Linear(2 * d_mol, d_hidden), nn.LayerNorm(d_hidden), nn.SiLU())
        self.blocks = nn.Sequential(
            _ResidualBlock(d_hidden, dropout), _ResidualBlock(d_hidden, dropout)
        )
        self.shared_head = _split_head(d_hidden, d_shared)
        self.priv_head = _split_head(d_hidden, d_priv)

    def forward(self, panel_values: Tensor, panel_obs_mask: Tensor) -> tuple[Tensor, Tensor]:
        h = self.blocks(self.stem(torch.cat([panel_values, panel_obs_mask], dim=-1)))
        return self.shared_head(h), self.priv_head(h)
