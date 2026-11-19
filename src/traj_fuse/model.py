"""SharedPrivateFuse modality-gated fusion and two-view SSL objective."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torch.nn import functional as fn

from traj_fuse.encoders import (
    IrregularTrajectoryEncoder,
    MolecularEncoder,
    Time2Vec,
    compute_dimensionless_time,
)

if TYPE_CHECKING:
    from traj_fuse.data import LandmarkedCohortSplits, SSLPretrainSplit


class SharedPrivateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    d_traj: int = Field(ge=1)
    d_mol: int = Field(ge=1)
    d_shared: int = Field(default=4, ge=1)
    d_priv: int = Field(default=4, ge=1)
    d_model: int = Field(default=64, ge=8)
    n_heads: int = Field(default=4, ge=1)
    n_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    p_mask: float = Field(default=0.20, ge=0.0, lt=1.0)
    p_drop: float = Field(default=0.30, ge=0.0, le=1.0)
    lambda_vic: float = Field(default=1.0, ge=0.0)
    lambda_orth: float = Field(default=0.5, ge=0.0)
    lambda_recon: float = Field(default=1.0, ge=0.0)
    lambda_drop: float = Field(default=1.0, ge=0.0)

    @property
    def d_fused(self) -> int:
        return self.d_shared + 2 * self.d_priv

    @classmethod
    def for_cohort(
        cls, cohort: LandmarkedCohortSplits | SSLPretrainSplit, *, p_drop: float = 0.30
    ) -> SharedPrivateConfig:
        ssl = cohort.train.ssl if hasattr(cohort, "train") else cohort
        d_traj, d_mol = int(ssl.traj_values.shape[2]), int(ssl.panel_values.shape[1])
        d_priv = 4 if d_mol >= 16 else min(2, max(1, d_mol - 4))
        return cls(d_traj=d_traj, d_mol=d_mol, d_shared=4, d_priv=d_priv, p_drop=p_drop)


class FusionOutputs(NamedTuple):
    z_traj_shared: Tensor
    z_traj_priv: Tensor
    z_mol_shared: Tensor
    z_mol_priv: Tensor
    z_shared: Tensor
    z_mol_priv_tilde: Tensor
    z_fused: Tensor


class SharedPrivateFuse(nn.Module):
    """Modality-gated shared-private fusion with zero-init residual adapter."""

    def __init__(self, config: SharedPrivateConfig) -> None:
        super().__init__()
        self.config = config
        self.traj_enc = IrregularTrajectoryEncoder(
            config.d_traj,
            config.d_model,
            config.n_heads,
            config.n_layers,
            config.d_shared,
            config.d_priv,
            config.dropout,
        )
        self.mol_enc = MolecularEncoder(
            config.d_mol, config.d_model, config.d_shared, config.d_priv, config.dropout
        )
        self.b_miss_mol = nn.Parameter(torch.zeros(config.d_priv))
        d_fused = config.d_fused
        last = nn.Linear(d_fused, d_fused)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.mlp_fuse = nn.Sequential(nn.Linear(d_fused + 1, d_fused), nn.SiLU(), last)

    def forward(
        self,
        traj_values: Tensor,
        traj_obs_mask: Tensor,
        visit_deltas: Tensor,
        landmark_lags: Tensor,
        visit_mask: Tensor,
        landmark_time: float,
        panel_values: Tensor,
        panel_obs_mask: Tensor,
        active_panel: Tensor,
    ) -> FusionOutputs:
        m = active_panel.to(traj_values.dtype).unsqueeze(-1)
        z_ts, z_tp = self.traj_enc(
            traj_values, traj_obs_mask, visit_deltas, landmark_lags, visit_mask, landmark_time
        )
        z_ms, z_mp = self.mol_enc(panel_values * m, panel_obs_mask * m)
        z_shared = (z_ts + m * z_ms) / (1.0 + m)
        z_mp_tilde = m * z_mp + (1.0 - m) * self.b_miss_mol.unsqueeze(0)
        concat = torch.cat([z_shared, z_tp, z_mp_tilde], dim=-1)
        delta_fuse: Tensor = self.mlp_fuse(torch.cat([concat, m], dim=-1))
        return FusionOutputs(z_ts, z_tp, z_ms, z_mp, z_shared, z_mp_tilde, concat + delta_fuse)


def _vicreg_var_cov(z: Tensor) -> Tensor:
    if z.shape[0] < 4:
        return z.new_tensor(0.0)
    std = torch.sqrt(z.var(dim=0, unbiased=True) + 1e-4)
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / float(z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return 25.0 * fn.relu(1.0 - std).mean() + off.square().sum() / float(z.shape[1])


def _cross_orth(a: Tensor, b: Tensor) -> Tensor:
    if a.shape[0] < 4:
        return a.new_tensor(0.0)
    ac, bc = a - a.mean(dim=0, keepdim=True), b - b.mean(dim=0, keepdim=True)
    return ((ac.T @ bc) / float(a.shape[0] - 1)).square().sum()


def _sample_coord_mask(obs_mask: Tensor, p_mask: float, gen: torch.Generator) -> Tensor:
    drop = (torch.rand(obs_mask.shape, generator=gen, device=obs_mask.device) < p_mask) & obs_mask
    if obs_mask.any() and not drop.any():
        drop[tuple(int(x) for x in torch.nonzero(obs_mask, as_tuple=False)[0])] = True
    return drop


class SSLPretrainModule(nn.Module):
    """Two-view clean/corrupted SSL wrapper around SharedPrivateFuse."""

    def __init__(self, config: SharedPrivateConfig) -> None:
        super().__init__()
        self.config = config
        self.fuse = SharedPrivateFuse(config)
        self.dec_t2v = Time2Vec(16)
        self.traj_dec = nn.Sequential(
            nn.Linear(config.d_fused + 16, 64), nn.SiLU(), nn.Linear(64, config.d_traj)
        )
        self.panel_dec = nn.Sequential(
            nn.Linear(config.d_fused, 64), nn.SiLU(), nn.Linear(64, config.d_mol)
        )

    def forward_loss(
        self,
        traj_values: Tensor,
        traj_obs_mask: Tensor,
        visit_deltas: Tensor,
        landmark_lags: Tensor,
        visit_mask: Tensor,
        landmark_time: float,
        panel_values: Tensor,
        panel_obs_mask: Tensor,
        panel_available: Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[Tensor, dict[str, float]]:
        cfg, m_obs = self.config, panel_available.to(traj_values.dtype)
        clean = self.fuse(
            traj_values,
            traj_obs_mask,
            visit_deltas,
            landmark_lags,
            visit_mask,
            landmark_time,
            panel_values,
            panel_obs_mask,
            m_obs,
        )
        n_p = int(panel_available.sum().item())
        l_vic = _vicreg_var_cov(clean.z_traj_shared) + _vicreg_var_cov(clean.z_traj_priv)
        if n_p >= 1:
            l_vic = l_vic + 25.0 * fn.mse_loss(
                clean.z_traj_shared[panel_available], clean.z_mol_shared[panel_available]
            )
        if n_p >= 4:
            l_vic = (
                l_vic
                + _vicreg_var_cov(clean.z_mol_shared[panel_available])
                + _vicreg_var_cov(clean.z_mol_priv[panel_available])
            )
        l_orth = _cross_orth(clean.z_traj_shared, clean.z_traj_priv)
        if n_p >= 4:
            l_orth = (
                l_orth
                + _cross_orth(
                    clean.z_mol_shared[panel_available], clean.z_mol_priv[panel_available]
                )
                + _cross_orth(clean.z_traj_priv[panel_available], clean.z_mol_priv[panel_available])
            )

        obs_t = (traj_obs_mask > 0.5) & visit_mask.unsqueeze(-1)
        drop_t = _sample_coord_mask(obs_t, cfg.p_mask, generator)
        obs_p = panel_obs_mask > 0.5
        drop_p = _sample_coord_mask(obs_p, cfg.p_mask, generator)
        m_drop = m_obs * (
            torch.rand(m_obs.shape, generator=generator, device=m_obs.device) >= cfg.p_drop
        ).to(m_obs.dtype)
        corrupt = self.fuse(
            traj_values.masked_fill(drop_t, 0.0),
            traj_obs_mask.masked_fill(drop_t, 0.0),
            visit_deltas,
            landmark_lags,
            visit_mask,
            landmark_time,
            panel_values.masked_fill(drop_p, 0.0) * m_drop.unsqueeze(-1),
            panel_obs_mask.masked_fill(drop_p, 0.0) * m_drop.unsqueeze(-1),
            m_drop,
        )
        l_recon = traj_values.new_tensor(0.0)
        if drop_t.any():
            b_idx, l_idx = torch.nonzero(drop_t.any(dim=-1), as_tuple=True)
            tau = compute_dimensionless_time(visit_deltas, landmark_lags, landmark_time)
            dec_in = torch.cat([corrupt.z_fused[b_idx], self.dec_t2v(tau[b_idx, l_idx])], dim=-1)
            pred_t: Tensor = self.traj_dec(dec_in)
            l_recon = l_recon + fn.mse_loss(
                pred_t[drop_t[b_idx, l_idx]], traj_values[b_idx, l_idx][drop_t[b_idx, l_idx]]
            )
        pred_p: Tensor = self.panel_dec(corrupt.z_fused)
        dropped_p = panel_available & (~m_drop.bool())
        recon_p = torch.where(m_drop.bool().unsqueeze(-1), drop_p, dropped_p.unsqueeze(-1) & obs_p)
        if recon_p.any():
            l_recon = l_recon + fn.mse_loss(pred_p[recon_p], panel_values[recon_p])
        l_drop = (
            fn.mse_loss(corrupt.z_fused[dropped_p], clean.z_fused[dropped_p].detach())
            if dropped_p.any()
            else traj_values.new_tensor(0.0)
        )
        total = (
            cfg.lambda_vic * l_vic
            + cfg.lambda_orth * l_orth
            + cfg.lambda_recon * l_recon
            + cfg.lambda_drop * l_drop
        )
        metrics = {
            "loss": float(total.item()),
            "vicreg": float(l_vic.item()),
            "orth": float(l_orth.item()),
            "recon": float(l_recon.item()),
            "drop": float(l_drop.item()),
        }
        return total, metrics
