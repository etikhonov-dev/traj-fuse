"""Shape, gradient, padding-invariance, and anti-collapse tests for tf-shared-private."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from traj_fuse.data import make_synthetic_subspace_cohort
from traj_fuse.encoders import IrregularTrajectoryEncoder, MolecularEncoder, Time2Vec
from traj_fuse.model import SharedPrivateConfig, SharedPrivateFuse, SSLPretrainModule
from traj_fuse.train import pretrain_ssl


def _make_traj_inputs(
    batch_size: int = 8,
    l_max: int = 12,
    d_traj: int = 8,
    *,
    seed: int = 0,
    single_visit_rows: tuple[int, ...] = (0,),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    traj_values = torch.randn(batch_size, l_max, d_traj, generator=gen)
    traj_obs_mask = (torch.rand(batch_size, l_max, d_traj, generator=gen) > 0.15).float()
    traj_obs_mask[:, 0, :] = 1.0

    visit_mask = torch.zeros(batch_size, l_max, dtype=torch.bool)
    visit_deltas = torch.zeros(batch_size, l_max)
    landmark_lags = torch.zeros(batch_size, l_max)
    for b in range(batch_size):
        m_i = 1 if b in single_visit_rows else int(torch.randint(2, l_max + 1, (1,), generator=gen))
        visit_mask[b, :m_i] = True
        times = torch.cat([torch.zeros(1), torch.sort(torch.rand(m_i - 1, generator=gen) * 1.9)[0]])
        if m_i > 1:
            visit_deltas[b, 1:m_i] = times[1:] - times[:-1]
        landmark_lags[b, :m_i] = 2.0 - times
        traj_values[b, m_i:, :] = 999.0
        traj_obs_mask[b, m_i:, :] = 0.0
    traj_values = torch.where(traj_obs_mask > 0.5, traj_values, torch.zeros_like(traj_values))
    return traj_values, traj_obs_mask, visit_deltas, landmark_lags, visit_mask


def test_time2vec_dimensionless_shapes_and_finite() -> None:
    t2v = Time2Vec(d_model=64)
    tau = torch.rand(4, 10, 2)
    out = t2v(tau)
    assert out.shape == (4, 10, 64)
    assert torch.isfinite(out).all()


def test_trajectory_encoder_padding_width_invariance() -> None:
    torch.manual_seed(7)
    enc = IrregularTrajectoryEncoder(
        d_traj=8, d_model=64, n_heads=4, n_layers=2, d_shared=4, d_priv=4
    )
    enc.eval()
    t_val, t_mask, v_del, l_lag, v_mask = _make_traj_inputs(
        batch_size=6, l_max=10, single_visit_rows=(0, 3)
    )

    pad = 8
    t_val_p = torch.cat([t_val, torch.full((6, pad, 8), 123.0)], dim=1)
    t_mask_p = torch.cat([t_mask, torch.zeros(6, pad, 8)], dim=1)
    v_del_p = torch.cat([v_del, torch.full((6, pad), 5.0)], dim=1)
    l_lag_p = torch.cat([l_lag, torch.full((6, pad), 9.0)], dim=1)
    v_mask_p = torch.cat([v_mask, torch.zeros(6, pad, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        zs_1, zp_1 = enc(t_val, t_mask, v_del, l_lag, v_mask, landmark_time=2.0)
        zs_2, zp_2 = enc(t_val_p, t_mask_p, v_del_p, l_lag_p, v_mask_p, landmark_time=2.0)

    assert torch.allclose(zs_1, zs_2, atol=1e-6)
    assert torch.allclose(zp_1, zp_2, atol=1e-6)


def test_single_and_multi_visit_finite_gradients_and_latency() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(11)
    enc = IrregularTrajectoryEncoder(
        d_traj=8, d_model=64, n_heads=4, n_layers=2, d_shared=4, d_priv=4
    )
    mol_enc = MolecularEncoder(d_mol=256, d_hidden=64, d_shared=4, d_priv=4)
    t_val, t_mask, v_del, l_lag, v_mask = _make_traj_inputs(
        batch_size=16, l_max=15, single_visit_rows=(0, 5)
    )
    v_mask[1, :15] = True
    p_val = torch.randn(16, 256)
    p_mask = (torch.rand(16, 256) > 0.2).float()

    zs_t, zp_t = enc(t_val, t_mask, v_del, l_lag, v_mask, landmark_time=2.0)
    zs_m, zp_m = mol_enc(p_val, p_mask)
    (
        zs_t.square().sum() + zp_t.square().sum() + zs_m.square().sum() + zp_m.square().sum()
    ).backward()
    enc.zero_grad(set_to_none=True)
    mol_enc.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    zs_t, zp_t = enc(t_val, t_mask, v_del, l_lag, v_mask, landmark_time=2.0)
    zs_m, zp_m = mol_enc(p_val, p_mask)
    loss = zs_t.square().sum() + zp_t.square().sum() + zs_m.square().sum() + zp_m.square().sum()
    loss.backward()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert zs_t.shape == (16, 4)
    assert zp_t.shape == (16, 4)
    assert zs_m.shape == (16, 4)
    assert zp_m.shape == (16, 4)
    assert torch.isfinite(zs_t).all() and torch.isfinite(zs_m).all()
    for p in list(enc.parameters()) + list(mol_enc.parameters()):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
    assert elapsed_ms < 25.0


def test_trajectory_encoder_preserves_radial_severity() -> None:
    splits = make_synthetic_subspace_cohort(n_patients=160, d_traj=8, d_mol=16, seed=19)
    cfg = SharedPrivateConfig(d_traj=8, d_mol=16, d_shared=4, d_priv=4, p_drop=0.30)
    ckpt = pretrain_ssl(splits.train.to_ssl_split(), cfg, epochs=12, batch_size=64, seed=19)
    enc = ckpt.model.traj_enc
    ssl = splits.test.to_ssl_split()
    t_mask = torch.from_numpy(ssl.traj_obs_mask[:1]).clone()
    v_del = torch.from_numpy(ssl.visit_deltas[:1])
    l_lag = torch.from_numpy(ssl.landmark_lags[:1])
    v_mask = torch.from_numpy(ssl.visit_mask[:1])
    base = torch.ones((1, t_mask.shape[1], 8), dtype=torch.float32) * 0.5
    scales = (0.5, 1.0, 2.0, 4.0)
    with torch.no_grad():
        z_low, _ = enc(base * scales[0], t_mask, v_del, l_lag, v_mask, landmark_time=2.0)
        z_high, _ = enc(base * scales[-1], t_mask, v_del, l_lag, v_mask, landmark_time=2.0)
        unit_dir = (z_high - z_low) / (z_high - z_low).norm(dim=-1, keepdim=True).clamp_min(1e-8)
        projs = []
        for s in scales:
            zs, _ = enc(base * s, t_mask, v_del, l_lag, v_mask, landmark_time=2.0)
            projs.append(float((zs * unit_dir).sum().item()))

    assert projs[0] < projs[1] < projs[2] < projs[3]
    assert pytest.approx(projs[3], rel=0.05) != projs[0]


def test_pdrop_activates_missing_token_and_fusion_gradients() -> None:
    torch.set_num_threads(1)
    t_val, t_mask, v_del, l_lag, v_mask = _make_traj_inputs(batch_size=16, l_max=8, seed=3)
    p_val = torch.randn(16, 24)
    p_mask = torch.ones(16, 24)
    p_avail = torch.ones(16, dtype=torch.bool)  # fully paired batch

    for p_drop, expect_nonzero_b in ((0.30, True), (0.0, False)):
        torch.manual_seed(5)
        cfg = SharedPrivateConfig(d_traj=8, d_mol=24, d_shared=4, d_priv=4, p_drop=p_drop)
        ssl_mod = SSLPretrainModule(cfg)
        gen = torch.Generator().manual_seed(17)
        loss, _ = ssl_mod.forward_loss(
            t_val, t_mask, v_del, l_lag, v_mask, 2.0, p_val, p_mask, p_avail, generator=gen
        )
        loss.backward()

        assert ssl_mod.fuse.b_miss_mol.grad is not None
        b_grad_norm = float(ssl_mod.fuse.b_miss_mol.grad.abs().sum().item())
        last_fuse = list(ssl_mod.fuse.mlp_fuse.modules())[-1]
        assert isinstance(last_fuse, torch.nn.Linear)
        assert last_fuse.weight.grad is not None
        fuse_grad_norm = float(last_fuse.weight.grad.abs().sum().item())

        if expect_nonzero_b:
            assert b_grad_norm > 0.0
            assert fuse_grad_norm > 0.0
        else:
            assert b_grad_norm == 0.0


@pytest.mark.parametrize("n_paired", [0, 2, 5])
def test_paired_subset_edge_cases_finite(n_paired: int) -> None:
    batch_size = 8
    t_val, t_mask, v_del, l_lag, v_mask = _make_traj_inputs(batch_size=batch_size, l_max=6, seed=9)
    p_val = torch.randn(batch_size, 16)
    p_mask = torch.zeros(batch_size, 16)
    p_avail = torch.zeros(batch_size, dtype=torch.bool)
    if n_paired > 0:
        p_avail[:n_paired] = True
        p_mask[:n_paired] = 1.0

    cfg = SharedPrivateConfig(d_traj=8, d_mol=16, d_shared=4, d_priv=4, p_drop=0.30)
    ssl_mod = SSLPretrainModule(cfg)
    gen = torch.Generator().manual_seed(42)
    loss, metrics = ssl_mod.forward_loss(
        t_val, t_mask, v_del, l_lag, v_mask, 2.0, p_val, p_mask, p_avail, generator=gen
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(np.isfinite(v) for v in metrics.values())


def test_ssl_pretraining_prevents_collapse_on_heldout_eval() -> None:
    splits = make_synthetic_subspace_cohort(n_patients=220, d_traj=8, d_mol=32, seed=21)
    cfg = SharedPrivateConfig(d_traj=8, d_mol=32, d_shared=4, d_priv=4, p_drop=0.30)
    ckpt = pretrain_ssl(splits.train.to_ssl_split(), cfg, epochs=15, batch_size=64, seed=21)
    emb = ckpt.encode(splits.test.to_ssl_split())
    coord_std = np.std(emb.z_fused, axis=0)
    assert float(coord_std.min()) > 0.45
    assert isinstance(ckpt.model, SharedPrivateFuse)
