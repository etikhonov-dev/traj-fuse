"""Shape, gradient, padding-invariance, and anti-collapse tests for tf-shared-private."""

from __future__ import annotations

import time

import pytest
import torch

from traj_fuse.encoders import IrregularTrajectoryEncoder, MolecularEncoder, Time2Vec


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

    # Warm up CPU dispatch before timing
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
    assert elapsed_ms < 25.0  # warm batch latency check (< 5ms typical)


def test_trajectory_encoder_preserves_radial_severity() -> None:
    torch.manual_seed(19)
    enc = IrregularTrajectoryEncoder(
        d_traj=8, d_model=64, n_heads=4, n_layers=2, d_shared=4, d_priv=4
    )
    enc.eval()
    t_val, t_mask, v_del, l_lag, v_mask = _make_traj_inputs(
        batch_size=1, l_max=6, single_visit_rows=()
    )
    v_mask[0, :4] = True
    t_mask[0, :4, :] = 1.0
    base = torch.ones_like(t_val) * 0.5
    scales = (0.5, 1.0, 2.0, 4.0)
    with torch.no_grad():
        unit_dir, _ = enc(base, t_mask, v_del, l_lag, v_mask, landmark_time=2.0)
        unit_dir = unit_dir / unit_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        projs = []
        for s in scales:
            zs, _ = enc(base * s, t_mask, v_del, l_lag, v_mask, landmark_time=2.0)
            projs.append(float((zs * unit_dir).sum().item()))

    assert projs[0] < projs[1] < projs[2] < projs[3]
    assert pytest.approx(projs[3], rel=0.05) != projs[0]
