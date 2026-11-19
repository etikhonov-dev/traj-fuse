"""Tests for tf-probe-stability: CCA null, Haar invariance, disentanglement & seed stability."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from traj_fuse.data import LandmarkedCohortSplits, load_pbc2_cohort
from traj_fuse.model import SharedPrivateConfig
from traj_fuse.probe import (
    evaluate_seed_stability,
    evaluate_subspace_disentanglement,
    linear_cka,
    procrustes_cosine,
    raw_unaligned_cosine,
    ridge_cca_correlations,
)
from traj_fuse.train import FrozenSSLCheckpoint, pretrain_ssl


def _random_haar_orthogonal(dim: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(0.0, 1.0, (dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs).astype(np.float64)


def test_white_noise_cca_null_and_dimension_guard() -> None:
    rng = np.random.default_rng(101)
    x = rng.normal(0.0, 1.0, (50, 8))
    y = rng.normal(0.0, 1.0, (50, 8))
    res = ridge_cca_correlations(x, y, reg=1e-2, n_perm=20, seed=101)
    assert res.mean_cca < 0.45
    assert abs(res.excess_cca) < 0.10

    with pytest.raises(ValueError, match="N - 5"):
        ridge_cca_correlations(rng.normal(size=(15, 6)), rng.normal(size=(15, 6)))


def test_haar_rotation_collapses_raw_cosine_but_preserves_cka_cca_procrustes() -> None:
    rng = np.random.default_rng(202)
    z_a = rng.normal(0.0, 1.0, (120, 8))
    z_b = 0.85 * z_a + rng.normal(0.0, 0.35, (120, 8))
    q = _random_haar_orthogonal(8, seed=99)
    z_b_rot = z_b @ q

    cka_orig = linear_cka(z_a, z_b)
    cka_rot = linear_cka(z_a, z_b_rot)
    cca_orig = ridge_cca_correlations(z_a, z_b, seed=7).mean_cca
    cca_rot = ridge_cca_correlations(z_a, z_b_rot, seed=7).mean_cca
    proc_orig = procrustes_cosine(z_a, z_b)
    proc_rot = procrustes_cosine(z_a, z_b_rot)

    assert np.isclose(cka_orig, cka_rot, atol=1e-5)
    assert np.isclose(cca_orig, cca_rot, atol=1e-5)
    assert np.isclose(proc_orig, proc_rot, atol=1e-5)

    assert raw_unaligned_cosine(z_a, z_a @ q) < 0.45
    assert procrustes_cosine(z_a, z_a @ q) == pytest.approx(1.0, abs=1e-5)


def test_tier1_subspace_disentanglement_and_3seed_pbc2_stability(
    tier1_splits_and_ckpt: tuple[LandmarkedCohortSplits, FrozenSSLCheckpoint],
) -> None:
    syn, syn_ckpt = tier1_splits_and_ckpt
    dis = evaluate_subspace_disentanglement(syn, syn_ckpt, seed=42)

    assert dis.cca_shared_to_true_shared.mean_cca > 0.45
    assert dis.cca_priv_traj_to_true_traj.mean_cca > dis.cca_priv_traj_to_true_mol.mean_cca + 0.15
    assert dis.cca_priv_mol_to_true_mol.mean_cca > dis.cca_priv_mol_to_true_traj.mean_cca + 0.15

    pbc = load_pbc2_cohort(landmark_time=2.0, seed=42)
    pbc_cfg = SharedPrivateConfig.for_cohort(pbc, p_drop=0.30)
    ckpts = {
        s: pretrain_ssl(pbc.train.to_ssl_split(), pbc_cfg, epochs=16, batch_size=64, seed=s)
        for s in (11, 22, 33)
    }
    stab = evaluate_seed_stability(pbc, ckpts, seed=42)
    assert stab.mean_linear_cka > 0.62
    assert stab.mean_procrustes_cosine > 0.70
    assert stab.mean_excess_cca > 0.25
    assert stab.mean_procrustes_cosine > stab.mean_raw_unaligned_cosine + 0.30
