"""Tests for tf-probe-stability Task 5: Cox/IPCW probes, scarcity & missing-panel ablation."""

from __future__ import annotations

import numpy as np
import pytest

from traj_fuse.data import LandmarkedCohortSplits, load_pbc2_cohort
from traj_fuse.model import SharedPrivateConfig
from traj_fuse.probe import (
    breslow_nll_and_grad,
    concordance_index,
    evaluate_missing_panel_ablation,
    evaluate_scarcity_probes,
    fit_ipcw_linear_probe,
    fit_ridge_cox,
)
from traj_fuse.train import FrozenSSLCheckpoint, pretrain_ssl


def test_concordance_index_exact_ties_and_empty_error() -> None:
    dur = np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float64)
    ev = np.array([1, 1, 0, 1], dtype=np.int64)
    risk = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float64)
    # Comparable pairs: (0,1), (0,2), (0,3), (1,2) [tied time, j=censored], (1,3) -> 5 pairs
    assert concordance_index(dur, ev, risk) == pytest.approx(1.0)

    tied_risk = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    assert concordance_index(dur, ev, tied_risk) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="comparable"):
        concordance_index(np.array([1.0, 2.0]), np.array([0, 0]), np.array([1.0, 0.0]))


def test_breslow_tie_invariance_and_extreme_logit_stability() -> None:
    rng = np.random.default_rng(7)
    n, d = 60, 6
    x = rng.normal(0.0, 1.0, (n, d))
    # Heavy ties: only 5 discrete durations
    dur = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n).astype(np.float64)
    ev = (rng.random(n) < 0.65).astype(np.int64)

    m1 = fit_ridge_cox(x, dur, ev, lam=1.0)
    perm = rng.permutation(n)
    m2 = fit_ridge_cox(x[perm], dur[perm], ev[perm], lam=1.0)
    assert np.allclose(m1.predict_risk(x), m2.predict_risk(x), atol=1e-6)

    extreme_eta = np.linspace(-500.0, 500.0, n, dtype=np.float64)
    nll, grad = breslow_nll_and_grad(extreme_eta, dur, ev)
    assert np.isfinite(nll)
    assert np.isfinite(grad).all()


def test_ipcw_linear_probe_ranks_higher_risk_for_shorter_survival() -> None:
    rng = np.random.default_rng(13)
    n = 150
    x = rng.normal(0.0, 1.0, (n, 4))
    signal = x[:, 0] - 0.5 * x[:, 1]
    t_ev = np.exp(-0.8 * signal + rng.normal(0.0, 0.2, n))
    t_cens = rng.uniform(0.5, 4.0, n)
    dur = np.minimum(t_ev, t_cens)
    ev = (t_ev <= t_cens).astype(np.int64)

    probe = fit_ipcw_linear_probe(x, dur, ev, lam=1.0)
    c_idx = concordance_index(dur, ev, probe.predict_risk(x))
    assert c_idx > 0.78


def test_frozen_probe_immutability_zero_val_leakage_and_tier1_scarcity(
    tier1_splits_and_ckpt: tuple[LandmarkedCohortSplits, FrozenSSLCheckpoint],
) -> None:
    splits, ckpt = tier1_splits_and_ckpt
    before_sha = ckpt.parameter_sha256()

    # Poison validation outcomes to verify zero val label leakage during probing/CV
    orig_val_ev = splits.val.event.copy()
    poisoned_val_ev = orig_val_ev.copy()
    poisoned_val_ev[:] = 0
    splits.val.event[:] = 0
    try:
        report = evaluate_scarcity_probes(
            splits,
            ckpt,
            fractions=(0.01, 0.05, 0.10, 1.00),
            n_draws=4,
            seed=42,
            e2e_epochs=5,
        )
        assert ckpt.parameter_sha256() == before_sha
        assert np.array_equal(splits.val.event, poisoned_val_ev)
    finally:
        splits.val.event[:] = orig_val_ev

    by_frac = {round(pt.fraction, 4): pt for pt in report.points}
    for f in (0.01, 0.05, 0.10, 1.00):
        assert by_frac[f].identifiable is True

    assert by_frac[1.00].traj_fuse_cox_c_index is not None
    assert by_frac[1.00].traj_fuse_ipcw_c_index is not None
    assert by_frac[1.00].last_visit_cox_c_index is not None
    assert by_frac[1.00].supervised_e2e_c_index is not None
    assert by_frac[1.00].traj_fuse_cox_c_index > 0.74
    assert by_frac[1.00].traj_fuse_ipcw_c_index > 0.74
    assert by_frac[1.00].traj_fuse_cox_c_index > by_frac[1.00].last_visit_cox_c_index
    assert by_frac[0.01].traj_fuse_cox_c_index is not None
    assert by_frac[0.01].last_visit_cox_c_index is not None
    assert by_frac[0.05].traj_fuse_cox_c_index is not None
    assert by_frac[0.05].last_visit_cox_c_index is not None
    assert by_frac[0.01].traj_fuse_cox_c_index > by_frac[0.01].last_visit_cox_c_index
    assert by_frac[0.05].traj_fuse_cox_c_index > by_frac[0.05].last_visit_cox_c_index


def test_pbc2_identifiability_gate_and_missing_panel_pdrop_gain() -> None:
    splits = load_pbc2_cohort(landmark_time=2.0, seed=42)
    cfg_drop = SharedPrivateConfig.for_cohort(splits, p_drop=0.30)
    cfg_nodrop = SharedPrivateConfig.for_cohort(splits, p_drop=0.0)
    ckpt_drop = pretrain_ssl(
        splits.train.to_ssl_split(), cfg_drop, epochs=20, batch_size=64, seed=42
    )
    ckpt_nodrop = pretrain_ssl(
        splits.train.to_ssl_split(), cfg_nodrop, epochs=20, batch_size=64, seed=42
    )

    scarcity = evaluate_scarcity_probes(
        splits,
        ckpt_drop,
        fractions=(0.01, 0.05, 0.10, 0.25, 1.00),
        n_draws=2,
        seed=42,
        e2e_epochs=8,
    )
    by_frac = {round(pt.fraction, 4): pt for pt in scarcity.points}
    assert by_frac[0.01].identifiable is False
    assert "min_identifiable" in (by_frac[0.01].skip_reason or "")
    for f in (0.05, 0.10, 0.25, 1.00):
        assert by_frac[f].identifiable is True

    ablation = evaluate_missing_panel_ablation(splits, ckpt_drop, ckpt_nodrop, seed=42)
    assert ablation.pdrop.c_index_drop100 > ablation.nodrop.c_index_drop100
