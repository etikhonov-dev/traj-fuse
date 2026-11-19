"""Shared session-scoped fixtures for Tier-1 cohort & pretrained SSL checkpoint."""

from __future__ import annotations

import pytest

from traj_fuse.data import LandmarkedCohortSplits, make_synthetic_subspace_cohort
from traj_fuse.model import SharedPrivateConfig
from traj_fuse.train import FrozenSSLCheckpoint, pretrain_ssl


@pytest.fixture(scope="session")
def tier1_splits_and_ckpt() -> tuple[LandmarkedCohortSplits, FrozenSSLCheckpoint]:
    splits = make_synthetic_subspace_cohort(
        n_patients=1_400, d_traj=8, d_mol=256, panel_missing_rate=0.25, seed=42
    )
    cfg = SharedPrivateConfig.for_cohort(splits, p_drop=0.30)
    ckpt = pretrain_ssl(splits.train.to_ssl_split(), cfg, epochs=30, seed=42)
    return splits, ckpt
