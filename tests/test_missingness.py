"""Missing-modality robustness, drop-consistency alignment, and frozen checkpoint tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from traj_fuse.data import make_synthetic_subspace_cohort
from traj_fuse.model import SharedPrivateConfig
from traj_fuse.train import FrozenSSLCheckpoint, pretrain_ssl


def test_frozen_checkpoint_requires_grad_false_and_weights_only_roundtrip(tmp_path: Path) -> None:
    splits = make_synthetic_subspace_cohort(n_patients=160, d_traj=8, d_mol=24, seed=7)
    cfg = SharedPrivateConfig.for_cohort(splits, p_drop=0.30)
    ckpt = pretrain_ssl(splits.train.to_ssl_split(), cfg, epochs=8, batch_size=64, seed=7)

    save_dir = ckpt.save(tmp_path / "ckpt")
    assert (save_dir / "config.json").exists()
    assert (save_dir / "weights.pt").exists()

    loaded = FrozenSSLCheckpoint.load(save_dir)
    assert not loaded.model.training
    for param in loaded.model.parameters():
        assert param.requires_grad is False

    emb_a = ckpt.encode(splits.test.to_ssl_split())
    emb_b = loaded.encode(splits.test.to_ssl_split())
    assert np.allclose(emb_a.z_fused, emb_b.z_fused, atol=1e-6)
    assert ckpt.parameter_sha256() == loaded.parameter_sha256()


def test_pretrain_ssl_rejects_labeled_landmarked_split() -> None:
    splits = make_synthetic_subspace_cohort(n_patients=120, d_traj=8, d_mol=16, seed=3)
    cfg = SharedPrivateConfig.for_cohort(splits, p_drop=0.30)
    with pytest.raises(TypeError, match="SSLPretrainSplit"):
        pretrain_ssl(cast(Any, splits.train), cfg, epochs=2)


def test_missing_panel_inference_and_pdrop_consistency_gain() -> None:
    splits = make_synthetic_subspace_cohort(
        n_patients=360, d_traj=8, d_mol=32, panel_missing_rate=0.0, seed=11
    )
    test_ssl = splits.test.to_ssl_split()
    n_test = len(test_ssl.patient_ids)

    cfg_drop = SharedPrivateConfig.for_cohort(splits, p_drop=0.30)
    cfg_nodrop = SharedPrivateConfig.for_cohort(splits, p_drop=0.0)

    ckpt_drop = pretrain_ssl(
        splits.train.to_ssl_split(), cfg_drop, epochs=25, batch_size=64, seed=11
    )
    ckpt_nodrop = pretrain_ssl(
        splits.train.to_ssl_split(), cfg_nodrop, epochs=25, batch_size=64, seed=11
    )

    drop_50 = np.zeros(n_test, dtype=np.bool_)
    drop_50[::2] = True
    for drop_arg in (False, drop_50, True):
        emb = ckpt_drop.encode(test_ssl, force_drop_panel=drop_arg)
        assert np.isfinite(emb.z_fused).all()
        assert np.isfinite(emb.z_shared).all()

    full_drop = ckpt_drop.encode(test_ssl, force_drop_panel=False).z_fused
    miss_drop = ckpt_drop.encode(test_ssl, force_drop_panel=True).z_fused
    mse_with_pdrop = float(np.mean((miss_drop - full_drop) ** 2))

    full_nodrop = ckpt_nodrop.encode(test_ssl, force_drop_panel=False).z_fused
    miss_nodrop = ckpt_nodrop.encode(test_ssl, force_drop_panel=True).z_fused
    mse_without_pdrop = float(np.mean((miss_nodrop - full_nodrop) ** 2))

    assert mse_with_pdrop < mse_without_pdrop
    assert torch.isfinite(torch.tensor(mse_with_pdrop))
