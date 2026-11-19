"""Tests for tf-cohorts: landmark truncation, t=0 panel firewall, splits, and DGP."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from traj_fuse.data import (
    PBC2_PARQUET_SHA256,
    LandmarkCohortConfig,
    LandmarkedCohortSplits,
    RawLongitudinalCohort,
    SSLPretrainSplit,
    build_landmarked_splits,
    generate_raw_synthetic_subspace_cohort,
    load_pbc2_cohort,
    load_raw_pbc2_cohort,
    make_synthetic_subspace_cohort,
)

FORBIDDEN_SSL_FIELDS = frozenset(
    {
        "residual_duration",
        "followup_time",
        "duration",
        "event",
        "status",
        "ground_truth",
        "u_shared",
        "u_traj",
        "u_mol",
    }
)


def _assert_splits_bit_identical(a: LandmarkedCohortSplits, b: LandmarkedCohortSplits) -> None:
    assert np.array_equal(a.traj_mean, b.traj_mean)
    assert np.array_equal(a.traj_std, b.traj_std)
    assert np.array_equal(a.panel_mean, b.panel_mean)
    assert np.array_equal(a.panel_std, b.panel_std)

    for split_a, split_b in ((a.train, b.train), (a.val, b.val), (a.test, b.test)):
        assert split_a.ssl.patient_ids == split_b.ssl.patient_ids
        assert np.array_equal(split_a.ssl.traj_values, split_b.ssl.traj_values)
        assert np.array_equal(split_a.ssl.traj_obs_mask, split_b.ssl.traj_obs_mask)
        assert np.array_equal(split_a.ssl.visit_times, split_b.ssl.visit_times)
        assert np.array_equal(split_a.ssl.visit_deltas, split_b.ssl.visit_deltas)
        assert np.array_equal(split_a.ssl.landmark_lags, split_b.ssl.landmark_lags)
        assert np.array_equal(split_a.ssl.visit_mask, split_b.ssl.visit_mask)
        assert np.array_equal(split_a.ssl.panel_values, split_b.ssl.panel_values)
        assert np.array_equal(split_a.ssl.panel_obs_mask, split_b.ssl.panel_obs_mask)
        assert np.array_equal(split_a.ssl.panel_available, split_b.ssl.panel_available)
        assert np.array_equal(split_a.residual_duration, split_b.residual_duration)
        assert np.array_equal(split_a.event, split_b.event)


def test_landmark_cohort_config_validation() -> None:
    cfg = LandmarkCohortConfig(landmark_time=2.0)
    assert cfg.landmark_time == 2.0
    assert cfg.train_frac == 0.65
    assert cfg.val_frac == 0.15

    with pytest.raises(ValidationError):
        LandmarkCohortConfig(landmark_time=0.0)

    with pytest.raises(ValidationError):
        LandmarkCohortConfig(landmark_time=2.0, train_frac=0.80, val_frac=0.15)

    with pytest.raises(ValidationError):
        LandmarkCohortConfig.model_validate({"landmark_time": 2.0, "unknown_field": 1})


@pytest.mark.parametrize("cohort_kind", ["synthetic", "pbc2"])
def test_zero_patient_overlap_and_stratification(cohort_kind: str) -> None:
    splits = (
        make_synthetic_subspace_cohort(seed=42)
        if cohort_kind == "synthetic"
        else load_pbc2_cohort(seed=42)
    )
    train_ids = set(splits.train.ssl.patient_ids)
    val_ids = set(splits.val.ssl.patient_ids)
    test_ids = set(splits.test.ssl.patient_ids)

    assert len(train_ids) == len(splits.train.ssl.patient_ids)
    assert len(val_ids) == len(splits.val.ssl.patient_ids)
    assert len(test_ids) == len(splits.test.ssl.patient_ids)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    rates = [
        float(splits.train.event.mean()),
        float(splits.val.event.mean()),
        float(splits.test.event.mean()),
    ]
    assert max(rates) - min(rates) < 0.08


def test_hard_landmark_and_enrollment_panel_firewall_pbc2() -> None:
    raw = load_raw_pbc2_cohort()
    cfg = LandmarkCohortConfig(landmark_time=2.0, seed=42)
    baseline_splits = build_landmarked_splits(raw, cfg)

    df_perturbed = raw.df.copy()
    post_landmark = df_perturbed["visit_time"] > cfg.landmark_time
    assert post_landmark.any()
    for col in raw.traj_cols:
        df_perturbed.loc[post_landmark, col] = (
            df_perturbed.loc[post_landmark, col].fillna(1.0) * 1000.0 + 999.0
        )

    post_enrollment = df_perturbed["visit_time"] > 0.0
    assert post_enrollment.any()
    for col in raw.panel_cols:
        df_perturbed.loc[post_enrollment, col] = 777.0

    perturbed_raw = dataclasses.replace(raw, df=df_perturbed)
    perturbed_splits = build_landmarked_splits(perturbed_raw, cfg)
    _assert_splits_bit_identical(baseline_splits, perturbed_splits)

    for split in (baseline_splits.train, baseline_splits.val, baseline_splits.test):
        assert (split.residual_duration > 0.0).all()
        assert bool(split.ssl.visit_mask[:, 0].all())
        assert (split.ssl.visit_times <= cfg.landmark_time + 1e-6).all()
        assert (split.ssl.visit_deltas >= 0.0).all()
        assert (split.ssl.landmark_lags >= 0.0).all()
        assert np.isfinite(split.ssl.traj_values).all()
        assert np.isfinite(split.ssl.panel_values).all()


def test_hard_landmark_firewall_synthetic_at_boundary() -> None:
    raw = generate_raw_synthetic_subspace_cohort(n_patients=400, seed=7)
    cfg = LandmarkCohortConfig(landmark_time=2.0, seed=7)
    base_splits = build_landmarked_splits(raw, cfg)

    df_perturbed = raw.df.copy()
    boundary_post = df_perturbed["visit_time"] >= cfg.landmark_time + 0.005
    assert boundary_post.any()
    for col in raw.traj_cols:
        df_perturbed.loc[boundary_post, col] = 5000.0
    for col in raw.panel_cols[:10]:
        df_perturbed.loc[df_perturbed["visit_time"] > 0.0, col] = 999.0

    perturbed_splits = build_landmarked_splits(dataclasses.replace(raw, df=df_perturbed), cfg)
    _assert_splits_bit_identical(base_splits, perturbed_splits)


def test_train_only_scaler_invariance() -> None:
    raw = load_raw_pbc2_cohort()
    cfg = LandmarkCohortConfig(landmark_time=2.0, seed=42)
    base_splits = build_landmarked_splits(raw, cfg)

    non_train_ids = set(base_splits.val.ssl.patient_ids) | set(base_splits.test.ssl.patient_ids)
    df_perturbed = raw.df.copy()
    mask = df_perturbed["patient_id"].isin(non_train_ids)
    for col in raw.traj_cols:
        df_perturbed.loc[mask, col] = df_perturbed.loc[mask, col] * 100.0 + 50.0
    for col in raw.panel_cols:
        df_perturbed.loc[mask, col] = df_perturbed.loc[mask, col] * 100.0 + 50.0

    perturbed_splits = build_landmarked_splits(dataclasses.replace(raw, df=df_perturbed), cfg)

    assert np.array_equal(base_splits.traj_mean, perturbed_splits.traj_mean)
    assert np.array_equal(base_splits.traj_std, perturbed_splits.traj_std)
    assert np.array_equal(base_splits.panel_mean, perturbed_splits.panel_mean)
    assert np.array_equal(base_splits.panel_std, perturbed_splits.panel_std)
    assert np.array_equal(base_splits.train.ssl.traj_values, perturbed_splits.train.ssl.traj_values)
    assert np.array_equal(
        base_splits.train.ssl.panel_values, perturbed_splits.train.ssl.panel_values
    )


def test_tier1_post_landmark_subspace_orthogonality_and_shapes() -> None:
    t0 = time.perf_counter()
    splits = make_synthetic_subspace_cohort(
        n_patients=1_400,
        d_traj=8,
        d_mol=256,
        landmark_time=2.0,
        panel_missing_rate=0.25,
        seed=42,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 300.0

    assert splits.train.ssl.traj_values.shape[0] >= 750
    assert splits.train.ssl.traj_values.shape[2] == 8
    assert splits.train.ssl.panel_values.shape[1] == 256

    all_splits = (splits.train, splits.val, splits.test)
    u_s = np.vstack([s.ground_truth.u_shared for s in all_splits if s.ground_truth is not None])
    u_t = np.vstack([s.ground_truth.u_traj for s in all_splits if s.ground_truth is not None])
    u_m = np.vstack([s.ground_truth.u_mol for s in all_splits if s.ground_truth is not None])

    combined = np.hstack([u_s, u_t, u_m])
    corr = np.corrcoef(combined, rowvar=False)
    k_s, k_t, k_m = u_s.shape[1], u_t.shape[1], u_m.shape[1]

    cross_st = np.abs(corr[:k_s, k_s : k_s + k_t]).max()
    cross_sm = np.abs(corr[:k_s, k_s + k_t : k_s + k_t + k_m]).max()
    cross_tm = np.abs(corr[k_s : k_s + k_t, k_s + k_t :]).max()
    assert max(float(cross_st), float(cross_sm), float(cross_tm)) < 0.08

    avail_rate = float(np.concatenate([s.ssl.panel_available for s in all_splits]).mean())
    assert 0.68 <= avail_rate <= 0.82


@pytest.mark.parametrize("cohort_kind", ["synthetic", "pbc2"])
def test_ssl_pretrain_split_is_outcome_blind(cohort_kind: str) -> None:
    splits = (
        make_synthetic_subspace_cohort(n_patients=400, seed=42)
        if cohort_kind == "synthetic"
        else load_pbc2_cohort(seed=42)
    )
    ssl_view = splits.train.to_ssl_split()
    assert isinstance(ssl_view, SSLPretrainSplit)

    field_names = {f.name for f in dataclasses.fields(ssl_view)}
    assert field_names.isdisjoint(FORBIDDEN_SSL_FIELDS)
    for forbidden in FORBIDDEN_SSL_FIELDS:
        assert not hasattr(ssl_view, forbidden)


def test_pbc2_cohort_contracts_and_validation(tmp_path: Path) -> None:
    t0 = time.perf_counter()
    splits = load_pbc2_cohort(landmark_time=2.0, seed=42)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 300.0

    total_n = (
        len(splits.train.ssl.patient_ids)
        + len(splits.val.ssl.patient_ids)
        + len(splits.test.ssl.patient_ids)
    )
    assert total_n == 278
    assert splits.train.ssl.traj_values.shape[2] == 8
    assert splits.train.ssl.panel_values.shape[1] == 6
    assert splits.train.ground_truth is None
    assert splits.traj_feature_names[-2:] == ("age", "sex_female")
    assert splits.panel_feature_names[1] == "baseline_serChol"
    assert (splits.train.ssl.panel_obs_mask[:, 1] == 0.0).any()

    with pytest.raises(FileNotFoundError):
        load_pbc2_cohort(parquet_path=tmp_path / "missing.parquet")

    src_parquet = Path(__file__).resolve().parent.parent / "data" / "pbc2_longitudinal.parquet"
    tampered = tmp_path / "tampered.parquet"
    tampered.write_bytes(src_parquet.read_bytes())
    with pytest.raises(ValueError, match="SHA-256"):
        load_pbc2_cohort(parquet_path=tampered, expected_sha256="0" * 64)

    bad_df = pd.read_parquet(src_parquet)
    bad_df = bad_df[bad_df["visit_time"] > 0.0].copy()
    bad_raw = RawLongitudinalCohort(
        cohort_name="bad_pbc2",
        sha256=PBC2_PARQUET_SHA256,
        df=bad_df,
        traj_cols=load_raw_pbc2_cohort().traj_cols,
        panel_cols=load_raw_pbc2_cohort().panel_cols,
    )
    with pytest.raises(ValueError, match="enrollment"):
        build_landmarked_splits(bad_raw, LandmarkCohortConfig(landmark_time=2.0))
