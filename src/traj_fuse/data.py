"""Leak-free landmarked longitudinal cohorts, Tier 1 DGP, and Mayo PBC2 loader."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

PBC2_PARQUET_SHA256 = "27f132e9f3b18a5889c2af2b35cdd42602ecc70bc9f0ca575d7c2475879fc8b3"
DEFAULT_PBC2_PARQUET = Path(__file__).resolve().parents[2] / "data" / "pbc2_longitudinal.parquet"

PBC2_TRAJ_COLS: tuple[str, ...] = (
    "serBilir",
    "albumin",
    "alkaline",
    "SGOT",
    "platelets",
    "prothrombin",
    "age",
    "sex_female",
)
PBC2_LOG_TRAJ_COLS: tuple[str, ...] = ("serBilir", "alkaline", "SGOT", "prothrombin")
PBC2_RAW_PANEL_COLS: tuple[str, ...] = (
    "histologic",
    "serChol",
    "ascites",
    "hepatomegaly",
    "spiders",
    "edema_score",
)
PBC2_PANEL_FEATURE_NAMES: tuple[str, ...] = (
    "histologic",
    "baseline_serChol",
    "ascites",
    "hepatomegaly",
    "spiders",
    "edema_score",
)
PBC2_LOG_PANEL_COLS: tuple[str, ...] = ("serChol",)
OracleTuple = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]


class LandmarkCohortConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    landmark_time: float = Field(gt=0.0, description="Hard landmark T_L in years.")
    train_frac: float = Field(default=0.65, gt=0.0, lt=1.0)
    val_frac: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: int = Field(default=42)

    @model_validator(mode="after")
    def _validate_split_fractions(self) -> LandmarkCohortConfig:
        if self.train_frac + self.val_frac >= 0.95:
            raise ValueError(
                f"train_frac + val_frac ({self.train_frac + self.val_frac:.2f}) must be < 0.95."
            )
        return self


@dataclass(frozen=True)
class GroundTruthSubspaces:
    """Tier-1 ONLY evaluation oracle; omitted from SSLPretrainSplit."""

    u_shared: NDArray[np.float32]
    u_traj: NDArray[np.float32]
    u_mol: NDArray[np.float32]


@dataclass(frozen=True)
class SSLPretrainSplit:
    """Outcome-blind tensor view passed to tf-shared-private (zero survival/oracle fields)."""

    patient_ids: tuple[str, ...]
    landmark_time: float
    traj_values: NDArray[np.float32]
    traj_obs_mask: NDArray[np.float32]
    visit_times: NDArray[np.float32]
    visit_deltas: NDArray[np.float32]
    landmark_lags: NDArray[np.float32]
    visit_mask: NDArray[np.bool_]
    panel_values: NDArray[np.float32]
    panel_obs_mask: NDArray[np.float32]
    panel_available: NDArray[np.bool_]


@dataclass(frozen=True)
class LandmarkedSplit:
    ssl: SSLPretrainSplit
    residual_duration: NDArray[np.float64]
    event: NDArray[np.int64]
    ground_truth: GroundTruthSubspaces | None = None

    def to_ssl_split(self) -> SSLPretrainSplit:
        return self.ssl


@dataclass(frozen=True)
class RawLongitudinalCohort:
    cohort_name: str
    sha256: str
    df: pd.DataFrame
    traj_cols: tuple[str, ...]
    panel_cols: tuple[str, ...]
    log_transform_traj_cols: tuple[str, ...] = ()
    log_transform_panel_cols: tuple[str, ...] = ()
    panel_feature_names: tuple[str, ...] | None = None
    ground_truth_by_id: dict[str, OracleTuple] | None = None


@dataclass(frozen=True)
class LandmarkedCohortSplits:
    cohort_name: str
    cohort_sha256: str
    config: LandmarkCohortConfig
    traj_feature_names: tuple[str, ...]
    panel_feature_names: tuple[str, ...]
    train: LandmarkedSplit
    val: LandmarkedSplit
    test: LandmarkedSplit
    traj_mean: NDArray[np.float32]
    traj_std: NDArray[np.float32]
    panel_mean: NDArray[np.float32]
    panel_std: NDArray[np.float32]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_dataframe(name: str, df: pd.DataFrame) -> str:
    digest = hashlib.sha256(name.encode("utf-8"))
    for col in df.columns:
        digest.update(str(col).encode("utf-8"))
        digest.update(df[col].to_numpy().tobytes())
    return digest.hexdigest()


def _stratified_split_indices(
    events: NDArray[np.int64], train_frac: float, val_frac: float, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for label in (0, 1):
        cls_idx = np.flatnonzero(events == label).astype(np.int64)
        if cls_idx.size == 0:
            continue
        perm = cls_idx[rng.permutation(cls_idx.size)]
        n_cls = int(perm.size)
        n_tr = max(1, int(round(n_cls * train_frac)))
        n_va = max(1, int(round(n_cls * val_frac))) if n_cls >= 3 else 0
        if n_tr + n_va >= n_cls:
            n_va = max(0, n_cls - n_tr - 1)
        tr.append(perm[:n_tr])
        va.append(perm[n_tr : n_tr + n_va])
        te.append(perm[n_tr + n_va :])
    return (
        np.sort(np.concatenate(tr)).astype(np.int64),
        np.sort(np.concatenate(va)).astype(np.int64),
        np.sort(np.concatenate(te)).astype(np.int64),
    )


def _fit_observed_stats(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    n_cols = values.shape[1]
    means, stds = np.zeros(n_cols, dtype=np.float32), np.ones(n_cols, dtype=np.float32)
    for col in range(n_cols):
        valid = values[~np.isnan(values[:, col]), col]
        if valid.size > 0:
            means[col] = np.float32(valid.mean())
            stds[col] = np.float32(max(float(valid.std(ddof=0)), 1e-6))
    return means, stds


def _apply_log1p(
    df: pd.DataFrame, cols: tuple[str, ...], log_cols: tuple[str, ...]
) -> NDArray[np.float64]:
    sub = df[list(cols)].astype("float64").copy()
    for col in log_cols:
        if col in sub.columns:
            sub[col] = np.log1p(np.maximum(sub[col].to_numpy(dtype=np.float64), 0.0))
    return sub.to_numpy(dtype=np.float64)


def build_landmarked_splits(
    raw: RawLongitudinalCohort, config: LandmarkCohortConfig
) -> LandmarkedCohortSplits:
    """Apply hard landmarking (T_i > T_L, t_ij <= T_L), t=0 panel extraction, and splits."""
    df = raw.df.sort_values(["patient_id", "visit_time"], kind="mergesort").reset_index(drop=True)
    if (df["visit_time"] < 0.0).any():
        raise ValueError("All visit_time values must be >= 0.0.")
    if df.duplicated(subset=["patient_id", "visit_time"]).any():
        raise ValueError("Duplicate (patient_id, visit_time) visits detected.")
    eligible_df = df[df["followup_time"] > config.landmark_time]
    if eligible_df.empty:
        raise ValueError(f"No patients survive past landmark_time={config.landmark_time}.")
    pre_df = eligible_df[eligible_df["visit_time"] <= config.landmark_time]
    if set(eligible_df["patient_id"].unique()) != set(pre_df["patient_id"].unique()):
        raise ValueError("Every eligible patient must have >= 1 visit at or before landmark_time.")

    enroll_df = pre_df.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)
    if (enroll_df["visit_time"] > 0.05).any():
        raise ValueError("Every patient must have an enrollment visit at t <= 0.05 years.")
    patient_ids = tuple(str(pid) for pid in enroll_df["patient_id"].tolist())
    pid_to_pos = {pid: idx for idx, pid in enumerate(patient_ids)}
    followup_time = enroll_df["followup_time"].to_numpy(dtype=np.float64)
    event = enroll_df["event"].to_numpy(dtype=np.int64)
    if not np.isin(event, (0, 1)).all():
        raise ValueError("Event values must be strictly in {0, 1}.")

    idx_splits = _stratified_split_indices(event, config.train_frac, config.val_frac, config.seed)
    split_pids = tuple(tuple(patient_ids[i] for i in idx) for idx in idx_splits)
    traj_mat = _apply_log1p(pre_df, raw.traj_cols, raw.log_transform_traj_cols)
    panel_mat = _apply_log1p(enroll_df, raw.panel_cols, raw.log_transform_panel_cols)
    tr_visit = pre_df["patient_id"].isin(set(split_pids[0])).to_numpy(dtype=np.bool_)
    traj_mean, traj_std = _fit_observed_stats(traj_mat[tr_visit])
    panel_mean, panel_std = _fit_observed_stats(panel_mat[idx_splits[0]])

    row_pids, row_times = pre_df["patient_id"].to_numpy(), pre_df["visit_time"].to_numpy(np.float64)
    d_traj, t_land = traj_mat.shape[1], float(config.landmark_time)
    built: list[LandmarkedSplit] = []
    for sp_pids in split_pids:
        n_sp, lmap = len(sp_pids), {pid: i for i, pid in enumerate(sp_pids)}
        keep = np.fromiter((p in lmap for p in row_pids), dtype=np.bool_, count=len(row_pids))
        p_idx = np.fromiter(
            (lmap[str(p)] for p in row_pids[keep]), dtype=np.int64, count=int(keep.sum())
        )
        counts = np.bincount(p_idx, minlength=n_sp)
        l_max = int(counts.max())
        offsets = np.zeros(n_sp, dtype=np.int64)
        if n_sp > 1:
            offsets[1:] = np.cumsum(counts[:-1])
        order = np.argsort(p_idx, kind="mergesort")
        p_idx = p_idx[order]
        s_times, s_traj = row_times[keep][order], traj_mat[keep][order]
        step = np.arange(len(p_idx), dtype=np.int64) - offsets[p_idx]

        t_vals = np.zeros((n_sp, l_max, d_traj), dtype=np.float32)
        t_mask = np.zeros((n_sp, l_max, d_traj), dtype=np.float32)
        v_times = np.zeros((n_sp, l_max), dtype=np.float32)
        v_deltas = np.zeros((n_sp, l_max), dtype=np.float32)
        v_lags = np.zeros((n_sp, l_max), dtype=np.float32)
        v_mask = np.zeros((n_sp, l_max), dtype=np.bool_)
        obs_t = ~np.isnan(s_traj)
        t_vals[p_idx, step, :] = np.where(obs_t, (s_traj - traj_mean) / traj_std, 0.0).astype(
            np.float32
        )
        t_mask[p_idx, step, :] = obs_t.astype(np.float32)
        v_times[p_idx, step], v_lags[p_idx, step] = (
            s_times.astype(np.float32),
            (t_land - s_times).astype(np.float32),
        )
        v_mask[p_idx, step] = True
        if l_max > 1:
            v_deltas[:, 1:] = np.where(
                v_mask[:, 1:], np.maximum(v_times[:, 1:] - v_times[:, :-1], 0.0), 0.0
            )

        g_idx = np.fromiter((pid_to_pos[p] for p in sp_pids), dtype=np.int64, count=n_sp)
        raw_p = panel_mat[g_idx]
        obs_p = ~np.isnan(raw_p)
        p_vals = np.where(obs_p, (raw_p - panel_mean) / panel_std, 0.0).astype(np.float32)
        gt: GroundTruthSubspaces | None = None
        if raw.ground_truth_by_id is not None:
            gt = GroundTruthSubspaces(
                u_shared=np.vstack([raw.ground_truth_by_id[p][0] for p in sp_pids]).astype(
                    np.float32
                ),
                u_traj=np.vstack([raw.ground_truth_by_id[p][1] for p in sp_pids]).astype(
                    np.float32
                ),
                u_mol=np.vstack([raw.ground_truth_by_id[p][2] for p in sp_pids]).astype(np.float32),
            )
        built.append(
            LandmarkedSplit(
                ssl=SSLPretrainSplit(
                    sp_pids,
                    t_land,
                    t_vals,
                    t_mask,
                    v_times,
                    v_deltas,
                    v_lags,
                    v_mask,
                    p_vals,
                    obs_p.astype(np.float32),
                    obs_p.any(axis=1),
                ),
                residual_duration=(followup_time[g_idx] - t_land).astype(np.float64),
                event=event[g_idx].astype(np.int64),
                ground_truth=gt,
            )
        )

    pnames = raw.panel_feature_names if raw.panel_feature_names is not None else raw.panel_cols
    return LandmarkedCohortSplits(
        raw.cohort_name,
        raw.sha256,
        config,
        raw.traj_cols,
        pnames,
        built[0],
        built[1],
        built[2],
        traj_mean,
        traj_std,
        panel_mean,
        panel_std,
    )


def generate_raw_synthetic_subspace_cohort(
    n_patients: int = 1_400,
    d_traj: int = 8,
    d_mol: int = 256,
    landmark_time: float = 2.0,
    panel_missing_rate: float = 0.25,
    seed: int = 42,
) -> RawLongitudinalCohort:
    """Generate Tier 1 collider-free orthogonal subspace longitudinal survival cohort."""
    if n_patients < 100 or d_traj < 4 or d_mol < 8:
        raise ValueError("Require n_patients >= 100, d_traj >= 4, and d_mol >= 8.")

    rng = np.random.default_rng(seed)
    k_sub = 4
    u_s, u_t, u_m = (rng.normal(0.0, 1.0, (n_patients, k_sub)) for _ in range(3))
    half_t, half_m = d_traj // 2, d_mol // 2
    w_ts, w_tt = np.zeros((k_sub, d_traj)), np.zeros((k_sub, d_traj))
    w_ts[:, :half_t] = 1.45 * np.linalg.qr(rng.normal(0.0, 1.0, (half_t, k_sub)))[0].T
    w_tt[:, half_t:] = 1.45 * np.linalg.qr(rng.normal(0.0, 1.0, (d_traj - half_t, k_sub)))[0].T
    w_ms, w_mm = np.zeros((k_sub, d_mol)), np.zeros((k_sub, d_mol))
    w_ms[:, :half_m] = (rng.random((k_sub, half_m)) < 0.30) * rng.normal(0.0, 1.15, (k_sub, half_m))
    w_mm[:, half_m:] = (rng.random((k_sub, d_mol - half_m)) < 0.30) * rng.normal(
        0.0, 1.15, (k_sub, d_mol - half_m)
    )

    eta = (
        u_s @ np.array([0.65, -0.55, 0.50, -0.45])
        + u_t @ np.array([0.40, -0.35, 0.30, -0.25])
        + u_m @ np.array([0.40, -0.35, 0.30, -0.25])
    )
    pre_drop = rng.random(n_patients) < 0.12
    weibull_res = (
        -np.log(np.clip(rng.random(n_patients), 1e-6, 1.0 - 1e-6)) / (0.065 * np.exp(eta))
    ) ** (1.0 / 1.5)
    censor_res = rng.uniform(2.5, 9.5, n_patients)
    followup_time = np.where(
        pre_drop,
        rng.uniform(0.1, landmark_time * 0.95, n_patients),
        landmark_time + np.maximum(np.minimum(weibull_res, censor_res), 0.05),
    )
    event = np.where(
        pre_drop, rng.integers(0, 2, n_patients), (weibull_res <= censor_res).astype(np.int64)
    )

    panel_base = u_s @ w_ms + u_m @ w_mm + rng.normal(0.0, 0.35, (n_patients, d_mol))
    panel_base[rng.random(n_patients) < panel_missing_rate, :] = np.nan
    panel_base[rng.random((n_patients, d_mol)) < 0.03] = np.nan

    n_pre = np.clip(1 + rng.poisson(np.exp(0.75 + 0.35 * u_s[:, 0])), 1, 11)
    pids = [f"syn_{idx:04d}" for idx in range(n_patients)]
    base_s, base_t = u_s @ w_ts, u_t @ w_tt
    rows_pid, rows_fu, rows_ev, rows_vt, enroll_rows, traj_blocks, cursor = (
        [],
        [],
        [],
        [],
        [],
        [],
        0,
    )
    for i in range(n_patients):
        t_i = np.concatenate(
            [
                [0.0],
                np.sort(rng.uniform(0.12, landmark_time, int(n_pre[i]))),
                [landmark_time + 0.35, landmark_time + 1.10],
            ]
        )
        n_v = len(t_i)
        tau = t_i[:, None] / landmark_time
        obs = (
            (0.85 + 0.20 * tau) * base_s[i : i + 1]
            + (0.35 + 1.15 * tau) * base_t[i : i + 1]
            + (t_i > landmark_time)[:, None] * (1.5 * eta[i])
            + rng.normal(0.0, 0.25, (n_v, d_traj))
        )
        mcar = rng.random((n_v, d_traj)) < 0.05
        mcar[0, :] = False
        obs[mcar] = np.nan
        enroll_rows.append(cursor)
        cursor += n_v
        rows_pid.extend([pids[i]] * n_v)
        rows_fu.extend([float(followup_time[i])] * n_v)
        rows_ev.extend([int(event[i])] * n_v)
        rows_vt.extend(t_i.tolist())
        traj_blocks.append(obs)

    traj_cols = tuple(f"traj_{j}" for j in range(d_traj))
    panel_cols = tuple(f"mol_{j}" for j in range(d_mol))
    panel_all = np.full((cursor, d_mol), np.nan, dtype=np.float64)
    panel_all[np.asarray(enroll_rows, dtype=np.int64)] = panel_base
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "patient_id": rows_pid,
                    "followup_time": np.asarray(rows_fu, dtype=np.float64),
                    "event": np.asarray(rows_ev, dtype=np.int64),
                    "visit_time": np.asarray(rows_vt, dtype=np.float64),
                }
            ),
            pd.DataFrame(np.vstack(traj_blocks), columns=list(traj_cols)),
            pd.DataFrame(panel_all, columns=list(panel_cols)),
        ],
        axis=1,
    )
    u_s32, u_t32, u_m32 = u_s.astype(np.float32), u_t.astype(np.float32), u_m.astype(np.float32)
    gt_by_id: dict[str, OracleTuple] = {
        pids[i]: (u_s32[i], u_t32[i], u_m32[i]) for i in range(n_patients)
    }
    name = f"synthetic_subspace_n{n_patients}_dt{d_traj}_dm{d_mol}_seed{seed}"
    return RawLongitudinalCohort(
        name, _hash_dataframe(name, df), df, traj_cols, panel_cols, ground_truth_by_id=gt_by_id
    )


def make_synthetic_subspace_cohort(
    n_patients: int = 1_400,
    d_traj: int = 8,
    d_mol: int = 256,
    landmark_time: float = 2.0,
    panel_missing_rate: float = 0.25,
    seed: int = 42,
    train_frac: float = 0.65,
    val_frac: float = 0.15,
) -> LandmarkedCohortSplits:
    """Build landmarked splits for the Tier 1 collider-free subspace survival cohort."""
    raw = generate_raw_synthetic_subspace_cohort(
        n_patients, d_traj, d_mol, landmark_time, panel_missing_rate, seed
    )
    return build_landmarked_splits(
        raw,
        LandmarkCohortConfig(
            landmark_time=landmark_time, train_frac=train_frac, val_frac=val_frac, seed=seed
        ),
    )


def load_raw_pbc2_cohort(
    parquet_path: Path | None = None, *, expected_sha256: str | None = PBC2_PARQUET_SHA256
) -> RawLongitudinalCohort:
    """Load and verify the bundled Mayo Clinic PBC2 longitudinal parquet cohort."""
    path = parquet_path if parquet_path is not None else DEFAULT_PBC2_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"PBC2 cohort parquet not found at {path}")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"PBC2 parquet SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    df = pq.read_table(path).to_pandas()
    missing = {
        "patient_id",
        "followup_time",
        "event",
        "visit_time",
        *PBC2_TRAJ_COLS,
        *PBC2_RAW_PANEL_COLS,
    } - set(df.columns)
    if missing:
        raise ValueError(f"PBC2 parquet missing required columns: {sorted(missing)}")
    return RawLongitudinalCohort(
        "pbc2_longitudinal",
        actual_sha256,
        df,
        PBC2_TRAJ_COLS,
        PBC2_RAW_PANEL_COLS,
        PBC2_LOG_TRAJ_COLS,
        PBC2_LOG_PANEL_COLS,
        PBC2_PANEL_FEATURE_NAMES,
        None,
    )


def load_pbc2_cohort(
    landmark_time: float = 2.0,
    seed: int = 42,
    *,
    train_frac: float = 0.65,
    val_frac: float = 0.15,
    parquet_path: Path | None = None,
    expected_sha256: str | None = PBC2_PARQUET_SHA256,
) -> LandmarkedCohortSplits:
    """Load, verify, landmark, and split the Mayo Clinic PBC2 longitudinal cohort."""
    raw = load_raw_pbc2_cohort(parquet_path=parquet_path, expected_sha256=expected_sha256)
    cfg = LandmarkCohortConfig(
        landmark_time=landmark_time, train_frac=train_frac, val_frac=val_frac, seed=seed
    )
    return build_landmarked_splits(raw, cfg)
