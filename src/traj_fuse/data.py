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
                f"train_frac + val_frac ({self.train_frac + self.val_frac:.2f}) must be < 0.95 "
                "to leave a non-trivial held-out test split."
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
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
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
    means = np.zeros(n_cols, dtype=np.float32)
    stds = np.ones(n_cols, dtype=np.float32)
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

    eligible_df = df[df["followup_time"] > config.landmark_time].copy()
    if eligible_df.empty:
        raise ValueError(f"No patients survive past landmark_time={config.landmark_time}.")

    pre_df = eligible_df[eligible_df["visit_time"] <= config.landmark_time].copy()
    if set(eligible_df["patient_id"].unique()) != set(pre_df["patient_id"].unique()):
        raise ValueError("Every eligible patient must have >= 1 visit at or before landmark_time.")

    # Strict enrollment-only (t=0, skipna=False) baseline panel extraction
    enroll_df = pre_df.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)
    if (enroll_df["visit_time"] > 0.05).any():
        raise ValueError("Every patient must have an enrollment visit at t <= 0.05 years.")

    patient_ids = tuple(str(pid) for pid in enroll_df["patient_id"].tolist())
    pid_to_pos = {pid: idx for idx, pid in enumerate(patient_ids)}
    followup_time = enroll_df["followup_time"].to_numpy(dtype=np.float64)
    event = enroll_df["event"].to_numpy(dtype=np.int64)
    if not np.isin(event, (0, 1)).all():
        raise ValueError("Event values must be strictly in {0, 1}.")

    tr_idx, va_idx, te_idx = _stratified_split_indices(
        event, config.train_frac, config.val_frac, config.seed
    )
    split_pids_tuple = tuple(tuple(patient_ids[i] for i in idx) for idx in (tr_idx, va_idx, te_idx))

    traj_mat = _apply_log1p(pre_df, raw.traj_cols, raw.log_transform_traj_cols)
    enroll_panel_mat = _apply_log1p(enroll_df, raw.panel_cols, raw.log_transform_panel_cols)
    train_visit_mask = pre_df["patient_id"].isin(set(split_pids_tuple[0])).to_numpy(dtype=np.bool_)
    traj_mean, traj_std = _fit_observed_stats(traj_mat[train_visit_mask])
    panel_mean, panel_std = _fit_observed_stats(enroll_panel_mat[tr_idx])

    row_pids = pre_df["patient_id"].to_numpy()
    row_times = pre_df["visit_time"].to_numpy(dtype=np.float64)
    d_traj = traj_mat.shape[1]
    t_land = float(config.landmark_time)

    built: list[LandmarkedSplit] = []
    for sp_pids in split_pids_tuple:
        n_sp = len(sp_pids)
        local_map = {pid: i for i, pid in enumerate(sp_pids)}
        keep = np.fromiter(
            (pid in local_map for pid in row_pids), dtype=np.bool_, count=len(row_pids)
        )
        p_idx = np.fromiter(
            (local_map[str(pid)] for pid in row_pids[keep]), dtype=np.int64, count=int(keep.sum())
        )
        counts = np.bincount(p_idx, minlength=n_sp)
        l_max = int(counts.max())
        offsets = np.zeros(n_sp, dtype=np.int64)
        if n_sp > 1:
            offsets[1:] = np.cumsum(counts[:-1])
        order = np.argsort(p_idx, kind="mergesort")
        p_idx = p_idx[order]
        s_times = row_times[keep][order]
        s_traj = traj_mat[keep][order]
        step_idx = np.arange(len(p_idx), dtype=np.int64) - offsets[p_idx]

        t_vals = np.zeros((n_sp, l_max, d_traj), dtype=np.float32)
        t_mask = np.zeros((n_sp, l_max, d_traj), dtype=np.float32)
        v_times = np.zeros((n_sp, l_max), dtype=np.float32)
        v_deltas = np.zeros((n_sp, l_max), dtype=np.float32)
        v_lags = np.zeros((n_sp, l_max), dtype=np.float32)
        v_mask = np.zeros((n_sp, l_max), dtype=np.bool_)

        obs_t = ~np.isnan(s_traj)
        t_vals[p_idx, step_idx, :] = np.where(
            obs_t, (s_traj - traj_mean[None, :]) / traj_std[None, :], 0.0
        ).astype(np.float32)
        t_mask[p_idx, step_idx, :] = obs_t.astype(np.float32)
        v_times[p_idx, step_idx] = s_times.astype(np.float32)
        v_lags[p_idx, step_idx] = (t_land - s_times).astype(np.float32)
        v_mask[p_idx, step_idx] = True
        if l_max > 1:
            v_deltas[:, 1:] = np.where(
                v_mask[:, 1:], np.maximum(v_times[:, 1:] - v_times[:, :-1], 0.0), 0.0
            )

        g_idx = np.fromiter((pid_to_pos[p] for p in sp_pids), dtype=np.int64, count=n_sp)
        raw_p = enroll_panel_mat[g_idx]
        obs_p = ~np.isnan(raw_p)
        p_vals = np.where(obs_p, (raw_p - panel_mean[None, :]) / panel_std[None, :], 0.0).astype(
            np.float32
        )

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
                    patient_ids=sp_pids,
                    landmark_time=t_land,
                    traj_values=t_vals,
                    traj_obs_mask=t_mask,
                    visit_times=v_times,
                    visit_deltas=v_deltas,
                    landmark_lags=v_lags,
                    visit_mask=v_mask,
                    panel_values=p_vals,
                    panel_obs_mask=obs_p.astype(np.float32),
                    panel_available=obs_p.any(axis=1),
                ),
                residual_duration=(followup_time[g_idx] - t_land).astype(np.float64),
                event=event[g_idx].astype(np.int64),
                ground_truth=gt,
            )
        )

    panel_names = raw.panel_feature_names if raw.panel_feature_names is not None else raw.panel_cols
    return LandmarkedCohortSplits(
        cohort_name=raw.cohort_name,
        cohort_sha256=raw.sha256,
        config=config,
        traj_feature_names=raw.traj_cols,
        panel_feature_names=panel_names,
        train=built[0],
        val=built[1],
        test=built[2],
        traj_mean=traj_mean,
        traj_std=traj_std,
        panel_mean=panel_mean,
        panel_std=panel_std,
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
    u_shared = rng.normal(0.0, 1.0, size=(n_patients, k_sub))
    u_traj = rng.normal(0.0, 1.0, size=(n_patients, k_sub))
    u_mol = rng.normal(0.0, 1.0, size=(n_patients, k_sub))

    half_t, half_m = d_traj // 2, d_mol // 2
    w_traj_s = np.zeros((k_sub, d_traj), dtype=np.float64)
    w_traj_t = np.zeros((k_sub, d_traj), dtype=np.float64)
    q_s, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(half_t, k_sub)))
    q_t, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(d_traj - half_t, k_sub)))
    w_traj_s[:, :half_t], w_traj_t[:, half_t:] = 1.45 * q_s.T, 1.45 * q_t.T

    w_mol_s = np.zeros((k_sub, d_mol), dtype=np.float64)
    w_mol_m = np.zeros((k_sub, d_mol), dtype=np.float64)
    w_mol_s[:, :half_m] = (rng.random((k_sub, half_m)) < 0.30) * rng.normal(
        0.0, 1.15, size=(k_sub, half_m)
    )
    w_mol_m[:, half_m:] = (rng.random((k_sub, d_mol - half_m)) < 0.30) * rng.normal(
        0.0, 1.15, size=(k_sub, d_mol - half_m)
    )

    eta = (
        u_shared @ np.array([0.65, -0.55, 0.50, -0.45])
        + u_traj @ np.array([0.40, -0.35, 0.30, -0.25])
        + u_mol @ np.array([0.40, -0.35, 0.30, -0.25])
    )
    pre_drop = rng.random(n_patients) < 0.12
    u_unif = np.clip(rng.random(n_patients), 1e-6, 1.0 - 1e-6)
    weibull_res = (-np.log(u_unif) / (0.065 * np.exp(eta))) ** (1.0 / 1.5)
    censor_res = rng.uniform(2.5, 9.5, size=n_patients)
    followup_time = np.where(
        pre_drop,
        rng.uniform(0.1, landmark_time * 0.95, size=n_patients),
        landmark_time + np.maximum(np.minimum(weibull_res, censor_res), 0.05),
    )
    event = np.where(
        pre_drop, rng.integers(0, 2, size=n_patients), (weibull_res <= censor_res).astype(np.int64)
    )

    panel_base = (
        u_shared @ w_mol_s + u_mol @ w_mol_m + rng.normal(0.0, 0.35, size=(n_patients, d_mol))
    )
    panel_base[rng.random(n_patients) < panel_missing_rate, :] = np.nan
    panel_base[rng.random((n_patients, d_mol)) < 0.03] = np.nan

    n_pre_extra = np.clip(1 + rng.poisson(np.exp(0.75 + 0.35 * u_shared[:, 0])), 1, 11)
    patient_ids = [f"syn_{idx:04d}" for idx in range(n_patients)]
    base_shared, base_priv = u_shared @ w_traj_s, u_traj @ w_traj_t

    rows_pid: list[str] = []
    rows_futime: list[float] = []
    rows_event: list[int] = []
    rows_vtime: list[float] = []
    enroll_rows: list[int] = []
    traj_blocks: list[NDArray[np.float64]] = []
    cursor = 0

    for i in range(n_patients):
        k_pre = int(n_pre_extra[i])
        times_i = np.concatenate(
            [
                [0.0],
                np.sort(rng.uniform(0.12, landmark_time, size=k_pre)),
                [landmark_time + 0.35, landmark_time + 1.10],
            ]
        )
        n_vis = len(times_i)
        tau_i = times_i[:, None] / landmark_time
        mean_traj = (
            (0.85 + 0.20 * tau_i) * base_shared[i : i + 1, :]
            + (0.35 + 1.15 * tau_i) * base_priv[i : i + 1, :]
            + (times_i > landmark_time)[:, None] * (1.5 * eta[i])
        )
        obs_traj = mean_traj + rng.normal(0.0, 0.25, size=(n_vis, d_traj))
        mcar_t = rng.random((n_vis, d_traj)) < 0.05
        mcar_t[0, :] = False
        obs_traj[mcar_t] = np.nan

        enroll_rows.append(cursor)
        cursor += n_vis
        rows_pid.extend([patient_ids[i]] * n_vis)
        rows_futime.extend([float(followup_time[i])] * n_vis)
        rows_event.extend([int(event[i])] * n_vis)
        rows_vtime.extend(times_i.tolist())
        traj_blocks.append(obs_traj)

    traj_cols = tuple(f"traj_{j}" for j in range(d_traj))
    panel_cols = tuple(f"mol_{j}" for j in range(d_mol))
    panel_all = np.full((cursor, d_mol), np.nan, dtype=np.float64)
    panel_all[np.asarray(enroll_rows, dtype=np.int64)] = panel_base

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "patient_id": rows_pid,
                    "followup_time": np.asarray(rows_futime, dtype=np.float64),
                    "event": np.asarray(rows_event, dtype=np.int64),
                    "visit_time": np.asarray(rows_vtime, dtype=np.float64),
                }
            ),
            pd.DataFrame(np.vstack(traj_blocks), columns=list(traj_cols)),
            pd.DataFrame(panel_all, columns=list(panel_cols)),
        ],
        axis=1,
    )

    u_s32, u_t32, u_m32 = (
        u_shared.astype(np.float32),
        u_traj.astype(np.float32),
        u_mol.astype(np.float32),
    )
    gt_by_id: dict[str, OracleTuple] = {
        patient_ids[i]: (u_s32[i], u_t32[i], u_m32[i]) for i in range(n_patients)
    }
    name = f"synthetic_subspace_n{n_patients}_dt{d_traj}_dm{d_mol}_seed{seed}"
    return RawLongitudinalCohort(
        cohort_name=name,
        sha256=_hash_dataframe(name, df),
        df=df,
        traj_cols=traj_cols,
        panel_cols=panel_cols,
        ground_truth_by_id=gt_by_id,
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
        n_patients=n_patients,
        d_traj=d_traj,
        d_mol=d_mol,
        landmark_time=landmark_time,
        panel_missing_rate=panel_missing_rate,
        seed=seed,
    )
    cfg = LandmarkCohortConfig(
        landmark_time=landmark_time, train_frac=train_frac, val_frac=val_frac, seed=seed
    )
    return build_landmarked_splits(raw, cfg)


def load_raw_pbc2_cohort(
    parquet_path: Path | None = None,
    *,
    expected_sha256: str | None = PBC2_PARQUET_SHA256,
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
    required = {
        "patient_id",
        "followup_time",
        "event",
        "visit_time",
        *PBC2_TRAJ_COLS,
        *PBC2_RAW_PANEL_COLS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PBC2 parquet missing required columns: {sorted(missing)}")
    return RawLongitudinalCohort(
        cohort_name="pbc2_longitudinal",
        sha256=actual_sha256,
        df=df,
        traj_cols=PBC2_TRAJ_COLS,
        panel_cols=PBC2_RAW_PANEL_COLS,
        log_transform_traj_cols=PBC2_LOG_TRAJ_COLS,
        log_transform_panel_cols=PBC2_LOG_PANEL_COLS,
        panel_feature_names=PBC2_PANEL_FEATURE_NAMES,
        ground_truth_by_id=None,
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
