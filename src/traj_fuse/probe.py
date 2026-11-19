"""Survival probing (tie-exact Breslow Cox, IPCW linear), scarcity & missing-panel ablation."""

from __future__ import annotations

import hashlib
import html
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from traj_fuse.data import LandmarkedCohortSplits, SSLPretrainSplit
from traj_fuse.model import SharedPrivateConfig, SharedPrivateFuse
from traj_fuse.train import FrozenSSLCheckpoint

RIDGE_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)


def count_comparable_pairs(durations: NDArray[np.float64], events: NDArray[np.int64]) -> int:
    """Count comparable pairs (delta_i == 1 and (T_i < T_j or (T_i == T_j and delta_j == 0)))."""
    t, d = np.asarray(durations, dtype=np.float64), np.asarray(events, dtype=np.int64)
    comp = (d[:, None] == 1) & (
        (t[:, None] < t[None, :]) | ((t[:, None] == t[None, :]) & (d[None, :] == 0))
    )
    return int(comp.sum())


def concordance_index(
    durations: NDArray[np.float64], events: NDArray[np.int64], risk_scores: NDArray[np.float64]
) -> float:
    """Exact Harrell concordance index with 0.5 credit for tied risk predictions."""
    t = np.asarray(durations, dtype=np.float64)
    d = np.asarray(events, dtype=np.int64)
    r = np.asarray(risk_scores, dtype=np.float64)
    if not (t.ndim == d.ndim == r.ndim == 1 and len(t) == len(d) == len(r)):
        raise ValueError("durations, events, and risk_scores must be 1D arrays of equal length.")
    comp = (d[:, None] == 1) & (
        (t[:, None] < t[None, :]) | ((t[:, None] == t[None, :]) & (d[None, :] == 0))
    )
    n_comp = int(comp.sum())
    if n_comp == 0:
        raise ValueError("No comparable pairs available for concordance_index.")
    diff = r[:, None] - r[None, :]
    score = np.sum(comp & (diff > 1e-12)) + 0.5 * np.sum(comp & (np.abs(diff) <= 1e-12))
    return float(score / float(n_comp))


def _breslow_precompute(
    durations: NDArray[np.float64], events: NDArray[np.int64]
) -> tuple[int, list[tuple[NDArray[np.float64], NDArray[np.bool_]]]]:
    n_events = int((events == 1).sum())
    if n_events == 0:
        raise ValueError("At least one event (event == 1) is required for Breslow Cox.")
    u_times = np.unique(durations[events == 1])
    chunks: list[tuple[NDArray[np.float64], NDArray[np.bool_]]] = []
    for start in range(0, len(u_times), 256):
        tc = u_times[start : start + 256]
        d_g = (
            ((events[None, :] == 1) & (durations[None, :] == tc[:, None]))
            .sum(axis=1)
            .astype(np.float64)
        )
        r_mask = durations[None, :] >= tc[:, None]
        chunks.append((d_g, r_mask))
    return n_events, chunks


def _breslow_from_precomp(
    eta: NDArray[np.float64],
    events: NDArray[np.int64],
    n_events: int,
    chunks: list[tuple[NDArray[np.float64], NDArray[np.bool_]]],
) -> tuple[float, NDArray[np.float64]]:
    sum_log_s = 0.0
    grad_risk = np.zeros_like(eta)
    for d_g, r_mask in chunks:
        eta_m = np.where(r_mask, eta[None, :], -np.inf)
        m_g = np.max(eta_m, axis=1, keepdims=True)
        w = np.exp(eta_m - m_g)
        w_sum = np.sum(w, axis=1, keepdims=True)
        sum_log_s += float(np.sum(d_g * (m_g[:, 0] + np.log(w_sum[:, 0]))))
        grad_risk += np.sum((d_g[:, None] * w) / w_sum, axis=0)
    nll = -(float(np.sum(eta[events == 1])) - sum_log_s) / float(n_events)
    grad_eta = (-events.astype(np.float64) + grad_risk) / float(n_events)
    return nll, grad_eta


def breslow_nll_and_grad(
    eta: NDArray[np.float64], durations: NDArray[np.float64], events: NDArray[np.int64]
) -> tuple[float, NDArray[np.float64]]:
    """Tie-exact Breslow partial-likelihood NLL and gradient with per-risk-set max shift."""
    eta_f = np.asarray(eta, dtype=np.float64)
    t = np.asarray(durations, dtype=np.float64)
    d = np.asarray(events, dtype=np.int64)
    n_events, chunks = _breslow_precompute(t, d)
    return _breslow_from_precomp(eta_f, d, n_events, chunks)


@dataclass(frozen=True)
class LinearRiskModel:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    beta: NDArray[np.float64]
    intercept: float
    lam: float
    negate_prediction: bool = False

    def predict_risk(self, x: NDArray[np.float64] | NDArray[np.float32]) -> NDArray[np.float64]:
        xs = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        pred = xs @ self.beta + self.intercept
        return -pred if self.negate_prediction else pred


def _standardize_train(
    x: NDArray[np.float64] | NDArray[np.float32],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    xf = np.asarray(x, dtype=np.float64)
    mean = xf.mean(axis=0)
    std = xf.std(axis=0, ddof=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (xf - mean) / std, mean, std


def fit_ridge_cox(
    x: NDArray[np.float64] | NDArray[np.float32],
    durations: NDArray[np.float64],
    events: NDArray[np.int64],
    lam: float = 1.0,
) -> LinearRiskModel:
    """Fit L2-regularized tie-exact Breslow Cox model via L-BFGS-B."""
    from scipy.optimize import minimize

    xs, mean, std = _standardize_train(x)
    t, d = np.asarray(durations, dtype=np.float64), np.asarray(events, dtype=np.int64)
    n_events, chunks = _breslow_precompute(t, d)

    def obj_grad(beta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        nll, grad_eta = _breslow_from_precomp(xs @ beta, d, n_events, chunks)
        return nll + 0.5 * lam * float(np.dot(beta, beta)), xs.T @ grad_eta + lam * beta

    res = minimize(
        obj_grad,
        np.zeros(xs.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 60, "ftol": 1e-8},
    )
    return LinearRiskModel(
        mean=mean, std=std, beta=np.asarray(res.x, dtype=np.float64), intercept=0.0, lam=lam
    )


def fit_ipcw_linear_probe(
    x: NDArray[np.float64] | NDArray[np.float32],
    durations: NDArray[np.float64],
    events: NDArray[np.int64],
    lam: float = 1.0,
) -> LinearRiskModel:
    """Fit Kaplan-Meier IPCW ridge linear probe predicting log(T* + 1e-3); risk = -y_hat."""
    xs, mean, std = _standardize_train(x)
    t, d = np.asarray(durations, dtype=np.float64), np.asarray(events, dtype=np.int64)
    ev_idx = np.flatnonzero(d == 1)
    if ev_idx.size == 0:
        raise ValueError("At least one event required for IPCW linear probe.")

    c_times = np.unique(t[d == 0])
    if c_times.size > 0:
        factors = np.array(
            [
                1.0 - float(((t == tk) & (d == 0)).sum()) / max(float((t >= tk).sum()), 1.0)
                for tk in c_times
            ],
            dtype=np.float64,
        )
        g_curve = np.cumprod(factors)
        pos = np.searchsorted(c_times, t[ev_idx], side="right") - 1
        g_ev = np.where(pos >= 0, g_curve[np.maximum(pos, 0)], 1.0)
    else:
        g_ev = np.ones(ev_idx.size, dtype=np.float64)

    w = 1.0 / np.maximum(g_ev, 0.05)
    y = np.log(np.maximum(t[ev_idx], 1e-6) + 1e-3)
    x_ev = xs[ev_idx]
    w_sum = float(w.sum())
    x_bar = (w[:, None] * x_ev).sum(axis=0) / w_sum
    y_bar = float((w * y).sum() / w_sum)
    xc, yc = x_ev - x_bar, y - y_bar
    gram = xc.T @ (w[:, None] * xc) + lam * np.eye(xs.shape[1], dtype=np.float64)
    beta = np.linalg.solve(gram, xc.T @ (w * yc))
    intercept = float(y_bar - np.dot(x_bar, beta))
    return LinearRiskModel(
        mean=mean, std=std, beta=beta, intercept=intercept, lam=lam, negate_prediction=True
    )


def _select_lambda_cv(
    fitter: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], float], LinearRiskModel
    ],
    x: NDArray[np.float64] | NDArray[np.float32],
    durations: NDArray[np.float64],
    events: NDArray[np.int64],
    seed: int,
) -> LinearRiskModel:
    xf = np.asarray(x, dtype=np.float64)
    t, d = np.asarray(durations, dtype=np.float64), np.asarray(events, dtype=np.int64)
    if int((d == 1).sum()) < 6:
        return fitter(xf, t, d, 1.0)

    rng = np.random.default_rng(seed)
    folds = np.zeros(len(t), dtype=np.int64)
    for lbl in (0, 1):
        idx = np.flatnonzero(d == lbl)
        perm = idx[rng.permutation(len(idx))]
        folds[perm] = np.arange(len(perm)) % 3

    best_lam, best_score = 1.0, -1.0
    for lam in RIDGE_GRID:
        scores: list[float] = []
        for f in range(3):
            tr, va = folds != f, folds == f
            if int((d[tr] == 1).sum()) < 1 or count_comparable_pairs(t[va], d[va]) < 1:
                continue
            mdl = fitter(xf[tr], t[tr], d[tr], lam)
            scores.append(concordance_index(t[va], d[va], mdl.predict_risk(xf[va])))
        if scores:
            mean_s = float(np.mean(scores))
            if mean_s > best_score + 1e-12:
                best_score, best_lam = mean_s, lam
    return fitter(xf, t, d, best_lam)


def extract_locf_features(ssl: SSLPretrainSplit) -> NDArray[np.float32]:
    """Extract causal per-biomarker LOCF within [0, T_L] + ever-obs mask + enrollment panel."""
    steps = np.arange(ssl.traj_values.shape[1], dtype=np.int64)[None, :, None]
    valid = ssl.visit_mask[:, :, None] & (ssl.traj_obs_mask > 0.5)
    last_step = np.where(valid, steps, -1).max(axis=1)
    ever_obs = (last_step >= 0).astype(np.float32)
    safe_step = np.maximum(last_step, 0)
    locf_vals = (
        np.take_along_axis(ssl.traj_values, safe_step[:, None, :], axis=1)[:, 0, :] * ever_obs
    )
    return np.hstack([locf_vals, ever_obs, ssl.panel_values, ssl.panel_obs_mask]).astype(np.float32)


def _breslow_torch_loss(eta: Tensor, durations: Tensor, events: Tensor) -> Tensor:
    ev_mask = events == 1
    n_ev = int(ev_mask.sum().item())
    u_times = torch.unique(durations[ev_mask])
    e_mat = ev_mask.unsqueeze(0) & (durations.unsqueeze(0) == u_times.unsqueeze(1))
    d_g = e_mat.sum(dim=1).to(eta.dtype)
    r_mat = durations.unsqueeze(0) >= u_times.unsqueeze(1)
    log_s = torch.logsumexp(eta.unsqueeze(0).masked_fill(~r_mat, -torch.inf), dim=1)
    nll: Tensor = -(eta[ev_mask].sum() - (d_g * log_s).sum()) / float(n_ev)
    return nll


def _fit_eval_supervised_e2e(
    train_ssl: SSLPretrainSplit,
    idx: NDArray[np.int64],
    dur_sub: NDArray[np.float64],
    ev_sub: NDArray[np.int64],
    test_ssl: SSLPretrainSplit,
    test_dur: NDArray[np.float64],
    test_ev: NDArray[np.int64],
    config: SharedPrivateConfig,
    epochs: int,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    fuse = SharedPrivateFuse(config)
    head = nn.Linear(config.d_fused, 1)
    opt = torch.optim.AdamW(
        list(fuse.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-2
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    tv = torch.from_numpy(train_ssl.traj_values[idx])
    tm = torch.from_numpy(train_ssl.traj_obs_mask[idx])
    vd = torch.from_numpy(train_ssl.visit_deltas[idx])
    ll = torch.from_numpy(train_ssl.landmark_lags[idx])
    vm = torch.from_numpy(train_ssl.visit_mask[idx])
    pv = torch.from_numpy(train_ssl.panel_values[idx])
    pm = torch.from_numpy(train_ssl.panel_obs_mask[idx])
    pa = torch.from_numpy(train_ssl.panel_available[idx])
    dur_t, ev_t = torch.from_numpy(dur_sub), torch.from_numpy(ev_sub)

    fuse.train()
    head.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out = fuse(tv, tm, vd, ll, vm, train_ssl.landmark_time, pv, pm, pa)
        risk: Tensor = head(out.z_fused).squeeze(-1)
        loss = _breslow_torch_loss(risk, dur_t, ev_t)
        torch.autograd.backward(loss)
        opt.step()
        sched.step()

    fuse.eval()
    head.eval()
    with torch.inference_mode():
        t_out = fuse(
            torch.from_numpy(test_ssl.traj_values),
            torch.from_numpy(test_ssl.traj_obs_mask),
            torch.from_numpy(test_ssl.visit_deltas),
            torch.from_numpy(test_ssl.landmark_lags),
            torch.from_numpy(test_ssl.visit_mask),
            test_ssl.landmark_time,
            torch.from_numpy(test_ssl.panel_values),
            torch.from_numpy(test_ssl.panel_obs_mask),
            torch.from_numpy(test_ssl.panel_available),
        )
        t_risk = head(t_out.z_fused).squeeze(-1).cpu().numpy().astype(np.float64)
    return concordance_index(test_dur, test_ev, t_risk)


@dataclass(frozen=True)
class ScarcityPoint:
    fraction: float
    n_labeled: int
    identifiable: bool
    skip_reason: str | None
    traj_fuse_cox_c_index: float | None
    traj_fuse_ipcw_c_index: float | None
    supervised_e2e_c_index: float | None
    last_visit_cox_c_index: float | None


@dataclass(frozen=True)
class ScarcityEvaluation:
    points: tuple[ScarcityPoint, ...]


def _draw_seed(seed: int, fraction: float, draw_idx: int) -> int:
    digest = hashlib.sha256(f"{seed}:{fraction:.6f}:{draw_idx}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _sample_stratified_subset(
    events: NDArray[np.int64], n_sub: int, draw_seed: int
) -> NDArray[np.int64]:
    rng = np.random.default_rng(draw_seed)
    ev_idx = np.flatnonzero(events == 1)
    ne_idx = np.flatnonzero(events == 0)
    n_ev = min(len(ev_idx), max(1, int(round(n_sub * len(ev_idx) / len(events)))))
    n_ne = min(len(ne_idx), n_sub - n_ev)
    chosen = np.concatenate(
        [rng.choice(ev_idx, size=n_ev, replace=False), rng.choice(ne_idx, size=n_ne, replace=False)]
    )
    return np.sort(chosen).astype(np.int64)


def evaluate_scarcity_probes(
    splits: LandmarkedCohortSplits,
    checkpoint: FrozenSSLCheckpoint,
    *,
    fractions: tuple[float, ...] | None = None,
    n_draws: int = 5,
    seed: int = 42,
    e2e_epochs: int = 25,
) -> ScarcityEvaluation:
    """Evaluate frozen Cox & IPCW linear probes vs baselines across label scarcity fractions."""
    before_sha = checkpoint.parameter_sha256()
    tr_ssl, te_ssl = splits.train.to_ssl_split(), splits.test.to_ssl_split()
    n_train = len(tr_ssl.patient_ids)
    fracs = (
        fractions
        if fractions is not None
        else ((0.01, 0.05, 0.10, 1.00) if n_train >= 400 else (0.01, 0.05, 0.10, 0.25, 1.00))
    )

    emb_tr = checkpoint.encode(tr_ssl).z_fused
    emb_te = checkpoint.encode(te_ssl).z_fused
    locf_tr = extract_locf_features(tr_ssl)
    locf_te = extract_locf_features(te_ssl)
    tr_dur, tr_ev = splits.train.residual_duration, splits.train.event
    te_dur, te_ev = splits.test.residual_duration, splits.test.event

    points: list[ScarcityPoint] = []
    for f in fracs:
        n_sub = n_train if f >= 0.9999 else int(round(f * n_train))
        if n_sub < 8:
            points.append(
                ScarcityPoint(
                    f,
                    n_sub,
                    False,
                    f"n_labeled={n_sub} < min_identifiable=8",
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue

        draws = 1 if n_sub >= n_train else n_draws
        cox_s, ipcw_s, e2e_s, locf_s = [], [], [], []
        skip_reason: str | None = None
        for d_idx in range(draws):
            d_seed = _draw_seed(seed, f, d_idx)
            idx = (
                np.arange(n_train, dtype=np.int64)
                if n_sub >= n_train
                else _sample_stratified_subset(tr_ev, n_sub, d_seed)
            )
            ev_sub, dur_sub = tr_ev[idx], tr_dur[idx]
            n_e, n_ne = int((ev_sub == 1).sum()), int((ev_sub == 0).sum())
            n_cp = count_comparable_pairs(dur_sub, ev_sub)
            if n_e < 3 or n_ne < 2 or n_cp < 10:
                skip_reason = (
                    f"identifiability failed (events={n_e}, non_events={n_ne}, pairs={n_cp})"
                )
                break

            m_cox = _select_lambda_cv(fit_ridge_cox, emb_tr[idx], dur_sub, ev_sub, d_seed)
            m_ipcw = _select_lambda_cv(fit_ipcw_linear_probe, emb_tr[idx], dur_sub, ev_sub, d_seed)
            m_locf = _select_lambda_cv(fit_ridge_cox, locf_tr[idx], dur_sub, ev_sub, d_seed)
            cox_s.append(concordance_index(te_dur, te_ev, m_cox.predict_risk(emb_te)))
            ipcw_s.append(concordance_index(te_dur, te_ev, m_ipcw.predict_risk(emb_te)))
            locf_s.append(concordance_index(te_dur, te_ev, m_locf.predict_risk(locf_te)))
            e2e_s.append(
                _fit_eval_supervised_e2e(
                    tr_ssl,
                    idx,
                    dur_sub,
                    ev_sub,
                    te_ssl,
                    te_dur,
                    te_ev,
                    checkpoint.config,
                    e2e_epochs,
                    d_seed,
                )
            )

        if skip_reason is not None:
            points.append(ScarcityPoint(f, n_sub, False, skip_reason, None, None, None, None))
        else:
            points.append(
                ScarcityPoint(
                    f,
                    n_sub,
                    True,
                    None,
                    float(np.mean(cox_s)),
                    float(np.mean(ipcw_s)),
                    float(np.mean(e2e_s)),
                    float(np.mean(locf_s)),
                )
            )

    if checkpoint.parameter_sha256() != before_sha:
        raise RuntimeError("FrozenSSLCheckpoint parameters mutated during probing!")
    return ScarcityEvaluation(points=tuple(points))


@dataclass(frozen=True)
class MissingnessCurve:
    c_index_full: float
    c_index_drop50: float
    c_index_drop100: float
    delta_c_index_100: float


@dataclass(frozen=True)
class MissingPanelAblation:
    pdrop: MissingnessCurve
    nodrop: MissingnessCurve


def _eval_drop_curve(
    splits: LandmarkedCohortSplits, ckpt: FrozenSSLCheckpoint, mask_50: NDArray[np.bool_], seed: int
) -> MissingnessCurve:
    emb_tr = ckpt.encode(splits.train.to_ssl_split()).z_fused
    probe = _select_lambda_cv(
        fit_ridge_cox, emb_tr, splits.train.residual_duration, splits.train.event, seed
    )
    te_ssl, te_dur, te_ev = (
        splits.test.to_ssl_split(),
        splits.test.residual_duration,
        splits.test.event,
    )
    c_full = concordance_index(
        te_dur, te_ev, probe.predict_risk(ckpt.encode(te_ssl, force_drop_panel=False).z_fused)
    )
    c_50 = concordance_index(
        te_dur, te_ev, probe.predict_risk(ckpt.encode(te_ssl, force_drop_panel=mask_50).z_fused)
    )
    c_100 = concordance_index(
        te_dur, te_ev, probe.predict_risk(ckpt.encode(te_ssl, force_drop_panel=True).z_fused)
    )
    return MissingnessCurve(c_full, c_50, c_100, c_100 - c_full)


def evaluate_missing_panel_ablation(
    splits: LandmarkedCohortSplits,
    ckpt_pdrop: FrozenSSLCheckpoint,
    ckpt_nodrop: FrozenSSLCheckpoint,
    *,
    seed: int = 42,
) -> MissingPanelAblation:
    """Evaluate 0%, 50%, 100% test-time panel drop comparing p_drop=0.30 vs p_drop=0.0."""
    n_test = len(splits.test.ssl.patient_ids)
    rng = np.random.default_rng(seed)
    mask_50 = np.zeros(n_test, dtype=np.bool_)
    mask_50[rng.choice(n_test, size=n_test // 2, replace=False)] = True
    return MissingPanelAblation(
        pdrop=_eval_drop_curve(splits, ckpt_pdrop, mask_50, seed),
        nodrop=_eval_drop_curve(splits, ckpt_nodrop, mask_50, seed),
    )


@dataclass(frozen=True)
class CCAResult:
    canonical_corrs: tuple[float, ...]
    mean_cca: float
    perm_null_cca: float
    excess_cca: float


@dataclass(frozen=True)
class DisentanglementReport:
    cca_shared_to_true_shared: CCAResult
    cca_priv_traj_to_true_traj: CCAResult
    cca_priv_traj_to_true_mol: CCAResult
    cca_priv_mol_to_true_mol: CCAResult
    cca_priv_mol_to_true_traj: CCAResult


@dataclass(frozen=True)
class SeedPairStability:
    seed_a: int
    seed_b: int
    linear_cka: float
    mean_ridge_cca: float
    perm_null_cca: float
    excess_cca: float
    procrustes_cosine: float
    raw_unaligned_cosine: float


@dataclass(frozen=True)
class SeedStabilityReport:
    pairs: tuple[SeedPairStability, ...]
    mean_linear_cka: float
    mean_ridge_cca: float
    mean_perm_null_cca: float
    mean_excess_cca: float
    mean_procrustes_cosine: float
    mean_raw_unaligned_cosine: float


def _inv_sqrt_cov(xc: NDArray[np.float64], reg: float) -> NDArray[np.float64]:
    cov = (xc.T @ xc) / float(len(xc) - 1) + reg * np.eye(xc.shape[1], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(evals, 1e-12))
    return evecs @ (inv_sqrt[:, None] * evecs.T)


def ridge_cca_correlations(
    x: NDArray[np.float64] | NDArray[np.float32],
    y: NDArray[np.float64] | NDArray[np.float32],
    *,
    reg: float = 1e-2,
    n_perm: int = 20,
    seed: int = 42,
) -> CCAResult:
    """Compute ridge-regularized SVD-CCA with D_X + D_Y < N - 5 guard and permutation null."""
    xf, yf = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if xf.shape[0] != yf.shape[0]:
        raise ValueError("x and y must have identical row counts.")
    n, d_x = xf.shape
    d_y = yf.shape[1]
    if d_x + d_y >= n - 5:
        raise ValueError(f"CCA requires D_X + D_Y < N - 5; got {d_x} + {d_y} >= {n} - 5.")

    xc, yc = xf - xf.mean(axis=0), yf - yf.mean(axis=0)
    ix, iy = _inv_sqrt_cov(xc, reg), _inv_sqrt_cov(yc, reg)
    k = min(d_x, d_y)
    sv = np.clip(
        np.linalg.svd(ix @ ((xc.T @ yc) / float(n - 1)) @ iy, compute_uv=False)[:k], 0.0, 1.0
    )
    mean_cca = float(sv.mean())

    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_perm):
        yp = yc[rng.permutation(n)]
        sv_p = np.clip(
            np.linalg.svd(ix @ ((xc.T @ yp) / float(n - 1)) @ iy, compute_uv=False)[:k], 0.0, 1.0
        )
        nulls.append(float(sv_p.mean()))
    perm_null = float(np.mean(nulls))
    return CCAResult(tuple(float(v) for v in sv), mean_cca, perm_null, mean_cca - perm_null)


def linear_cka(
    x: NDArray[np.float64] | NDArray[np.float32], y: NDArray[np.float64] | NDArray[np.float32]
) -> float:
    """Centered Linear CKA similarity in [0, 1] (invariant to orthogonal rotation & scaling)."""
    xf, yf = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xc, yc = xf - xf.mean(axis=0), yf - yf.mean(axis=0)
    hsic_xy = float(np.sum((xc.T @ yc) ** 2))
    hsic_xx = float(np.sum((xc.T @ xc) ** 2))
    hsic_yy = float(np.sum((yc.T @ yc) ** 2))
    return float(hsic_xy / max(float(np.sqrt(hsic_xx * hsic_yy)), 1e-12))


def raw_unaligned_cosine(
    x: NDArray[np.float64] | NDArray[np.float32], y: NDArray[np.float64] | NDArray[np.float32]
) -> float:
    """Mean per-patient centered cosine similarity without basis alignment."""
    xf, yf = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xc, yc = xf - xf.mean(axis=0), yf - yf.mean(axis=0)
    nx = np.maximum(np.linalg.norm(xc, axis=1), 1e-12)
    ny = np.maximum(np.linalg.norm(yc, axis=1), 1e-12)
    return float(np.mean(np.sum(xc * yc, axis=1) / (nx * ny)))


def procrustes_cosine(
    x: NDArray[np.float64] | NDArray[np.float32], y: NDArray[np.float64] | NDArray[np.float32]
) -> float:
    """Mean per-patient centered cosine similarity after orthogonal Procrustes alignment."""
    xf, yf = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xc, yc = xf - xf.mean(axis=0), yf - yf.mean(axis=0)
    u, _, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    return raw_unaligned_cosine(xc @ (u @ vt), yc)


def evaluate_subspace_disentanglement(
    splits: LandmarkedCohortSplits,
    checkpoint: FrozenSSLCheckpoint,
    *,
    reg: float = 1e-2,
    n_perm: int = 20,
    seed: int = 42,
) -> DisentanglementReport:
    """Verify Tier-1 subspace recovery (filtering panel_available==True for z_priv_mol)."""
    gt = splits.test.ground_truth
    if gt is None:
        raise ValueError("Subspace disentanglement requires Tier-1 ground_truth subspaces.")
    emb = checkpoint.encode(splits.test.to_ssl_split())
    pa = splits.test.ssl.panel_available
    return DisentanglementReport(
        cca_shared_to_true_shared=ridge_cca_correlations(
            emb.z_shared, gt.u_shared, reg=reg, n_perm=n_perm, seed=seed
        ),
        cca_priv_traj_to_true_traj=ridge_cca_correlations(
            emb.z_priv_traj, gt.u_traj, reg=reg, n_perm=n_perm, seed=seed
        ),
        cca_priv_traj_to_true_mol=ridge_cca_correlations(
            emb.z_priv_traj, gt.u_mol, reg=reg, n_perm=n_perm, seed=seed
        ),
        cca_priv_mol_to_true_mol=ridge_cca_correlations(
            emb.z_priv_mol[pa], gt.u_mol[pa], reg=reg, n_perm=n_perm, seed=seed
        ),
        cca_priv_mol_to_true_traj=ridge_cca_correlations(
            emb.z_priv_mol[pa], gt.u_traj[pa], reg=reg, n_perm=n_perm, seed=seed
        ),
    )


def evaluate_seed_stability(
    splits: LandmarkedCohortSplits,
    checkpoints_by_seed: dict[int, FrozenSSLCheckpoint],
    *,
    reg: float = 1e-2,
    n_perm: int = 20,
    seed: int = 42,
) -> SeedStabilityReport:
    """Evaluate pairwise cross-seed stability on held-out test z_fused embeddings."""
    if len(checkpoints_by_seed) < 2:
        raise ValueError("Need at least 2 seed checkpoints to evaluate cross-seed stability.")
    te_ssl = splits.test.to_ssl_split()
    embs = {s: ckpt.encode(te_ssl).z_fused for s, ckpt in sorted(checkpoints_by_seed.items())}
    pairs: list[SeedPairStability] = []
    for sa, sb in combinations(sorted(embs), 2):
        za, zb = embs[sa], embs[sb]
        cca = ridge_cca_correlations(za, zb, reg=reg, n_perm=n_perm, seed=seed)
        pairs.append(
            SeedPairStability(
                sa,
                sb,
                linear_cka(za, zb),
                cca.mean_cca,
                cca.perm_null_cca,
                cca.excess_cca,
                procrustes_cosine(za, zb),
                raw_unaligned_cosine(za, zb),
            )
        )
    return SeedStabilityReport(
        pairs=tuple(pairs),
        mean_linear_cka=float(np.mean([p.linear_cka for p in pairs])),
        mean_ridge_cca=float(np.mean([p.mean_ridge_cca for p in pairs])),
        mean_perm_null_cca=float(np.mean([p.perm_null_cca for p in pairs])),
        mean_excess_cca=float(np.mean([p.excess_cca for p in pairs])),
        mean_procrustes_cosine=float(np.mean([p.procrustes_cosine for p in pairs])),
        mean_raw_unaligned_cosine=float(np.mean([p.raw_unaligned_cosine for p in pairs])),
    )


def _svg_panel(
    x0: float,
    y0: float,
    w: float,
    h: float,
    title: str,
    x_labels: list[str],
    series: list[tuple[str, str, list[float]]],
    note: str | None = None,
) -> list[str]:
    vals = [v for _, _, ys in series for v in ys]
    ymin = max(0.45, min(vals) - 0.04)
    ymax = min(1.0, max(vals) + 0.04)
    if ymax - ymin < 0.10:
        ymax = min(1.0, ymin + 0.10)
    lines = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>',
        f'<text x="{x0 + w / 2:.1f}" y="{y0 - 10:.1f}" text-anchor="middle" '
        f'font-size="13" font-weight="600" fill="#0f172a">{html.escape(title)}</text>',
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = ymin + frac * (ymax - ymin)
        py = y0 + h - frac * h
        lines.append(
            f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x0 + w:.1f}" y2="{py:.1f}" '
            'stroke="#e2e8f0" stroke-dasharray="3,3"/>'
        )
        lines.append(
            f'<text x="{x0 - 6:.1f}" y="{py + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="#475569">{yv:.2f}</text>'
        )
    n_x = max(len(x_labels), 1)
    xs = [x0 + (i + 0.5) * (w / n_x) for i in range(n_x)]
    for px, lbl in zip(xs, x_labels, strict=True):
        lines.append(
            f'<text x="{px:.1f}" y="{y0 + h + 16:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#334155">{html.escape(lbl)}</text>'
        )
    for idx, (name, color, ys) in enumerate(series):
        pts = " ".join(
            f"{px:.1f},{y0 + h - ((yv - ymin) / (ymax - ymin)) * h:.1f}"
            for px, yv in zip(xs, ys, strict=True)
        )
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{pts}"/>')
        for px, yv in zip(xs, ys, strict=True):
            py = y0 + h - ((yv - ymin) / (ymax - ymin)) * h
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}"/>')
        ly = y0 + 16 + idx * 16
        lines.append(
            f'<line x1="{x0 + 12:.1f}" y1="{ly:.1f}" x2="{x0 + 28:.1f}" y2="{ly:.1f}" '
            f'stroke="{color}" stroke-width="2.2"/>'
        )
        lines.append(
            f'<text x="{x0 + 33:.1f}" y="{ly + 3.5:.1f}" font-size="10" '
            f'fill="#1e293b">{html.escape(name)}</text>'
        )
    if note:
        lines.append(
            f'<text x="{x0 + 8:.1f}" y="{y0 + h - 8:.1f}" font-size="9.5" '
            f'fill="#991b1b">{html.escape(note)}</text>'
        )
    return lines


def render_scarcity_ablation_svg(
    scarcity: ScarcityEvaluation,
    ablation: MissingPanelAblation,
    out_path: Path,
    *,
    title_prefix: str = "traj-fuse",
) -> Path:
    """Render deterministic 2-panel SVG for label-scarcity C-index & missing-panel ablation."""
    id_pts = [p for p in scarcity.points if p.identifiable]
    x_lbls = [f"{int(round(100 * p.fraction))}% (n={p.n_labeled})" for p in id_pts]
    s_left = [
        (
            "traj-fuse frozen (Cox)",
            "#0f766e",
            [float(p.traj_fuse_cox_c_index or 0.5) for p in id_pts],
        ),
        (
            "traj-fuse frozen (IPCW)",
            "#0284c7",
            [float(p.traj_fuse_ipcw_c_index or 0.5) for p in id_pts],
        ),
        ("supervised e2e", "#d97706", [float(p.supervised_e2e_c_index or 0.5) for p in id_pts]),
        (
            "last-visit LOCF Cox",
            "#64748b",
            [float(p.last_visit_cox_c_index or 0.5) for p in id_pts],
        ),
    ]
    skipped = [p for p in scarcity.points if not p.identifiable]
    note = (
        "Excluded (<8 labels): "
        + ", ".join(f"{int(round(100 * p.fraction))}% (n={p.n_labeled})" for p in skipped)
        if skipped
        else None
    )
    s_right = [
        (
            "SharedPrivateFuse (p_drop=0.30)",
            "#0f766e",
            [
                ablation.pdrop.c_index_full,
                ablation.pdrop.c_index_drop50,
                ablation.pdrop.c_index_drop100,
            ],
        ),
        (
            "Ablation (p_drop=0.0)",
            "#dc2626",
            [
                ablation.nodrop.c_index_full,
                ablation.nodrop.c_index_drop50,
                ablation.nodrop.c_index_drop100,
            ],
        ),
    ]
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="940" height="360" '
        'viewBox="0 0 940 360" font-family="ui-sans-serif, system-ui, sans-serif">',
        '<rect width="940" height="360" fill="#f8fafc"/>',
        *_svg_panel(
            62, 40, 380, 265, f"{title_prefix}: Label-Scarcity Probing", x_lbls, s_left, note
        ),
        *_svg_panel(
            520,
            40,
            380,
            265,
            f"{title_prefix}: Missing-Panel Ablation",
            ["0% drop", "50% drop", "100% drop"],
            s_right,
        ),
        '<text x="16" y="180" transform="rotate(-90 16 180)" text-anchor="middle" '
        'font-size="11" fill="#334155">Held-out Test C-index</text>',
        "</svg>",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body) + "\n", "utf-8")
    return out_path
