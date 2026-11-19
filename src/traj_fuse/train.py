"""Outcome-blind SSL pretraining loop and FrozenSSLCheckpoint serialization."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from traj_fuse.data import SSLPretrainSplit
from traj_fuse.model import SharedPrivateConfig, SharedPrivateFuse, SSLPretrainModule


def configure_determinism(seed: int) -> None:
    """Configure single-thread deterministic CPU execution for torch and numpy."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


@dataclass(frozen=True)
class PatientEmbeddings:
    patient_ids: tuple[str, ...]
    z_shared: NDArray[np.float32]
    z_priv_traj: NDArray[np.float32]
    z_priv_mol: NDArray[np.float32]
    z_fused: NDArray[np.float32]
    panel_active: NDArray[np.bool_]


@dataclass(frozen=True)
class FrozenSSLCheckpoint:
    config: SharedPrivateConfig
    model: SharedPrivateFuse
    history: tuple[dict[str, float], ...]

    @classmethod
    def from_model(
        cls,
        config: SharedPrivateConfig,
        model: SharedPrivateFuse,
        history: tuple[dict[str, float], ...] = (),
    ) -> FrozenSSLCheckpoint:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(config=config, model=model, history=history)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(self.config.model_dump_json(indent=2), "utf-8")
        torch.save(self.model.state_dict(), directory / "weights.pt")
        return directory

    @classmethod
    def load(cls, directory: Path) -> FrozenSSLCheckpoint:
        cfg = SharedPrivateConfig.model_validate_json(
            (directory / "config.json").read_text("utf-8")
        )
        model = SharedPrivateFuse(cfg)
        state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return cls.from_model(config=cfg, model=model)

    def parameter_sha256(self) -> dict[str, str]:
        return {
            n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            for n, p in self.model.named_parameters()
        }

    def encode(
        self, ssl_split: SSLPretrainSplit, force_drop_panel: bool | NDArray[np.bool_] = False
    ) -> PatientEmbeddings:
        if not isinstance(ssl_split, SSLPretrainSplit):
            raise TypeError("encode() requires an outcome-blind SSLPretrainSplit.")
        self.model.eval()
        n = len(ssl_split.patient_ids)
        drop_arr = (
            np.full(n, force_drop_panel, np.bool_)
            if isinstance(force_drop_panel, bool)
            else np.asarray(force_drop_panel, np.bool_)
        )
        if drop_arr.shape != (n,):
            raise ValueError(f"force_drop_panel shape {drop_arr.shape} != ({n},)")
        active = ssl_split.panel_available & (~drop_arr)

        s = ssl_split
        with torch.inference_mode():
            out = self.model(
                torch.from_numpy(s.traj_values),
                torch.from_numpy(s.traj_obs_mask),
                torch.from_numpy(s.visit_deltas),
                torch.from_numpy(s.landmark_lags),
                torch.from_numpy(s.visit_mask),
                s.landmark_time,
                torch.from_numpy(s.panel_values),
                torch.from_numpy(s.panel_obs_mask),
                torch.from_numpy(active),
            )
        return PatientEmbeddings(
            s.patient_ids,
            out.z_shared.cpu().numpy().astype(np.float32),
            out.z_traj_priv.cpu().numpy().astype(np.float32),
            out.z_mol_priv_tilde.cpu().numpy().astype(np.float32),
            out.z_fused.cpu().numpy().astype(np.float32),
            active.copy(),
        )


def pretrain_ssl(
    ssl_split: SSLPretrainSplit,
    config: SharedPrivateConfig,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> FrozenSSLCheckpoint:
    """Train SharedPrivateFuse strictly on outcome-blind SSLPretrainSplit."""
    if not isinstance(ssl_split, SSLPretrainSplit):
        raise TypeError("pretrain_ssl() accepts ONLY an outcome-blind SSLPretrainSplit.")
    configure_determinism(seed)

    ssl_mod = SSLPretrainModule(config)
    ssl_mod.train()
    opt = torch.optim.AdamW(ssl_mod.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    gen = torch.Generator().manual_seed(seed)

    s = ssl_split
    tv, tm, vd, ll, vm = (
        torch.from_numpy(s.traj_values),
        torch.from_numpy(s.traj_obs_mask),
        torch.from_numpy(s.visit_deltas),
        torch.from_numpy(s.landmark_lags),
        torch.from_numpy(s.visit_mask),
    )
    pv, pm, pa = (
        torch.from_numpy(s.panel_values),
        torch.from_numpy(s.panel_obs_mask),
        torch.from_numpy(s.panel_available),
    )
    n_total = len(s.patient_ids)
    bs = min(batch_size, n_total)

    history: list[dict[str, float]] = []
    for _ in range(epochs):
        perm = torch.randperm(n_total, generator=gen)
        acc = {"loss": 0.0, "vicreg": 0.0, "orth": 0.0, "recon": 0.0, "drop": 0.0}
        n_steps = 0
        for start in range(0, n_total, bs):
            idx = perm[start : start + bs]
            if idx.numel() == 1 and n_total >= 4:
                continue
            opt.zero_grad(set_to_none=True)
            loss, metrics = ssl_mod.forward_loss(
                tv[idx],
                tm[idx],
                vd[idx],
                ll[idx],
                vm[idx],
                s.landmark_time,
                pv[idx],
                pm[idx],
                pa[idx],
                generator=gen,
            )
            torch.autograd.backward(loss)
            opt.step()
            for k, v in metrics.items():
                acc[k] += v
            n_steps += 1
        sched.step()
        history.append({k: v / max(n_steps, 1) for k, v in acc.items()})

    return FrozenSSLCheckpoint.from_model(config=config, model=ssl_mod.fuse, history=tuple(history))
