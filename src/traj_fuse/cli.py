"""CLI for traj-fuse benchmarks (`run`) and checkpoint inspection (`inspect-checkpoint`)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from traj_fuse.data import load_pbc2_cohort, make_synthetic_subspace_cohort
from traj_fuse.model import SharedPrivateConfig
from traj_fuse.probe import (
    evaluate_missing_panel_ablation,
    evaluate_scarcity_probes,
    evaluate_seed_stability,
    evaluate_subspace_disentanglement,
    render_scarcity_ablation_svg,
)
from traj_fuse.train import FrozenSSLCheckpoint, pretrain_ssl


def _run_benchmark(args: argparse.Namespace) -> int:
    quick = args.preset == "quick"
    splits = (
        load_pbc2_cohort(landmark_time=2.0, seed=args.seed)
        if args.cohort == "pbc2"
        else make_synthetic_subspace_cohort(n_patients=1_400, seed=args.seed)
    )
    epochs = args.epochs if args.epochs is not None else (14 if quick else 30)
    batch_size = 64 if args.cohort == "pbc2" else 128
    n_draws = 2 if quick else 5
    e2e_epochs = 5 if quick else 20
    seeds = (11, 22) if quick else (11, 22, 33)

    cfg_drop = SharedPrivateConfig.for_cohort(splits, p_drop=0.30)
    cfg_nodrop = SharedPrivateConfig.for_cohort(splits, p_drop=0.0)
    tr_ssl = splits.train.to_ssl_split()

    ckpts = {
        s: pretrain_ssl(tr_ssl, cfg_drop, epochs=epochs, batch_size=batch_size, seed=s)
        for s in seeds
    }
    primary = ckpts[seeds[0]]
    ckpt_nodrop = pretrain_ssl(
        tr_ssl, cfg_nodrop, epochs=epochs, batch_size=batch_size, seed=seeds[0]
    )

    scarcity = evaluate_scarcity_probes(
        splits, primary, n_draws=n_draws, seed=args.seed, e2e_epochs=e2e_epochs
    )
    ablation = evaluate_missing_panel_ablation(splits, primary, ckpt_nodrop, seed=args.seed)
    stability = evaluate_seed_stability(splits, ckpts, seed=args.seed)

    report: dict[str, Any] = {
        "cohort": args.cohort,
        "cohort_name": splits.cohort_name,
        "cohort_sha256": splits.cohort_sha256,
        "preset": args.preset,
        "epochs": epochs,
        "seeds": list(seeds),
        "n_train": len(splits.train.ssl.patient_ids),
        "n_val": len(splits.val.ssl.patient_ids),
        "n_test": len(splits.test.ssl.patient_ids),
        "scarcity": asdict(scarcity),
        "missing_panel_ablation": asdict(ablation),
        "stability": asdict(stability),
    }
    if splits.test.ground_truth is not None:
        report["disentanglement"] = asdict(
            evaluate_subspace_disentanglement(splits, primary, seed=args.seed)
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary.save(out_dir / "checkpoint")
    svg_path = out_dir / f"{args.cohort}_scarcity_ablation.svg"
    title_prefix = "Mayo PBC2" if args.cohort == "pbc2" else "Tier-1 Subspace"
    render_scarcity_ablation_svg(scarcity, ablation, svg_path, title_prefix=title_prefix)
    (out_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2), "utf-8")
    return 0


def _inspect_checkpoint(args: argparse.Namespace) -> int:
    ckpt = FrozenSSLCheckpoint.load(Path(args.checkpoint))
    params = list(ckpt.model.parameters())
    payload = {
        "config": ckpt.config.model_dump(),
        "training": ckpt.model.training,
        "all_frozen": all(not p.requires_grad for p in params),
        "num_parameters": int(sum(p.numel() for p in params)),
        "parameter_sha256": ckpt.parameter_sha256(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="traj-fuse", description="traj-fuse survival SSL CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="Pretrain SSL and run scarcity, ablation & stability probes."
    )
    p_run.add_argument("--cohort", choices=["pbc2", "synthetic"], default="pbc2")
    p_run.add_argument("--preset", choices=["full", "quick"], default="full")
    p_run.add_argument("--out", default="artifacts/pbc2")
    p_run.add_argument("--epochs", type=int, default=None)
    p_run.add_argument("--seed", type=int, default=42)

    p_ins = sub.add_parser("inspect-checkpoint", help="Load frozen checkpoint (weights_only=True).")
    p_ins.add_argument("--checkpoint", required=True)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "run":
        return _run_benchmark(args)
    return _inspect_checkpoint(args)


if __name__ == "__main__":
    raise SystemExit(main())
