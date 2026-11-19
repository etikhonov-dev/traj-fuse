"""End-to-end CLI smoke tests for traj-fuse run and inspect-checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from traj_fuse.cli import main


def test_cli_pbc2_quick_and_inspect_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "pbc2_out"
    t0 = time.perf_counter()
    rc = main(["run", "--cohort", "pbc2", "--preset", "quick", "--out", str(out_dir)])
    elapsed = time.perf_counter() - t0
    assert rc == 0
    assert elapsed < 12.0

    report_path = out_dir / "benchmark_report.json"
    svg_path = out_dir / "pbc2_scarcity_ablation.svg"
    ckpt_dir = out_dir / "checkpoint"
    assert report_path.exists()
    assert svg_path.exists()
    assert "<svg" in svg_path.read_text("utf-8")
    assert (ckpt_dir / "config.json").exists()
    assert (ckpt_dir / "weights.pt").exists()

    report = json.loads(report_path.read_text("utf-8"))
    assert report["cohort"] == "pbc2"
    by_frac = {round(pt["fraction"], 4): pt for pt in report["scarcity"]["points"]}
    assert by_frac[0.01]["identifiable"] is False
    for f in (0.05, 0.10, 0.25, 1.00):
        assert by_frac[f]["identifiable"] is True
    assert report["stability"]["mean_linear_cka"] > 0.62
    assert report["stability"]["mean_procrustes_cosine"] > 0.70

    capsys.readouterr()
    rc_inspect = main(["inspect-checkpoint", "--checkpoint", str(ckpt_dir)])
    assert rc_inspect == 0
    captured = capsys.readouterr()
    info = json.loads(captured.out)
    assert info["all_frozen"] is True
    assert info["num_parameters"] > 0
