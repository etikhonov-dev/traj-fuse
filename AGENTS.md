# Agent Guide: `traj-fuse`

Read `CONSTRAINTS.md` before writing code. Do not weaken it to make a change pass.

## Repository Rules

- Follow test-driven development (`RED -> GREEN -> REFACTOR`): write failing tests before implementation.
- Enforce hard landmark truncation (`followup_time > T_L`, `visit_time <= T_L`) and enrollment-only (`t = 0`, `skipna=False`) baseline panel extraction before fitting any scaler.
- Keep `age` and `sex_female` in Modality 1 (`D_traj = 8`) on Mayo `PBC2` so missing-panel ablation tests specialty staging rather than basic demographics.
- Never pass survival outcomes or ground-truth oracle subspaces into `SSLPretrainSplit` or `train.py`.
- Keep `src/traj_fuse/` surgical (`<= 1,100` LOC total) and maintain clean feature branches merged to `main` after all quality gates pass.

## Standard Verification

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy --strict src
uv run pytest --cov=traj_fuse --cov-fail-under=90
uv run pip-audit
python3 scripts/floor_guard.py --base main
```
