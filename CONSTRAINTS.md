# Constraints

Last reviewed: 2026-11-19 by @etikhonov

## Floor (always enforced, no setup required)

- No new suppression comments: `# noqa`, `# type: ignore`, `pyright: ignore`
- No unimplemented stubs: `raise NotImplementedError`, `TODO`, `pass # stub`, silent `except ...: pass`
- No skipped or deleted tests without a reason in the commit message
- No secrets in source
- This file does not get weakened to make a change pass

## Enforced with numbers

| Dimension | Rule | Checked by | Runs at |
|-----------|------|-----------|---------|
| Types | Zero type errors (`0` errors) | `uv run mypy --strict src` | task end, CI |
| Lint & Static Security | Zero ruff or bandit errors (`0` errors) | `uv run ruff check src tests scripts` | every edit, CI |
| Format | Zero formatting diffs (`0` diffs) | `uv run ruff format --check src tests scripts` | every edit, CI |
| Coverage | Line coverage >= 90% on `src/traj_fuse` | `uv run pytest --cov=traj_fuse --cov-fail-under=90` | task end, CI |
| Security: deps | Zero known vulnerabilities (`0` vulnerabilities) | `uv run pip-audit` | task end, CI |
| Floor guard | Zero floor violations (`0` violations) | `python3 scripts/floor_guard.py --base main` | task end, CI |
| Performance: cohorts | Tier 1 and Tier 2 cohort load <= 300 ms | `uv run pytest tests/test_splits.py` | task end, CI |
| Performance: suite | Full pytest suite <= 25.0 s on CPU | `uv run pytest` | task end, CI |
| Performance: benchmark | Full 3-seed PBC2 CLI benchmark <= 25.0 s on CPU | `uv run traj-fuse run --cohort pbc2` | task end, CI |
| Scope: source LOC | Total `src/traj_fuse` source <= 1100 LOC | `wc -l src/traj_fuse/*.py` | task end, CI |

## Measured, not yet enforced

| Metric | Today | Direction |
|--------|-------|-----------|
| Package line coverage | — | must not fall below 90% |

## Exceptions

| ID | Rule | Path | Reason | Owner | Expires |
|----|------|------|--------|-------|---------|
