# ADR-0001: Hard Landmark Truncation & Enrollment-Only Panel Firewall

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Longitudinal multimodal survival models frequently suffer from two subtle leakage pathways:
1. **Immortal-time & post-landmark leakage:** Including visits at $t_{ij} > T_L$ or keeping subjects whose event/censoring occurred at $T_i \le T_L$ leaks post-landmark survival outcomes into trajectory scaling, sequence padding (`L_max`), and encoder representations.
2. **Future-visit baseline backfill & demographic confounding:** Calling `groupby("patient_id").first()` in pandas (default `skipna=True`) silently backfills missing enrollment ($t = 0$) staging/lab values from later follow-up visits ($t > 0$). Furthermore, placing universal EHR demographics (`age`, `sex`) inside the drop-ablated "molecular/staging" modality artificially inflates missing-panel ablation drops.

## Decision

1. **Hard Landmark Truncation ($T_L$):** `build_landmarked_splits()` filters risk-set eligibility (`followup_time > T_L`), physically discards every visit row with `visit_time > T_L` before computing padding or train-only scalers, and resets downstream survival clocks to `residual_duration = followup_time - T_L > 0`.
2. **Strict Enrollment-Only Panel Extraction (`skipna=False`):** On Mayo `PBC2`, Modality 2 (`histologic`, `baseline_serChol`, `ascites`, `hepatomegaly`, `spiders`, `edema_score`) is extracted strictly from the $t = 0.0$ enrollment row without forward-backfilling, while universal `age` and `sex_female` stay in Modality 1 (`D_traj = 8`).
3. **Outcome-Blind `SSLPretrainSplit` Contract:** `pretrain_ssl()` and `FrozenSSLCheckpoint.encode()` enforce `isinstance(ssl_split, SSLPretrainSplit)`, physically excluding `residual_duration`, `event`, and Tier-1 oracle subspaces (`u_shared`, `u_traj`, `u_mol`).

## Rejected Alternatives

- **Post-hoc masking of $t > T_L$ visits inside `DataLoader`:** Rejected because `L_max` padding and global z-scoring still leak post-landmark summary statistics.
- **Imputing missing enrollment `serChol` from follow-up visits:** Rejected because future clinic visits occur conditionally on post-enrollment survival.

## Consequences

- Perturbing post-landmark visits ($t > T_L$), follow-up staging columns ($t > 0$), or `val`/`test` cohorts leaves standardized `train` tensors bit-identical (`np.array_equal`).
- `Tier 1` post-landmark latent subspaces $(U^{\text{shared}}, U^{\text{traj}}, U^{\text{mol}})$ remain mutually orthogonal (`max |corr| < 0.08`) due to exogenous collider-free pre-landmark attrition.
