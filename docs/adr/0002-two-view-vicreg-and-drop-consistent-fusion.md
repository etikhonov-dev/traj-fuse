# ADR-0002: Two-View VICReg & Drop-Consistent Modality-Gated Fusion

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Multimodal representation learning for clinical survival faces three architectural failure modes:
1. **Radial severity erasure:** Applying per-sample `LayerNorm` on output embeddings $z^{\text{shared}}$ projects every patient onto a fixed-radius hypersphere, destroying monotonic biomarker severity magnitudes needed by linear/Cox proportional-hazards probes.
2. **Silent missing-modality dead paths:** Applying `nn.Dropout` to binary observation masks corrupts `{0.0, 1.0}` missingness semantics, and computing VICReg/reconstruction only on observed panels leaves missing-panel tokens ($b_{\text{miss}}^{\text{mol}}$) and fallback adapters with zero gradient when `panel_available == True`.
3. **Small-batch variance collapse & rank inflation:** Uncoupled `VICReg` invariance weights (`25x` larger than reconstruction) can over-focus on easy shared correlations while ignoring masked reconstruction.

## Decision

1. **Unnormalized Output Heads & Mask-Safe Encoders:** Both `IrregularTrajectoryEncoder` (bidirectional pre-norm SDPA over $[0, T_L]$ with `-inf`-masked landmark-lag pooling) and `MolecularEncoder` emit unnormalized split heads ($z^{\text{shared}}, z^{\text{priv}}$) without per-sample `LayerNorm` on outputs and without `nn.Dropout` on binary masks.
2. **Clean/Corrupted Two-View SSL Objective:**
   - **Clean view ($m_i^{\text{obs}}$):** Computes $\mathcal{L}_{\text{VICReg}}$ (with variance hinge + unbiased covariance) and paired-safe centered cross-subspace orthogonality $\mathcal{L}_{\text{orth}}$ matching `.eval()` input statistics.
   - **Corrupted view (`p_mask = 0.20`, `p_drop = 0.30`):** Jointly masks `(value, obs_mask) -> (0, 0)`, drops paired panels with probability `p_drop`, routes masked trajectory/panel reconstruction ($\mathcal{L}_{\text{recon}}$) through $z^{\text{fused}}(m^{\text{drop}})$, and minimizes missing-panel self-distillation $\mathcal{L}_{\text{drop}} = \|z^{\text{fused, corrupt}}(0) - \operatorname{sg}(z^{\text{fused, clean}}(1))\|_2^2$.
   - Balanced loss weights (`lambda_vic=0.25, lambda_orth=0.25, lambda_recon=5.0, lambda_drop=2.0`) prevent VICReg from drowning out reconstruction.

## Rejected Alternatives

- **Per-sample `LayerNorm` on `z_shared`:** Rejected because it collapses radial disease severity scaling (`enc(base * s)`).
- **Zero-only missing imputation without `L_drop`:** Rejected because `b_miss_mol` and `mlp_fuse(..., m=0)` receive `0.0` gradient on paired cohorts when `p_drop=0.0`.

## Consequences

- `b_miss_mol` and `mlp_fuse` receive strictly non-zero gradients under `p_drop=0.30` (and exact `0.0` gradient on `b_miss_mol` when `p_drop=0.0` on paired batches).
- Held-out missing-panel MSE $\|z^{\text{fused}}(0) - z^{\text{fused}}(1)\|_2^2$ is strictly lower for `p_drop=0.30` than `p_drop=0.0`.
