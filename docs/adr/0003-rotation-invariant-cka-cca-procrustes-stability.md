# ADR-0003: Rotation-Invariant CKA, Ridge SVD-CCA & Procrustes Stability

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Evaluating representation stability and subspace disentanglement across random pretraining seeds via naive coordinate-wise cosine similarity or unregularized high-dimensional CCA produces two opposing measurement failures:
1. **False instability via orthogonal basis ambiguity:** VICReg and masked autoencoding objectives are equivariant to orthogonal rotations $Q \in O(D)$ of latent axes. Two seeds learning near-identical patient manifolds ($Z^{(2)} \approx Z^{(1)} Q$) routinely exhibit `raw_unaligned_cosine ≈ 0.0`.
2. **Grassmann CCA degeneracy:** When $2D \ge N - 1$ (e.g., $D=32, N=56$ on `PBC2` test), unregularized CCA finds spurious canonical correlations near `1.0` even between two independent Gaussian noise matrices.

## Decision

1. **Compact Latent Dimensions:** Default to non-overcomplete latents (`4/4 -> D=12` on Tier 1; `4/2 -> D=8` on Mayo `PBC2`) so $2D \ll N_{\text{test}} - 1$.
2. **Permutation-Calibrated Ridge SVD-CCA:** `ridge_cca_correlations()` enforces $D_X + D_Y < N - 5$, uses uniform $1/(N-1)$ covariance scaling with $\text{reg} = 10^{-2}$, and reports `perm_null_cca` (20 row permutations) alongside `excess_cca = mean_cca - perm_null_cca`.
3. **Orthogonal-Invariant Stability Suite:** Report **Linear CKA**, **ridge SVD-CCA**, and **orthogonal Procrustes-aligned cosine similarity** ($\min_{Q \in O(D)} \|\tilde{Z}^{(a)} Q - \tilde{Z}^{(b)}\|_F$) explicitly alongside `raw_unaligned_cosine` across seeds `(11, 22, 33)`.

## Rejected Alternatives

- **Raw coordinate cosine similarity alone:** Rejected because random Haar rotations $Q \in O(D)$ collapse `raw_unaligned_cosine` despite zero geometric change.
- **Unregularized CCA without $N$ guard or permutation null:** Rejected because finite-sample collinearity inflates white-noise correlations above `0.9`.

## Consequences

- Random Haar rotations preserve `linear_cka`, `mean_ridge_cca`, and `procrustes_cosine` within `atol=1e-5` while collapsing `raw_unaligned_cosine`.
- Cross-seed `(11, 22, 33)` stability on `PBC2` demonstrates `procrustes_cosine = 0.746` and `excess_cca = 0.325` even when `raw_unaligned_cosine = 0.136`.
