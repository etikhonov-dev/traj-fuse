# Mayo Clinic Primary Biliary Cirrhosis Longitudinal Cohort (`PBC2` / `pbcseq`)

## Provenance & License

- **Upstream source:** `https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv` (`survival::pbcseq`, equivalent to `JM::pbc2`)
- **Raw CSV SHA-256:** `03aca2f2e449c388b14ec130383654a09d91240f85f935ce364d206c8e848aed` (156,390 bytes)
- **License:** GPL (>= 2) per CRAN `survival` / `JM` packages
- **Output file:** `data/pbc2_longitudinal.parquet` (60,460 bytes)
- **Output SHA-256:** `27f132e9f3b18a5889c2af2b35cdd42602ecc70bc9f0ca575d7c2475879fc8b3`

## Cohort Summary

- **Patients:** `312` (`100%` have an enrollment visit at `t = 0.0`)
- **Irregular clinic visits:** `1,945` (min `1`, median `5.0`, max `16`)
- **Follow-up range (`followup_time`):** `0.112` to `14.305` years
- **Composite transplant-free survival endpoint (`event = 1`):** `169` (`140` dead + `29` transplanted; `143` alive)
- **Landmark risk set (`followup_time > 2.0` yr):** `278` patients (`135` composite events, `48.6%`)
- **Excluded columns:** `trt` (50/50 randomized drug arm: `158` D-penicillamine, `154` placebo) and raw 3-level `status` are excluded from the parquet.

## Modality Firewall & Missingness

- Modality 1 (`D_traj = 8`) keeps demographics (`age`, `sex_female`) with routine blood labs so missing-panel ablations isolate specialty staging.
- Modality 2 (`D_mol = 6`) is extracted strictly from enrollment `t = 0.0` with `skipna=False` (renaming `serChol -> baseline_serChol` in `data.py`).

| Feature | Modality | Missing Count | Missing Rate |
|---|---|---:|---:|
| `serBilir` | Modality 1 (Visit trajectory) | 0 | 0.00% |
| `albumin` | Modality 1 (Visit trajectory) | 0 | 0.00% |
| `alkaline` | Modality 1 (Visit trajectory) | 60 | 3.08% |
| `SGOT` | Modality 1 (Visit trajectory) | 0 | 0.00% |
| `platelets` | Modality 1 (Visit trajectory) | 73 | 3.75% |
| `prothrombin` | Modality 1 (Visit trajectory) | 0 | 0.00% |
| `age` | Modality 1 (Visit trajectory) | 0 | 0.00% |
| `sex_female` | Modality 1 (Visit trajectory) | 0 | 0.00% |
| `histologic` (at `t = 0`) | Modality 2 (Enrollment panel) | 0 | 0.00% |
| `serChol` (at `t = 0`) | Modality 2 (Enrollment panel) | 28 | 8.97% |
| `ascites` (at `t = 0`) | Modality 2 (Enrollment panel) | 0 | 0.00% |
| `hepatomegaly` (at `t = 0`) | Modality 2 (Enrollment panel) | 0 | 0.00% |
| `spiders` (at `t = 0`) | Modality 2 (Enrollment panel) | 0 | 0.00% |
| `edema_score` (at `t = 0`) | Modality 2 (Enrollment panel) | 0 | 0.00% |
