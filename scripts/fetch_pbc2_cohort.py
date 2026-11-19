#!/usr/bin/env python3
"""Fetch and validate the Mayo Clinic PBC2 longitudinal cohort (survival::pbcseq).

Source:
- Rdatasets mirror of survival::pbcseq (GPL >= 2), identical clinical trial records to JM::pbc2:
  312 randomized PBC patients, 1,945 irregular longitudinal visits spanning 0-14.3 years.

Outputs:
- data/pbc2_longitudinal.parquet (sorted by patient_id, visit_time; excludes trt & raw status)
- data/DATASETS.md (source URL, SHA-256 checksums, license provenance, codebook & summary)
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import ssl
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PARQUET_PATH = REPO_ROOT / "data" / "pbc2_longitudinal.parquet"
DEFAULT_DATASETS_MD_PATH = REPO_ROOT / "data" / "DATASETS.md"

PBCSEQ_CSV_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv"
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class SourceArtifact:
    name: str
    url: str
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssl_context() -> ssl.SSLContext:
    if importlib.util.find_spec("certifi") is not None:
        certifi_mod = importlib.import_module("certifi")
        return ssl.create_default_context(cafile=str(certifi_mod.where()))
    macos_ca = Path("/etc/ssl/cert.pem")
    if macos_ca.exists():
        return ssl.create_default_context(cafile=str(macos_ca))
    return ssl.create_default_context()


def _https_get_bytes(url: str, max_redirects: int = 5) -> bytes:
    current = url
    ctx = _ssl_context()
    for _ in range(max_redirects):
        parsed = urllib.parse.urlsplit(current)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"Only HTTPS URLs with host are permitted: {current}")
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        conn = http.client.HTTPSConnection(parsed.netloc, timeout=60, context=ctx)
        try:
            conn.request("GET", path, headers={"User-Agent": "traj-fuse/0.1"})
            resp = conn.getresponse()
            if resp.status in {301, 302, 303, 307, 308}:
                location = resp.getheader("Location")
                if not location:
                    raise RuntimeError(f"Redirect without Location from {current}")
                current = urllib.parse.urljoin(current, location)
                continue
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} fetching {current}")
            return resp.read()
        finally:
            conn.close()
    raise RuntimeError(f"Too many redirects fetching {url}")


def download_file(url: str, dest: Path) -> SourceArtifact:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(_https_get_bytes(url))
    return SourceArtifact(
        name=dest.name,
        url=url,
        sha256=sha256_file(dest),
        size_bytes=dest.stat().st_size,
    )


def build_pbc2_dataframe(raw_csv: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(raw_csv)
    if len(raw) != 1_945 or int(raw["id"].nunique()) != 312:
        raise ValueError(
            f"Unexpected pbcseq shape: rows={len(raw)}, patients={raw['id'].nunique()}"
        )

    raw = raw.sort_values(["id", "day"], kind="mergesort").reset_index(drop=True)
    if raw.duplicated(subset=["id", "day"]).any():
        raise ValueError("Duplicate (id, day) visits found in pbcseq.")
    first_days = raw.groupby("id")["day"].min()
    if not (first_days == 0).all():
        raise ValueError("Every pbcseq patient must have an enrollment visit at day == 0.")

    patient_level = raw.groupby("id", sort=False).first()
    summary = {
        "n_patients": int(len(patient_level)),
        "n_visits": int(len(raw)),
        "status_alive": int((patient_level["status"] == 0).sum()),
        "status_transplanted": int((patient_level["status"] == 1).sum()),
        "status_dead": int((patient_level["status"] == 2).sum()),
        "trt_1": int((patient_level["trt"] == 1).sum()),
        "trt_0": int((patient_level["trt"] == 0).sum()),
    }

    df = pd.DataFrame(
        {
            "patient_id": raw["id"].astype(int).map(lambda pid: f"pbc_{pid:03d}"),
            "followup_time": (raw["futime"].astype("float64") / DAYS_PER_YEAR),
            "event": raw["status"].isin([1, 2]).astype("int64"),
            "visit_time": (raw["day"].astype("float64") / DAYS_PER_YEAR),
            "serBilir": pd.to_numeric(raw["bili"], errors="raise").astype("float64"),
            "albumin": pd.to_numeric(raw["albumin"], errors="raise").astype("float64"),
            "alkaline": pd.to_numeric(raw["alk.phos"], errors="coerce").astype("float64"),
            "SGOT": pd.to_numeric(raw["ast"], errors="raise").astype("float64"),
            "platelets": pd.to_numeric(raw["platelet"], errors="coerce").astype("float64"),
            "prothrombin": pd.to_numeric(raw["protime"], errors="raise").astype("float64"),
            "age": pd.to_numeric(raw["age"], errors="raise").astype("float64"),
            "sex_female": np.where(raw["sex"] == "f", 1.0, 0.0).astype("float64"),
            "histologic": pd.to_numeric(raw["stage"], errors="coerce").astype("float64"),
            "serChol": pd.to_numeric(raw["chol"], errors="coerce").astype("float64"),
            "ascites": pd.to_numeric(raw["ascites"], errors="coerce").astype("float64"),
            "hepatomegaly": pd.to_numeric(raw["hepato"], errors="coerce").astype("float64"),
            "spiders": pd.to_numeric(raw["spiders"], errors="coerce").astype("float64"),
            "edema_score": pd.to_numeric(raw["edema"], errors="coerce").astype("float64"),
        }
    )
    return df, summary


def _render_datasets_md(
    df: pd.DataFrame,
    summary: dict[str, int],
    artifact: SourceArtifact,
    parquet_sha256: str,
    parquet_size: int,
) -> str:
    t0 = df[df["visit_time"] == 0.0]
    gt2 = df.groupby("patient_id")["followup_time"].first() > 2.0
    n_gt2 = int(gt2.sum())
    ev_gt2 = int(df.groupby("patient_id")["event"].first()[gt2].sum())
    visits_per_patient = df.groupby("patient_id").size()

    traj_cols = [
        "serBilir",
        "albumin",
        "alkaline",
        "SGOT",
        "platelets",
        "prothrombin",
        "age",
        "sex_female",
    ]
    panel_cols = [
        "histologic",
        "serChol",
        "ascites",
        "hepatomegaly",
        "spiders",
        "edema_score",
    ]

    traj_lines = [
        f"| `{col}` | Modality 1 (Visit trajectory) | {int(df[col].isna().sum())} "
        f"| {df[col].isna().mean() * 100:.2f}% |"
        for col in traj_cols
    ]
    panel_lines = [
        f"| `{col}` (at `t = 0`) | Modality 2 (Enrollment panel) | {int(t0[col].isna().sum())} "
        f"| {t0[col].isna().mean() * 100:.2f}% |"
        for col in panel_cols
    ]

    return "\n".join(
        [
            "# Mayo Clinic Primary Biliary Cirrhosis Longitudinal Cohort (`PBC2` / `pbcseq`)",
            "",
            "## Provenance & License",
            "",
            (
                f"- **Upstream source:** `{artifact.url}` "
                "(`survival::pbcseq`, equivalent to `JM::pbc2`)"
            ),
            f"- **Raw CSV SHA-256:** `{artifact.sha256}` ({artifact.size_bytes:,} bytes)",
            "- **License:** GPL (>= 2) per CRAN `survival` / `JM` packages",
            f"- **Output file:** `data/pbc2_longitudinal.parquet` ({parquet_size:,} bytes)",
            f"- **Output SHA-256:** `{parquet_sha256}`",
            "",
            "## Cohort Summary",
            "",
            (
                f"- **Patients:** `{summary['n_patients']}` "
                "(`100%` have an enrollment visit at `t = 0.0`)"
            ),
            (
                f"- **Irregular clinic visits:** `{summary['n_visits']:,}` "
                f"(min `{int(visits_per_patient.min())}`, median "
                f"`{visits_per_patient.median():.1f}`, max `{int(visits_per_patient.max())}`)"
            ),
            (
                f"- **Follow-up range (`followup_time`):** `{df['followup_time'].min():.3f}` "
                f"to `{df['followup_time'].max():.3f}` years"
            ),
            (
                "- **Composite transplant-free survival endpoint (`event = 1`):** "
                f"`{summary['status_dead'] + summary['status_transplanted']}` "
                f"(`{summary['status_dead']}` dead + `{summary['status_transplanted']}` "
                f"transplanted; `{summary['status_alive']}` alive)"
            ),
            (
                f"- **Landmark risk set (`followup_time > 2.0` yr):** `{n_gt2}` patients "
                f"(`{ev_gt2}` composite events, `{ev_gt2 / n_gt2 * 100:.1f}%`)"
            ),
            (
                "- **Excluded columns:** `trt` (50/50 randomized drug arm: "
                f"`{summary['trt_1']}` D-penicillamine, `{summary['trt_0']}` placebo) "
                "and raw 3-level `status` are excluded from the parquet."
            ),
            "",
            "## Modality Firewall & Missingness",
            "",
            (
                "- Modality 1 (`D_traj = 8`) keeps demographics (`age`, `sex_female`) "
                "with routine blood labs so missing-panel ablations isolate specialty staging."
            ),
            (
                "- Modality 2 (`D_mol = 6`) is extracted strictly from enrollment `t = 0.0` "
                "with `skipna=False` (renaming `serChol -> baseline_serChol` in `data.py`)."
            ),
            "",
            "| Feature | Modality | Missing Count | Missing Rate |",
            "|---|---|---:|---:|",
            *traj_lines,
            *panel_lines,
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and build Mayo PBC2 parquet cohort.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-parquet", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_DATASETS_MD_PATH)
    args = parser.parse_args(argv)

    raw_csv = args.raw_dir / "pbcseq.csv"
    artifact = download_file(PBCSEQ_CSV_URL, raw_csv)
    df, summary = build_pbc2_dataframe(raw_csv)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)

    parquet_sha = sha256_file(args.out_parquet)
    parquet_size = args.out_parquet.stat().st_size
    md_text = _render_datasets_md(
        df=df,
        summary=summary,
        artifact=artifact,
        parquet_sha256=parquet_sha,
        parquet_size=parquet_size,
    )
    args.out_md.write_text(md_text, encoding="utf-8")

    gt2 = df.groupby("patient_id")["followup_time"].first() > 2.0
    ev_gt2 = int(df.groupby("patient_id")["event"].first()[gt2].sum())
    t0 = df[df["visit_time"] == 0.0]
    chol_miss = int(t0["serChol"].isna().sum())
    print(
        f"Built {args.out_parquet} ({parquet_size:,} bytes, sha256={parquet_sha}): "
        f"patients={summary['n_patients']}, visits={summary['n_visits']}, "
        f"T>2.0y={int(gt2.sum())} (events={ev_gt2}), t0_serChol_missing={chol_miss}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
