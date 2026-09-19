"""
Step 0 — IBTrACS / research-ready quality flags by date window.

No ERA5 required. Writes checkpointed CSVs so a power cut is cheap to resume.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / "Final_Cyclone_Pred_Results_P-3" / "datasets"
READY = OLD / "bangladesh_nextstep_dataset_research_ready.csv"
IB = OLD / "ibtracs.NI.list.v04r01.csv"

OUT = ROOT / "results"
DOCS = ROOT / "docs"
PROG = OUT / "ibtracs_quality_progress.json"

WINDOWS = {
    "full_era5": (1940, 2024),
    "satellite": (1979, 2024),
    "modern": (1990, 2024),
    "pre_satellite": (1940, 1978),
    "pre_era5": (1842, 1939),
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["flag_latnext_mismatch"] == 0].copy()
    out = out[out["flag_irregular_dt"] == 0]
    out = out.dropna(subset=["dLAT", "dLON", "LAT_next", "LON_next"])
    return out.reset_index(drop=True)


def genesis_subbasin() -> pd.DataFrame:
    raw = pd.read_csv(
        IB,
        skiprows=[1],
        usecols=["SID", "SUBBASIN", "LAT", "ISO_TIME", "IFLAG", "TRACK_TYPE", "NATURE", "WMO_WIND", "USA_WIND", "NEWDELHI_WIND"],
        low_memory=False,
    )
    raw["LAT"] = pd.to_numeric(raw["LAT"], errors="coerce")
    raw["ISO_TIME"] = pd.to_datetime(raw["ISO_TIME"], errors="coerce")
    raw = raw.dropna(subset=["LAT", "ISO_TIME"]).sort_values(["SID", "ISO_TIME"])
    gen = raw.groupby("SID", as_index=False).first()
    gen["SID"] = gen["SID"].astype(str)
    return gen, raw


def storm_flags(ready_all: pd.DataFrame, ready_qc: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    ready = ready_all.copy()
    ready["SID"] = ready["SID"].astype(str)
    ready["ISO_TIME"] = pd.to_datetime(ready["ISO_TIME"], errors="coerce")
    ready["SEASON"] = pd.to_numeric(ready["SEASON"], errors="coerce")
    ready_qc = ready_qc.copy()
    ready_qc["SID"] = ready_qc["SID"].astype(str)

    miss_wind = (
        ready.groupby("SID")["NEWDELHI_WIND_missing"].mean()
        if "NEWDELHI_WIND_missing" in ready.columns
        else pd.Series(dtype=float)
    )
    irreg = ready.groupby("SID")["flag_irregular_dt"].mean()
    mismatch = ready.groupby("SID")["flag_latnext_mismatch"].mean()
    n_pts = ready_qc.groupby("SID").size()

    raw = raw.copy()
    raw["SID"] = raw["SID"].astype(str)
    for col in ["WMO_WIND", "USA_WIND", "NEWDELHI_WIND"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["IFLAG"] = raw["IFLAG"].astype(str)

    def _iflag_interp(s: pd.Series) -> float:
        # IBTrACS IFLAG: interpolated positions often marked; treat non-empty unusual flags
        return float((~s.str.contains("P", na=False) & s.str.strip().ne("") & s.str.strip().ne("nan")).mean())

    g = raw.groupby("SID")
    flags = pd.DataFrame(
        {
            "n_ibtracs_points": g.size(),
            "frac_missing_wmo_wind": g["WMO_WIND"].apply(lambda s: s.isna().mean()),
            "frac_missing_usa_wind": g["USA_WIND"].apply(lambda s: s.isna().mean()),
            "frac_missing_newdelhi_wind": g["NEWDELHI_WIND"].apply(lambda s: s.isna().mean()),
            "frac_track_not_main": g["TRACK_TYPE"].apply(lambda s: (s.astype(str) != "main").mean()),
            "frac_nature_nr": g["NATURE"].apply(lambda s: (s.astype(str) == "NR").mean()),
        }
    )
    flags["n_ready_points"] = n_pts
    flags["frac_newdelhi_wind_missing_ready"] = miss_wind
    flags["frac_irregular_dt"] = irreg
    flags["frac_latnext_mismatch"] = mismatch
    flags["genesis_year"] = ready_qc.groupby("SID")["SEASON"].min()
    flags["SID"] = flags.index.astype(str)
    flags = flags[flags["n_ready_points"].fillna(0) > 0]
    return flags.reset_index(drop=True)


def window_table(flags: pd.DataFrame, gen: pd.DataFrame) -> pd.DataFrame:
    gen = gen.rename(columns={"SUBBASIN": "genesis_subbasin"})
    m = flags.merge(gen[["SID", "genesis_subbasin"]], on="SID", how="left")
    rows = []
    for name, (y0, y1) in WINDOWS.items():
        sub = m[(m["genesis_year"] >= y0) & (m["genesis_year"] <= y1)]
        if sub.empty:
            continue

        def frac_affected(col: str, thresh: float = 0.5) -> float:
            if col not in sub.columns:
                return np.nan
            return float((sub[col].fillna(0) >= thresh).mean())

        rows.append(
            {
                "window": name,
                "year_start": y0,
                "year_end": y1,
                "n_storms": int(sub["SID"].nunique()),
                "n_track_points_ready": int(sub["n_ready_points"].fillna(0).sum()),
                "frac_storms_bob_genesis": float((sub["genesis_subbasin"] == "BB").mean()),
                "frac_storms_majority_missing_wmo_wind": frac_affected("frac_missing_wmo_wind"),
                "frac_storms_majority_missing_usa_wind": frac_affected("frac_missing_usa_wind"),
                "frac_storms_majority_missing_newdelhi_wind": frac_affected("frac_missing_newdelhi_wind"),
                "mean_frac_missing_wmo_wind": float(sub["frac_missing_wmo_wind"].mean()),
                "mean_frac_missing_newdelhi_wind_ready": float(
                    sub["frac_newdelhi_wind_missing_ready"].mean()
                ) if "frac_newdelhi_wind_missing_ready" in sub else np.nan,
                "mean_frac_irregular_dt": float(sub["frac_irregular_dt"].fillna(0).mean()),
                "mean_frac_latnext_mismatch": float(sub["frac_latnext_mismatch"].fillna(0).mean()),
                "mean_frac_track_not_main": float(sub["frac_track_not_main"].mean()),
                "mean_frac_nature_nr": float(sub["frac_nature_nr"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_doc(win: pd.DataFrame) -> None:
    lines = [
        "# IBTrACS quality by era (before ERA5 join)",
        "",
        f"Generated {_utc()}. Diagnostic only — no modeling.",
        "",
        "Windows use **genesis season** on the research-ready file after the same QC as the old pipeline.",
        "",
        "## Counts",
        "",
        win.to_string(index=False),
        "",
        "## How to read this",
        "",
        "- **pre_era5 (1842–1939)** cannot get ERA5. Large missing-wind fractions are expected.",
        "- **pre_satellite (1940–1978)** has ERA5 but sparse ocean obs + weaker IBTrACS.",
        "- **satellite (1979–2024)** is the usual climate-quality ERA5/IBTrACS era.",
        "- **modern (1990–2024)** matches a paper-like scope; smallest sample.",
        "",
        "IBTrACS fields used: `WMO_WIND`, `USA_WIND`, `NEWDELHI_WIND`, `TRACK_TYPE`, `NATURE`, `IFLAG` (raw),",
        "plus research-ready `NEWDELHI_WIND_missing`, `flag_irregular_dt`, `flag_latnext_mismatch`.",
        "",
        "A storm is “majority missing wind” if ≥50% of its IBTrACS points lack that wind field.",
        "",
        "Date-window **model** ablation still waits on `datasets/*_era5.csv`.",
        "",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "ibtracs_quality_by_era.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROG.write_text(json.dumps({"status": "running", "updated_at": _utc()}, indent=2), encoding="utf-8")

    ready_raw = pd.read_csv(READY)
    ready_qc = apply_qc(ready_raw)
    gen, raw = genesis_subbasin()
    flags = storm_flags(ready_raw, ready_qc, raw)
    flags_path = OUT / "ibtracs_storm_quality_flags.csv"
    flags.to_csv(flags_path, index=False)

    win = window_table(flags, gen)
    win_path = OUT / "ibtracs_quality_by_window.csv"
    win.to_csv(win_path, index=False)
    write_doc(win)

    PROG.write_text(
        json.dumps(
            {
                "status": "complete",
                "updated_at": _utc(),
                "flags": str(flags_path),
                "windows": str(win_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(win.to_string(index=False))
    print(f"Wrote {win_path}")
    print(f"Wrote {DOCS / 'ibtracs_quality_by_era.md'}")


if __name__ == "__main__":
    main()
