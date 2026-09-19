"""
Position vs wind/intensity data quality by era (IBTrACS + QC-passing kinematics).

Separates track-position trust from wind-observation sparsity. ERA5 along-track
variability is reported separately as atmospheric diversity, not IBTrACS quality.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / "Final_Cyclone_Pred_Results_P-3" / "datasets"
READY = OLD / "bangladesh_nextstep_dataset_research_ready.csv"
IB = OLD / "ibtracs.NI.list.v04r01.csv"
ERA5 = ROOT / "datasets" / "tracks_era5.csv"
OUT = ROOT / "results" / "position_vs_wind_quality_by_era.csv"
DOC = ROOT / "docs" / "position_vs_wind_quality.md"

ERAS = [
    ("pre_satellite", 1940, 1978),
    ("bridge_satellite", 1979, 1989),
    ("modern", 1990, 2024),
]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iflag_position_interp(s: pd.Series) -> pd.Series:
    """IBTrACS IFLAG: 'I' marks interpolated fix; blank/O often means no agency fix."""
    t = s.fillna("").astype(str)
    return t.str.contains("I", regex=False)


def _iflag_no_primary_position(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str)
    return ~t.str.contains("P", regex=False)


def load_ibtracs_bob() -> pd.DataFrame:
    raw = pd.read_csv(
        IB,
        skiprows=[1],
        usecols=["SID", "SEASON", "ISO_TIME", "LAT", "LON", "TRACK_TYPE", "IFLAG",
                 "WMO_WIND", "USA_WIND", "NEWDELHI_WIND", "SUBBASIN"],
        low_memory=False,
    )
    raw["ISO_TIME"] = pd.to_datetime(raw["ISO_TIME"], errors="coerce")
    raw["SEASON"] = pd.to_numeric(raw["SEASON"], errors="coerce")
    raw["LAT"] = pd.to_numeric(raw["LAT"], errors="coerce")
    raw["LON"] = pd.to_numeric(raw["LON"], errors="coerce")
    for c in ("WMO_WIND", "USA_WIND", "NEWDELHI_WIND"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["ISO_TIME", "LAT", "LON", "SEASON"])
    raw["SID"] = raw["SID"].astype(str)
    raw = raw[raw["TRACK_TYPE"].astype(str).str.strip().isin(["main", "spur-merge"])]
    # BoB-relevant: genesis in BB sub-basin (same logic as scope audit)
    gen = raw.sort_values("ISO_TIME").groupby("SID", as_index=False).first()
    bb = set(gen.loc[gen["SUBBASIN"].astype(str) == "BB", "SID"])
    return raw[raw["SID"].isin(bb)].copy()


def ibtracs_metrics(raw: pd.DataFrame, y0: int, y1: int) -> dict:
    sub = raw[(raw["SEASON"] >= y0) & (raw["SEASON"] <= y1)]
    if sub.empty:
        return {}
    g = sub.groupby("SID")
    storms = g.size().shape[0]
    return {
        "n_track_points": len(sub),
        "n_storms": storms,
        "frac_iflag_interpolated_position": float(_iflag_position_interp(sub["IFLAG"]).mean()),
        "frac_iflag_no_primary_position": float(_iflag_no_primary_position(sub["IFLAG"]).mean()),
        "frac_track_not_main": float((sub["TRACK_TYPE"].astype(str) != "main").mean()),
        "frac_missing_wmo_wind": float(sub["WMO_WIND"].isna().mean()),
        "frac_missing_usa_wind": float(sub["USA_WIND"].isna().mean()),
        "frac_missing_newdelhi_wind": float(sub["NEWDELHI_WIND"].isna().mean()),
        "storms_majority_missing_wmo_wind": float((g["WMO_WIND"].apply(lambda s: s.isna().mean()) >= 0.5).mean()),
        "storms_majority_missing_newdelhi_wind": float((g["NEWDELHI_WIND"].apply(lambda s: s.isna().mean()) >= 0.5).mean()),
    }


def ready_pre_qc_metrics(y0: int, y1: int) -> dict:
    ready = pd.read_csv(READY)
    ready["ISO_TIME"] = pd.to_datetime(ready["ISO_TIME"])
    ready["SEASON"] = pd.to_numeric(ready["SEASON"], errors="coerce")
    sub = ready[(ready["SEASON"] >= y0) & (ready["SEASON"] <= y1)]
    if sub.empty:
        return {}
    return {
        "ready_frac_irregular_dt": float(sub["flag_irregular_dt"].mean()),
        "ready_frac_latnext_mismatch": float(sub["flag_latnext_mismatch"].mean()),
        "ready_frac_newdelhi_wind_missing": float(sub["NEWDELHI_WIND_missing"].mean()),
        "ready_n_points": len(sub),
        "ready_n_storms": sub["SID"].nunique(),
    }


def ready_qc_kinematics(y0: int, y1: int) -> dict:
    ready = pd.read_csv(READY)
    ready["ISO_TIME"] = pd.to_datetime(ready["ISO_TIME"])
    ready["SEASON"] = pd.to_numeric(ready["SEASON"], errors="coerce")
    sub = ready[
        (ready["SEASON"] >= y0) & (ready["SEASON"] <= y1)
        & (ready["flag_irregular_dt"] == 0)
        & (ready["flag_latnext_mismatch"] == 0)
    ].dropna(subset=["dLAT", "dLON"])
    if sub.empty:
        return {}
    sub = sub.sort_values(["SID", "ISO_TIME"])
    bear_chg = sub.groupby("SID")["geo_bearing"].diff().abs()
    return {
        "qc_std_dlat_deg": float(sub["dLAT"].std()),
        "qc_std_dlon_deg": float(sub["dLON"].std()),
        "qc_median_geo_speed_kmh": float(sub["geo_speed_kmh"].median()),
        "qc_mean_abs_bearing_change_deg": float(bear_chg.mean()),
        "qc_n_points": len(sub),
    }


def era5_atmos_diversity(y0: int, y1: int) -> dict:
    """ERA5 is homogeneous in time; spread here is meteorology, not IBTrACS noise."""
    ready = pd.read_csv(READY, usecols=["SID", "ISO_TIME", "SEASON"])
    era5 = pd.read_csv(ERA5)
    ready["ISO_TIME"] = pd.to_datetime(ready["ISO_TIME"])
    era5["ISO_TIME"] = pd.to_datetime(era5["ISO_TIME"])
    ready["SEASON"] = pd.to_numeric(ready["SEASON"], errors="coerce")
    m = ready.merge(era5, on=["SID", "ISO_TIME"], how="inner")
    sub = m[(m["SEASON"] >= y0) & (m["SEASON"] <= y1)]
    if sub.empty:
        return {}
    u850 = sub["u850"].astype(float)
    v850 = sub["v850"].astype(float)
    wspd = np.sqrt(u850 ** 2 + v850 ** 2)
    return {
        "era5_std_u850": float(u850.std()),
        "era5_std_v850": float(v850.std()),
        "era5_std_wspd850": float(wspd.std()),
        "era5_n_points": len(sub),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = load_ibtracs_bob()
    rows = []
    for era, y0, y1 in ERAS:
        row = {"era": era, "year_start": y0, "year_end": y1}
        row.update(ibtracs_metrics(raw, y0, y1))
        row.update(ready_pre_qc_metrics(y0, y1))
        row.update(ready_qc_kinematics(y0, y1))
        row.update(era5_atmos_diversity(y0, y1))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    pre = df[df["era"] == "pre_satellite"].iloc[0]
    mod = df[df["era"] == "modern"].iloc[0]

    lines = [
        "# Position vs wind quality by era",
        "",
        f"Generated {_utc()}. BoB-genesis storms (SUBBASIN=BB).",
        "",
        "## Summary",
        "",
        "The pre-1979 **wind/intensity** gap is large; **position** proxies are much closer after QC.",
        "",
        "| Metric | Pre-satellite (1940–1978) | Modern (1990–2024) |",
        "|--------|-------------------------:|-------------------:|",
        f"| IBTrACS WMO wind missing (point) | {100*pre['frac_missing_wmo_wind']:.1f}% | {100*mod['frac_missing_wmo_wind']:.1f}% |",
        f"| IBTrACS NEWDELHI wind missing (point) | {100*pre['frac_missing_newdelhi_wind']:.1f}% | {100*mod['frac_missing_newdelhi_wind']:.1f}% |",
        f"| IFLAG interpolated position (point) | {100*pre['frac_iflag_interpolated_position']:.1f}% | {100*mod['frac_iflag_interpolated_position']:.1f}% |",
        f"| Research-ready irregular dt (pre-QC) | {100*pre['ready_frac_irregular_dt']:.2f}% | {100*mod['ready_frac_irregular_dt']:.2f}% |",
        f"| Research-ready lat/lon mismatch (pre-QC) | {100*pre['ready_frac_latnext_mismatch']:.2f}% | {100*mod['ready_frac_latnext_mismatch']:.2f}% |",
        f"| QC-passing std(dLAT) deg | {pre['qc_std_dlat_deg']:.3f} | {mod['qc_std_dlat_deg']:.3f} |",
        f"| QC-passing std(dLON) deg | {pre['qc_std_dlon_deg']:.3f} | {mod['qc_std_dlon_deg']:.3f} |",
        "",
        "## Interpretation",
        "",
        "- **Wind/intensity observations** are far sparser pre-1979 (expected; already in `ibtracs_quality_by_window.csv`).",
        "- **Position/timing proxies** (IFLAG interpolation, irregular dt, lat/lon mismatch, QC kinematic spread) are **much closer** between eras.",
        "- **ERA5 along-track wind spread** differs by era because of **meteorology**, not reanalysis quality (ERA5 is homogeneous 1940–2024).",
        "- Extra pre-1979 storms mainly add **trajectory + ERA5 environment** examples for long-lead track prediction,",
        "  even when IBTrACS wind reports are missing.",
        "",
        f"Full table: `{OUT.name}`",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nWrote {OUT}\nWrote {DOC}")


if __name__ == "__main__":
    main()
