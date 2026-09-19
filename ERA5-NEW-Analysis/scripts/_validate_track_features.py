"""Sanity-check track feature computation in bangladesh_nextstep_dataset_era5.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / "Final_Cyclone_Pred_Results_P-3"
sys.path.insert(0, str(OLD))

from pipeline_common import haversine_km  # noqa: E402

CSV = ROOT / "datasets" / "bangladesh_nextstep_dataset_era5.csv"


def main() -> None:
    df = pd.read_csv(CSV)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

    print("=== Internal consistency ===")
    dlat_ok = np.isclose(df["dLAT"], df["LAT_next"] - df["LAT"], atol=1e-6)
    dlon_ok = np.isclose(df["dLON"], df["LON_next"] - df["LON"], atol=1e-6)
    print(f"dLAT == LAT_next - LAT: {dlat_ok.mean() * 100:.2f}%")
    print(f"dLON == LON_next - LON: {dlon_ok.mean() * 100:.2f}%")

    # Recompute geo_speed from positions (3h step)
    dt_h = df["dt_hours_prev"].fillna(3.0).values
    dist_km = haversine_km(
        df["LAT"].values,
        df["LON"].values,
        df["LAT_next"].values,
        df["LON_next"].values,
    )
    speed_recalc = dist_km / dt_h
    speed_ok = np.isclose(df["geo_speed_kmh"], speed_recalc, rtol=0.01, atol=0.5)
    print(f"geo_speed_kmh matches haversine/dt: {speed_ok.mean() * 100:.2f}%")
    print(
        f"geo_speed_kmh range: min={df['geo_speed_kmh'].min():.1f} "
        f"med={df['geo_speed_kmh'].median():.1f} max={df['geo_speed_kmh'].max():.1f} "
        f"p99={df['geo_speed_kmh'].quantile(0.99):.1f} km/h"
    )

    bear = df["geo_bearing"]
    sin_ok = np.isclose(df["geo_bear_sin"], np.sin(np.radians(bear)), atol=1e-4)
    cos_ok = np.isclose(df["geo_bear_cos"], np.cos(np.radians(bear)), atol=1e-4)
    print(f"geo_bear_sin/cos match bearing: {sin_ok.mean() * 100:.1f}% / {cos_ok.mean() * 100:.1f}%")

    print("\n=== Lag chain (sample storms) ===")
    lag_checks = []
    for sid, g in df.groupby("SID"):
        g = g.sort_values("ISO_TIME").reset_index(drop=True)
        for i in range(1, len(g)):
            row, prev = g.iloc[i], g.iloc[i - 1]
            lag_checks.append(np.isclose(row["LAT_lag1"], prev["LAT"], atol=1e-6))
            lag_checks.append(np.isclose(row["LON_lag1"], prev["LON"], atol=1e-6))
            lag_checks.append(np.isclose(row["dLAT_lag1"], prev["dLAT"], atol=1e-6))
            lag_checks.append(np.isclose(row["dLON_lag1"], prev["dLON"], atol=1e-6))
            if i >= 2:
                prev2 = g.iloc[i - 2]
                lag_checks.append(np.isclose(row["LAT_lag2"], prev2["LAT"], atol=1e-6))
                lag_checks.append(np.isclose(row["dLAT_lag2"], prev2["dLAT"], atol=1e-6))
    print(f"Lag values match prior rows: {np.mean(lag_checks) * 100:.2f}%")

    print("\n=== Missing flags at storm start ===")
    starts = df.groupby("SID").head(1)
    print(f"Row-1 LAT_lag2==0: {(starts['LAT_lag2'] == 0).mean() * 100:.1f}%")
    print(f"Row-1 LAT_lag2_missing==1: {(starts['LAT_lag2_missing'] == 1).mean() * 100:.1f}%")
    print(f"Row-1 LAT_lag3_missing==1: {(starts['LAT_lag3_missing'] == 1).mean() * 100:.1f}%")
    row3 = df.groupby("SID").nth(2)
    print(f"Row-3 LAT_lag2_missing==0: {(row3['LAT_lag2_missing'] == 0).mean() * 100:.1f}%")

    print("\n=== QC flags (pre-filter rates in full ready file differ) ===")
    print(f"flag_irregular_dt: {df['flag_irregular_dt'].mean() * 100:.3f}%")
    print(f"flag_latnext_mismatch: {df['flag_latnext_mismatch'].mean() * 100:.3f}%")
    print(f"dt_hours_prev unique: {sorted(df['dt_hours_prev'].dropna().unique())}")

    print("\n=== Realism (BoB cyclones) ===")
    lat_ok = df["LAT"].between(5, 30)
    lon_ok = df["LON"].between(70, 102)
    print(f"LAT in 5-30N: {lat_ok.mean() * 100:.1f}%")
    print(f"LON in 70-102E: {lon_ok.mean() * 100:.1f}%")
    plausible_speed = df["geo_speed_kmh"].between(0, 80)
    print(f"Speed 0-80 km/h: {plausible_speed.mean() * 100:.1f}%")
    fast = (df["geo_speed_kmh"] > 50).sum()
    print(f"Rows with speed >50 km/h: {fast} ({fast/len(df)*100:.2f}%)")

    print("\n=== Example storm (first 3 rows, like screenshot) ===")
    sid = df["SID"].iloc[0]
    g = df[df["SID"] == sid].sort_values("ISO_TIME").head(3)
    cols = [
        "ISO_TIME", "LAT", "LON", "LAT_next", "LON_next", "dLAT", "dLON",
        "geo_speed_kmh", "geo_bearing", "LAT_lag1", "LAT_lag2",
        "LAT_lag2_missing", "flag_irregular_dt",
    ]
    print(g[cols].to_string(index=False))


if __name__ == "__main__":
    main()
