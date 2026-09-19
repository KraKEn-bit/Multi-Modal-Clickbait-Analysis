"""
Add sst_missing and fill missing SST with nearest valid ocean grid cell
at the same ERA5 timestep (physically reasonable for land/near-coast points).

Reads LAT/LON from the research-ready track file; writes tracks_era5.csv in place.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD_READY = ROOT.parent / "Final_Cyclone_Pred_Results_P-3" / "datasets" / "bangladesh_nextstep_dataset_research_ready.csv"
IN_CSV = ROOT / "datasets" / "tracks_era5.csv"
RAW = ROOT / "datasets" / "era5_raw"
PROG = ROOT / "results" / "prepare_tracks_era5_progress.json"

# Reuse ERA5 file discovery from join
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from era5_join import _open_kind  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sst_var(ds) -> str:
    for name in ("sst", "sea_surface_temperature"):
        if name in ds:
            return name
    raise KeyError(f"SST variable not in {list(ds.data_vars)}")


def nearest_ocean_sst(lats: np.ndarray, lons: np.ndarray, da) -> np.ndarray:
    """Nearest finite SST on the ERA5 grid for each (lat, lon)."""
    from scipy.spatial import cKDTree

    sst = np.asarray(da.values, dtype=float)
    lat1d = np.asarray(da.latitude.values, dtype=float)
    lon1d = np.asarray(da.longitude.values, dtype=float)
    llat, llon = np.meshgrid(lat1d, lon1d, indexing="ij")
    valid = np.isfinite(sst)
    if not valid.any():
        return np.full(len(lats), np.nan)

    vlat = llat[valid]
    vlon = llon[valid]
    vsst = sst[valid]
    # Small-area planar approx for KD-tree distances (BoB box).
    coords = np.column_stack([vlat, vlon * np.cos(np.radians(vlat))])
    tree = cKDTree(coords)
    qlon = lons * np.cos(np.radians(lats))
    _, idx = tree.query(np.column_stack([lats, qlon]), k=1)
    return vsst[idx]


def month_climatology_ocean(da) -> float:
    sst = np.asarray(da.values, dtype=float)
    valid = sst[np.isfinite(sst)]
    return float(np.median(valid)) if valid.size else np.nan


def fill_year(year: int, sub: pd.DataFrame, out_sst: dict[tuple[str, str], float]) -> int:
    sl = _open_kind(year, "sl")
    if sl is None:
        return 0
    try:
        sst_name = _sst_var(sl)
        lat = "latitude" if "latitude" in sl.coords else "lat"
        lon = "longitude" if "longitude" in sl.coords else "lon"
        if lat != "latitude":
            sl = sl.rename({lat: "latitude", lon: "longitude"})
        tname = "time" if "time" in sl.coords else "valid_time"
        if tname != "time":
            sl = sl.rename({tname: "time"})

        filled = 0
        for ts, grp in sub.groupby("ISO_TIME"):
            try:
                field = sl[sst_name].sel(time=pd.Timestamp(ts), method="nearest")
            except Exception:
                continue
            vals = nearest_ocean_sst(
                grp["LAT"].values.astype(float),
                grp["LON"].values.astype(float),
                field,
            )
            still_bad = ~np.isfinite(vals)
            if still_bad.any():
                clim = month_climatology_ocean(field)
                if np.isfinite(clim):
                    vals = np.where(still_bad, clim, vals)
            for i, (_, row) in enumerate(grp.iterrows()):
                if np.isfinite(vals[i]):
                    out_sst[(str(row["SID"]), str(row["ISO_TIME"]))] = float(vals[i])
                    filled += 1
        return filled
    finally:
        sl.close()


def main() -> None:
    PROG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

    ready = pd.read_csv(OLD_READY, usecols=["SID", "ISO_TIME", "LAT", "LON", "DIST2LAND"])
    ready["ISO_TIME"] = pd.to_datetime(ready["ISO_TIME"])
    df = df.merge(ready, on=["SID", "ISO_TIME"], how="left")

    df["sst_missing"] = df["sst"].isna().astype(int)
    n_miss = int(df["sst_missing"].sum())
    print(f"rows={len(df)}  sst_missing={n_miss} ({100*n_miss/len(df):.1f}%)", flush=True)

    out_map: dict[tuple[str, str], float] = {}
    miss = df[df["sst_missing"] == 1].copy()
    miss["year"] = miss["ISO_TIME"].dt.year

    for year in sorted(miss["year"].unique()):
        sub = miss[miss["year"] == year]
        n = fill_year(int(year), sub, out_map)
        print(f"  filled {year}: {n}/{len(sub)}", flush=True)
        PROG.write_text(
            json.dumps({"status": "running", "year": int(year), "filled_so_far": len(out_map), "updated_at": _utc()}, indent=2),
            encoding="utf-8",
        )

    for idx, row in df[df["sst_missing"] == 1].iterrows():
        key = (str(row["SID"]), str(row["ISO_TIME"]))
        if key in out_map:
            df.at[idx, "sst"] = out_map[key]

    still = int(df["sst"].isna().sum())
    print(f"remaining NaN sst after fill: {still}", flush=True)

    drop_cols = [c for c in ("LAT", "LON", "DIST2LAND", "year") if c in df.columns]
    out = df.drop(columns=drop_cols, errors="ignore")
    tmp = IN_CSV.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(IN_CSV)

    PROG.write_text(
        json.dumps(
            {
                "status": "complete",
                "updated_at": _utc(),
                "sst_missing_rows": n_miss,
                "filled": len(out_map),
                "sst_nan_remaining": still,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {IN_CSV} with sst_missing column", flush=True)


if __name__ == "__main__":
    main()
