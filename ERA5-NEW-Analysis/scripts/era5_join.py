"""
Join ERA5 year files onto research-ready track points (1940+).

Resumes by skipping rows already in the output CSV.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD_READY = ROOT.parent / "Final_Cyclone_Pred_Results_P-3" / "datasets" / "bangladesh_nextstep_dataset_research_ready.csv"
RAW = ROOT / "datasets" / "era5_raw"
OUT_CSV = ROOT / "datasets" / "tracks_era5.csv"
PROG = ROOT / "results" / "era5_join_progress.json"

ERA5_COLS = [
    "u850", "v850", "u500", "v500", "u200", "v200",
    "msl", "sst",
    "steer_u", "steer_v", "shear_u", "shear_v", "shear_mag",
]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def interp_field(ds, varname, lats, lons, times):
    import xarray as xr

    da = ds[varname]
    # nearest in space/time — ERA5 3-hourly
    out = da.sel(
        latitude=xr.DataArray(lats, dims="p"),
        longitude=xr.DataArray(lons, dims="p"),
        time=xr.DataArray(pd.to_datetime(times), dims="p"),
        method="nearest",
    )
    return np.asarray(out.values)


def _open_kind(year: int, kind: str):
    """A year is stored as some mix of wide batches, whole months and single days."""
    import xarray as xr

    year_path = RAW / f"era5_{kind}_{year}.nc"
    if year_path.exists():
        return xr.open_dataset(year_path)

    batch_files = sorted(RAW.glob(f"era5_{kind}_{year}_b*.nc"))
    covered: set[str] = set()
    for batch in batch_files:
        lo, hi = batch.stem.split("_b", 1)[1].split("_")
        covered.update(f"{m:02d}" for m in range(int(lo), int(hi) + 1))

    month_files = [
        p for p in sorted(RAW.glob(f"era5_{kind}_{year}_[0-9][0-9].nc"))
        if p.stem.rsplit("_", 1)[1] not in covered
    ]
    day_files = [
        p for p in sorted(RAW.glob(f"era5_{kind}_{year}_[0-9][0-9]_[0-9][0-9].nc"))
        if p.stem.split("_")[3] not in covered
        and not (RAW / f"era5_{kind}_{year}_{p.stem.split('_')[3]}.nc").exists()
    ]

    present = batch_files + month_files + day_files
    if not present:
        return None
    return xr.open_mfdataset(
        [str(p) for p in present], combine="by_coords", coords="minimal", compat="override"
    )


def process_year(year: int, tracks: pd.DataFrame) -> pd.DataFrame:
    pl = _open_kind(year, "pl")
    sl = _open_kind(year, "sl")
    if pl is None or sl is None:
        return pd.DataFrame()

    sub = tracks[tracks["year"] == year].copy()
    if sub.empty:
        pl.close()
        sl.close()
        return pd.DataFrame()
    # Standard ERA5 names vary (short vs long). Try common ones.
    def pick(ds, *names):
        for n in names:
            if n in ds:
                return n
        raise KeyError(f"none of {names} in {list(ds.data_vars)}")

    lat = "latitude" if "latitude" in pl.coords else "lat"
    lon = "longitude" if "longitude" in pl.coords else "lon"
    # rename for sel
    if lat != "latitude":
        pl = pl.rename({lat: "latitude"})
        sl = sl.rename({lat: "latitude"})
    if lon != "longitude":
        pl = pl.rename({lon: "longitude"})
        sl = sl.rename({lon: "longitude"})

    tname = "time" if "time" in pl.coords else "valid_time"
    if tname != "time":
        pl = pl.rename({tname: "time"})
        sl = sl.rename({tname: "time"})

    lev_name = "level" if "level" in pl.coords else "pressure_level"
    lats = sub["LAT"].values
    lons = ((sub["LON"].values + 180) % 360) - 180
    times = sub["ISO_TIME"].values

    rows = {"SID": sub["SID"].values, "ISO_TIME": sub["ISO_TIME"].values}

    u_name = pick(pl, "u", "u_component_of_wind")
    v_name = pick(pl, "v", "v_component_of_wind")
    for lev, tag in [(850, "850"), (500, "500"), (200, "200")]:
        u = pl[u_name].sel({lev_name: lev}, method="nearest")
        v = pl[v_name].sel({lev_name: lev}, method="nearest")
        ds_u = u.to_dataset(name="u")
        ds_v = v.to_dataset(name="v")
        rows[f"u{tag}"] = interp_field(ds_u, "u", lats, lons, times)
        rows[f"v{tag}"] = interp_field(ds_v, "v", lats, lons, times)

    msl_name = pick(sl, "msl", "mean_sea_level_pressure")
    sst_name = pick(sl, "sst", "sea_surface_temperature")
    rows["msl"] = interp_field(sl, msl_name, lats, lons, times)
    rows["sst"] = interp_field(sl, sst_name, lats, lons, times)

    pl.close()
    sl.close()

    out = pd.DataFrame(rows)
    rows_u = out
    rows_u["steer_u"] = (rows_u["u850"] + rows_u["u500"] + rows_u["u200"]) / 3.0
    rows_u["steer_v"] = (rows_u["v850"] + rows_u["v500"] + rows_u["v200"]) / 3.0
    rows_u["shear_u"] = rows_u["u200"] - rows_u["u850"]
    rows_u["shear_v"] = rows_u["v200"] - rows_u["v850"]
    rows_u["shear_mag"] = np.sqrt(rows_u["shear_u"] ** 2 + rows_u["shear_v"] ** 2)
    return rows_u


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    PROG.parent.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(OLD_READY, usecols=["SID", "ISO_TIME", "LAT", "LON", "SEASON"])
    tracks["ISO_TIME"] = pd.to_datetime(tracks["ISO_TIME"], errors="coerce")
    tracks["year"] = tracks["ISO_TIME"].dt.year
    tracks = tracks[(tracks["year"] >= 1940) & (tracks["year"] <= 2024)].copy()

    done_keys = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV, usecols=["SID", "ISO_TIME"])
        prev["ISO_TIME"] = pd.to_datetime(prev["ISO_TIME"])
        done_keys = set(zip(prev["SID"].astype(str), prev["ISO_TIME"].astype(str)))

    years = sorted(int(y) for y in tracks["year"].dropna().unique())
    chunks = []
    for year in years:
        PROG.write_text(
            json.dumps({"status": "running", "year": year, "updated_at": _utc()}, indent=2),
            encoding="utf-8",
        )
        sub = tracks[tracks["year"] == year]
        pending = sub[
            ~sub.apply(lambda r: (str(r["SID"]), str(r["ISO_TIME"])) in done_keys, axis=1)
        ]
        if pending.empty:
            print(f"Join skip {year} (already in CSV)", flush=True)
            continue
        part = process_year(year, pending)
        if part.empty:
            print(f"Join skip {year} (ERA5 files missing)", flush=True)
            continue
        chunks.append(part)
        print(f"Joined {year}: {len(part)} rows", flush=True)

    if chunks:
        new = pd.concat(chunks, ignore_index=True)
        if OUT_CSV.exists():
            old = pd.read_csv(OUT_CSV)
            old["ISO_TIME"] = pd.to_datetime(old["ISO_TIME"])
            new["ISO_TIME"] = pd.to_datetime(new["ISO_TIME"])
            all_df = pd.concat([old, new], ignore_index=True)
            all_df = all_df.drop_duplicates(subset=["SID", "ISO_TIME"], keep="last")
        else:
            all_df = new
        tmp = OUT_CSV.with_suffix(".csv.tmp")
        all_df.to_csv(tmp, index=False)
        tmp.replace(OUT_CSV)

    missing = [
        y
        for y in years
        if not ((RAW / f"era5_pl_{y}.nc").exists() and (RAW / f"era5_sl_{y}.nc").exists())
    ]
    status = "complete" if not missing else "partial"
    PROG.write_text(
        json.dumps(
            {
                "status": status,
                "updated_at": _utc(),
                "out": str(OUT_CSV) if OUT_CSV.exists() else None,
                "years_missing_era5": missing[:20],
                "n_missing": len(missing),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if OUT_CSV.exists():
        print(f"Wrote {OUT_CSV} (join {status})")
    else:
        print("No ERA5 years on disk yet — download first.")


if __name__ == "__main__":
    main()
