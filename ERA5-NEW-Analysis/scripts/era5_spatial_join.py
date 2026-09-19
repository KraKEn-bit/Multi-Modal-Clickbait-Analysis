"""
Extract spatial ERA5 patch features (Option A sidecars) from era5_raw/.

Writes two sidecars (keys + 6 spatial cols each, no point duplication):
  datasets/tracks_era5_spatial_5deg.csv  - 5x5 deg patch (half-width 2.5 deg)
  datasets/tracks_era5_spatial_3deg.csv  - 3x3 deg patch (half-width 1.5 deg, local 2-3 deg scale)

Resumes by skipping SID+ISO_TIME already present in each output CSV.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD_READY = ROOT.parent / "Final_Cyclone_Pred_Results_P-3" / "datasets" / "bangladesh_nextstep_dataset_research_ready.csv"
RAW = ROOT / "datasets" / "era5_raw"
PROG = ROOT / "results" / "era5_spatial_join_progress.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from era5_join import _open_kind  # noqa: E402

SPATIAL_COLS = [
    "shear_grad_motion",
    "wspd850_std_patch",
    "msl_trough_dist_km",
    "msl_trough_bearing_deg",
    "msl_patch_anom",
    "shear_asym_motion",
]

PATCHES = {
    "5deg": {"half_deg": 2.5, "out": ROOT / "datasets" / "tracks_era5_spatial_5deg.csv"},
    "3deg": {"half_deg": 1.5, "out": ROOT / "datasets" / "tracks_era5_spatial_3deg.csv"},
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pick(ds, *names: str) -> str:
    for n in names:
        if n in ds:
            return n
    raise KeyError(f"none of {names} in {list(ds.data_vars)}")


def _norm_lon(lon: float) -> float:
    return ((lon + 180) % 360) - 180


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(r * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    y = np.sin(lo2 - lo1) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)
    return float((np.degrees(np.arctan2(y, x)) + 360) % 360)


def _offset_latlon(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    br = np.radians(bearing_deg)
    dlat = (dist_km / 111.0) * np.cos(br)
    dlon = (dist_km / (111.0 * max(np.cos(np.radians(lat)), 0.1))) * np.sin(br)
    return lat + dlat, lon + dlon


def _lat_slice(da, lat0: float, half: float):
    lat = da["latitude"].values
    lo, hi = lat0 - half, lat0 + half
    if lat[0] > lat[-1]:
        return slice(hi, lo)
    return slice(lo, hi)


def _lon_slice(da, lon0: float, half: float):
    lon = da["longitude"].values
    lo, hi = lon0 - half, lon0 + half
    return slice(lo, hi)


def _shear_mag(u850, v850, u200, v200) -> np.ndarray:
    su, sv = u200 - u850, v200 - v850
    return np.sqrt(su * su + sv * sv)


def _wspd850(u850, v850) -> np.ndarray:
    return np.sqrt(u850 * u850 + v850 * v850)


def _nearest_2d(da, lat0: float, lon0: float) -> float:
    import xarray as xr

    v = da.sel(
        latitude=xr.DataArray([lat0], dims="p"),
        longitude=xr.DataArray([lon0], dims="p"),
        method="nearest",
    )
    return float(np.asarray(v.values).reshape(-1)[0])


def _patch_features(
    u850_t, v850_t, u200_t, v200_t, msl_t,
    lat0: float, lon0: float, half: float, motion_bearing: float,
) -> dict[str, float]:
    lat_sl = _lat_slice(u850_t, lat0, half)
    lon_sl = _lon_slice(u850_t, lon0, half)
    u850 = u850_t.sel(latitude=lat_sl, longitude=lon_sl)
    v850 = v850_t.sel(latitude=lat_sl, longitude=lon_sl)
    u200 = u200_t.sel(latitude=lat_sl, longitude=lon_sl)
    v200 = v200_t.sel(latitude=lat_sl, longitude=lon_sl)
    msl = msl_t.sel(latitude=lat_sl, longitude=lon_sl)

    shear = _shear_mag(u850.values, v850.values, u200.values, v200.values)
    wspd = _wspd850(u850.values, v850.values)
    msl_g = msl.values
    lats = msl["latitude"].values
    lons = msl["longitude"].values

    sh_c = float(_shear_mag(
        np.array([_nearest_2d(u850_t, lat0, lon0)]),
        np.array([_nearest_2d(v850_t, lat0, lon0)]),
        np.array([_nearest_2d(u200_t, lat0, lon0)]),
        np.array([_nearest_2d(v200_t, lat0, lon0)]),
    )[0])
    msl_c = _nearest_2d(msl_t, lat0, lon0)

    valid_shear = shear[np.isfinite(shear)]
    valid_wspd = wspd[np.isfinite(wspd)]
    valid_msl = msl_g[np.isfinite(msl_g)]

    wspd_std = float(np.std(valid_wspd)) if valid_wspd.size else np.nan
    msl_anom = float(msl_c - np.nanmean(valid_msl)) if valid_msl.size else np.nan

    # Trough: nearest grid minimum in patch
    if valid_msl.size:
        flat_idx = int(np.nanargmin(msl_g))
        ii, jj = np.unravel_index(flat_idx, msl_g.shape)
        tlat, tlon = float(lats[ii]), float(lons[jj])
        trough_dist = _haversine_km(lat0, lon0, tlat, tlon)
        trough_bear = _bearing_deg(lat0, lon0, tlat, tlon)
    else:
        trough_dist, trough_bear = np.nan, np.nan

    # Shear gradient and asymmetry along motion (sample +/- offset km)
    offset_km = min(half * 111.0 * 0.35, 120.0)
    lat_f, lon_f = _offset_latlon(lat0, lon0, motion_bearing, offset_km)
    lat_b, lon_b = _offset_latlon(lat0, lon0, (motion_bearing + 180) % 360, offset_km)

    def _shear_at(la, lo):
        return float(_shear_mag(
            np.array([_nearest_2d(u850_t, la, lo)]),
            np.array([_nearest_2d(v850_t, la, lo)]),
            np.array([_nearest_2d(u200_t, la, lo)]),
            np.array([_nearest_2d(v200_t, la, lo)]),
        )[0])

    sh_f, sh_b = _shear_at(lat_f, lon_f), _shear_at(lat_b, lon_b)
    shear_grad = (sh_f - sh_c) / max(offset_km, 1.0)
    shear_asym = sh_f - sh_b

    return {
        "shear_grad_motion": shear_grad,
        "wspd850_std_patch": wspd_std,
        "msl_trough_dist_km": trough_dist,
        "msl_trough_bearing_deg": trough_bear,
        "msl_patch_anom": msl_anom,
        "shear_asym_motion": shear_asym,
    }


def _prepare_ds(pl, sl):
    lat = "latitude" if "latitude" in pl.coords else "lat"
    lon = "longitude" if "longitude" in pl.coords else "lon"
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
    lev = "level" if "level" in pl.coords else "pressure_level"
    u_name = _pick(pl, "u", "u_component_of_wind")
    v_name = _pick(pl, "v", "v_component_of_wind")
    msl_name = _pick(sl, "msl", "mean_sea_level_pressure")
    return pl, sl, lev, u_name, v_name, msl_name


def process_year(year: int, tracks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pl = _open_kind(year, "pl")
    sl = _open_kind(year, "sl")
    if pl is None or sl is None:
        return {k: pd.DataFrame() for k in PATCHES}

    sub = tracks[tracks["year"] == year].copy()
    if sub.empty:
        pl.close()
        sl.close()
        return {k: pd.DataFrame() for k in PATCHES}

    pl, sl, lev, u_name, v_name, msl_name = _prepare_ds(pl, sl)
    out_rows = {k: [] for k in PATCHES}

    for _, row in sub.iterrows():
        lat0 = float(row["LAT"])
        lon0 = _norm_lon(float(row["LON"]))
        t = pd.Timestamp(row["ISO_TIME"])
        bear = float(row["geo_bearing"]) if np.isfinite(row["geo_bearing"]) else 0.0

        u850_t = pl[u_name].sel({lev: 850}, method="nearest").sel(time=t, method="nearest")
        v850_t = pl[v_name].sel({lev: 850}, method="nearest").sel(time=t, method="nearest")
        u200_t = pl[u_name].sel({lev: 200}, method="nearest").sel(time=t, method="nearest")
        v200_t = pl[v_name].sel({lev: 200}, method="nearest").sel(time=t, method="nearest")
        msl_t = sl[msl_name].sel(time=t, method="nearest")

        base = {"SID": row["SID"], "ISO_TIME": t}
        for tag, cfg in PATCHES.items():
            feats = _patch_features(u850_t, v850_t, u200_t, v200_t, msl_t, lat0, lon0, cfg["half_deg"], bear)
            out_rows[tag].append({**base, **feats})

    pl.close()
    sl.close()
    return {k: pd.DataFrame(v) for k, v in out_rows.items()}


def _done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    prev = pd.read_csv(path, usecols=["SID", "ISO_TIME"])
    prev["ISO_TIME"] = pd.to_datetime(prev["ISO_TIME"])
    return set(zip(prev["SID"].astype(str), prev["ISO_TIME"].astype(str)))


def _append(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    if path.exists():
        old = pd.read_csv(path)
        old["ISO_TIME"] = pd.to_datetime(old["ISO_TIME"])
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["SID", "ISO_TIME"], keep="last")
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    PROG.parent.mkdir(parents=True, exist_ok=True)
    done = {tag: _done_keys(cfg["out"]) for tag, cfg in PATCHES.items()}

    ready = pd.read_csv(
        OLD_READY,
        usecols=["SID", "ISO_TIME", "LAT", "LON", "geo_bearing", "SEASON"],
    )
    ready["ISO_TIME"] = pd.to_datetime(ready["ISO_TIME"], errors="coerce")
    ready["year"] = ready["ISO_TIME"].dt.year
    ready = ready[(ready["year"] >= 1940) & (ready["year"] <= 2024)].copy()

    years = sorted(int(y) for y in ready["year"].dropna().unique())
    print(f"Spatial join: {len(ready)} track points, {len(years)} years", flush=True)

    for year in years:
        PROG.write_text(
            json.dumps({"status": "running", "year": year, "updated_at": _utc()}, indent=2),
            encoding="utf-8",
        )
        sub = ready[ready["year"] == year]
        pending = sub[
            ~sub.apply(lambda r: (str(r["SID"]), str(r["ISO_TIME"])) in done["5deg"], axis=1)
        ]
        if pending.empty:
            print(f"skip {year} (already done)", flush=True)
            continue
        print(f"process {year}: {len(pending)} points", flush=True)
        parts = process_year(year, pending)
        for tag, cfg in PATCHES.items():
            _append(cfg["out"], parts[tag])
            done[tag] = _done_keys(cfg["out"])
        print(f"  wrote year {year}", flush=True)

    elapsed_h = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
    summary = {
        "status": "complete",
        "updated_at": _utc(),
        "elapsed_hours": round(elapsed_h, 2),
        "outputs": {tag: str(cfg["out"]) for tag, cfg in PATCHES.items()},
        "spatial_cols": SPATIAL_COLS,
        "n_5deg": int(sum(1 for _ in open(PATCHES["5deg"]["out"])) - 1) if PATCHES["5deg"]["out"].exists() else 0,
        "n_3deg": int(sum(1 for _ in open(PATCHES["3deg"]["out"])) - 1) if PATCHES["3deg"]["out"].exists() else 0,
    }
    PROG.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Complete in {elapsed_h:.2f} h", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
