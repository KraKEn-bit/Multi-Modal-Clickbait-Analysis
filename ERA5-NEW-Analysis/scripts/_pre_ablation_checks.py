"""One-off checks before date-window ablation."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "era5_raw"
CSV = ROOT / "datasets" / "tracks_era5.csv"
EXPECTED = (30.0, 70.0, 5.0, 102.0)  # N, W, S, E
TOL = 0.25


def check_box() -> None:
    print("=== CHECK 1: ERA5 bounding box consistency ===")
    print(f"BOX.txt on disk: { (RAW / 'BOX.txt').read_text(encoding='utf-8').strip() }")
    print(f"Expected N,W,S,E: {EXPECTED}")

    files = sorted(RAW.glob("*.nc"))
    print(f"NetCDF files scanned: {len(files)}")

    bad: list[tuple[str, object]] = []
    bounds_seen: Counter[tuple[float, float, float, float]] = Counter()
    month_bounds: dict[str, set[tuple[float, float, float, float]]] = defaultdict(set)

    for p in files:
        try:
            with xr.open_dataset(p) as ds:
                lat = ds["latitude"].values if "latitude" in ds else ds["lat"].values
                lon = ds["longitude"].values if "longitude" in ds else ds["lon"].values
                n, s = float(lat.max()), float(lat.min())
                w, e = float(lon.min()), float(lon.max())
                key = (round(n, 3), round(w, 3), round(s, 3), round(e, 3))
                bounds_seen[key] += 1
                ok = (
                    abs(n - EXPECTED[0]) <= TOL
                    and abs(s - EXPECTED[2]) <= TOL
                    and abs(w - EXPECTED[1]) <= TOL
                    and abs(e - EXPECTED[3]) <= TOL
                )
                if not ok:
                    bad.append((p.name, key))

                parts = p.stem.split("_")
                y = parts[2]
                if parts[3].startswith("b"):
                    tag = f"{y}-{parts[3][1:]}_{parts[4]}"
                elif parts[3].isdigit():
                    tag = f"{y}-{parts[3]}"
                else:
                    tag = p.stem
                month_bounds[tag].add(key)
        except Exception as exc:  # noqa: BLE001
            bad.append((p.name, f"ERROR: {exc}"))

    print(f"Distinct bounds seen: {len(bounds_seen)}")
    for b, c in bounds_seen.most_common(10):
        print(f"  {b} -> {c} files")

    if bad:
        print(f"OUT-OF-BOX files: {len(bad)}")
        for item in bad[:20]:
            print(" ", item)
    else:
        print("All files match expected box (within 0.25 deg grid tolerance).")

    mixed = {m: s for m, s in month_bounds.items() if len(s) > 1}
    print(f"Month/batch tags with >1 distinct bound: {len(mixed)}")
    for m, s in list(sorted(mixed.items()))[:15]:
        print(f"  {m}: {s}")


def check_missing() -> None:
    print("\n=== CHECK 2: Missing ERA5 by window and variable ===")
    df = pd.read_csv(CSV)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    df["year"] = df["ISO_TIME"].dt.year
    vars_ = [c for c in df.columns if c not in ("SID", "ISO_TIME", "year")]

    windows = {
        "1940-2024": (1940, 2024),
        "1979-2024": (1979, 2024),
        "1990-2024": (1990, 2024),
    }

    for w, (lo, hi) in windows.items():
        sub = df[(df["year"] >= lo) & (df["year"] <= hi)]
        any_miss = 100 * sub[vars_].isna().any(axis=1).mean()
        print(f"\nWindow {w}: {len(sub)} rows, {any_miss:.2f}% rows with any missing")
        for v in vars_:
            print(f"  {v:10s} {100 * sub[v].isna().mean():5.2f}% missing")

    print("\nBy-year (% rows with any ERA5 var missing):")
    yr = df.groupby("year").apply(
        lambda g: 100 * g[vars_].isna().any(axis=1).mean(), include_groups=False
    )
    nonzero = yr[yr > 0]
    for y, p in nonzero.items():
        print(f"  {int(y)}: {p:.1f}%")

    print("\nConcentration summary:")
    for label, lo, hi in [
        ("1940-1978", 1940, 1978),
        ("1979-2024", 1979, 2024),
        ("1990-2024", 1990, 2024),
    ]:
        sub = df[(df["year"] >= lo) & (df["year"] <= hi)]
        n_miss = int(sub[vars_].isna().any(axis=1).sum())
        pct = 100 * n_miss / len(sub) if len(sub) else 0
        print(f"  {label}: {pct:.2f}% rows ({n_miss}/{len(sub)})")

    print("\nTop missing years per variable:")
    for v in vars_:
        byy = df.groupby("year")[v].apply(lambda s: s.isna().mean(), include_groups=False)
        top = byy[byy > 0].sort_values(ascending=False).head(5)
        if len(top):
            txt = ", ".join(f"{int(y)}={100 * p:.1f}%" for y, p in top.items())
            print(f"  {v}: {txt}")
        else:
            print(f"  {v}: none")


if __name__ == "__main__":
    check_box()
    check_missing()
