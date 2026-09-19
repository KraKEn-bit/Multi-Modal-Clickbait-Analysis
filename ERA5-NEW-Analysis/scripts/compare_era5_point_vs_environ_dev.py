"""Compare point-ERA5 Phase 3 vs Phase 3 + environ subset (same seeds/rows)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
POINT_RAW = ROOT / "results" / "sece_era5" / "dev_multiseed_phase3_raw.csv"
ENV_RAW = ROOT / "results" / "sece_era5_environ" / "dev_multiseed_phase3_raw.csv"
OUT_MD = ROOT / "docs" / "era5_point_vs_environ_dev_comparison.md"
OUT_CSV = ROOT / "results" / "era5_point_vs_environ_dev_delta.csv"
SECE = "SECE v2 Phase3"
HORIZONS = ["3h", "12h", "24h", "48h"]


def main() -> None:
    point = pd.read_csv(POINT_RAW)
    env = pd.read_csv(ENV_RAW)

    print("=== SECE rank-1 (point-ERA5 vs +environ subset) ===")
    for label, df in [("point-ERA5", point), ("+environ subset", env)]:
        print(f"\n{label}:")
        for h in HORIZONS:
            sub = df[(df.horizon == h) & (df.model == SECE)]
            print(f"  {h}: {sub.is_best.sum()}/{sub.seed.nunique()}  mean={sub.median_km.mean():.2f} km")

    rows = []
    print("\n=== Delta (environ subset minus point-ERA5 SECE km) ===")
    for h in HORIZONS:
        e = env[(env.horizon == h) & (env.model == SECE)].set_index("seed")["median_km"]
        p = point[(point.horizon == h) & (point.model == SECE)].set_index("seed")["median_km"]
        common = sorted(set(e.index) & set(p.index))
        d = e.loc[common] - p.loc[common]
        print(
            f"  {h}: point={p.loc[common].mean():.2f}  environ={e.loc[common].mean():.2f}  "
            f"delta={d.mean():+.2f} km  ({int((d < 0).sum())}/{len(common)} improved)"
        )
        for seed in common:
            rows.append({
                "horizon": h,
                "seed": seed,
                "point_median_km": p.loc[seed],
                "environ_median_km": e.loc[seed],
                "delta_km": e.loc[seed] - p.loc[seed],
                "environ_better": e.loc[seed] < p.loc[seed],
            })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    lines = [
        "# Point-ERA5 vs environ-subset Phase 3 (controlled)",
        "",
        "Same 1940-2024 rows, 9 DEV seeds, 63 features. Only change: environ subset expert added.",
        "",
        "## SECE rank-1",
        "",
        "| Horizon | Point-ERA5 | +Environ subset |",
        "|---------|----------:|----------------:|",
    ]
    for h in HORIZONS:
        pr = point[(point.horizon == h) & (point.model == SECE)].is_best.sum()
        er = env[(env.horizon == h) & (env.model == SECE)].is_best.sum()
        n = point[(point.horizon == h) & (point.model == SECE)].seed.nunique()
        lines.append(f"| {h} | {pr}/{n} | {er}/{n} |")

    lines.extend([
        "",
        "## SECE mean median km",
        "",
        "| Horizon | Point-ERA5 | +Environ | Delta |",
        "|---------|----------:|---------:|------:|",
    ])
    for h in HORIZONS:
        e = env[(env.horizon == h) & (env.model == SECE)]
        p = point[(point.horizon == h) & (point.model == SECE)]
        d = e.median_km.mean() - p.median_km.mean()
        lines.append(f"| {h} | {p.median_km.mean():.2f} | {e.median_km.mean():.2f} | {d:+.2f} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
