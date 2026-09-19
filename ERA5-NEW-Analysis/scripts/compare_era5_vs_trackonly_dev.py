"""Report controlled ERA5 vs track-only Phase 3 dev comparison (1940-2024, 9 seeds)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ERA5_RAW = ROOT / "results" / "sece_era5" / "dev_multiseed_phase3_raw.csv"
TRACK_RAW = ROOT / "results" / "sece_trackonly_1940" / "dev_multiseed_phase3_raw.csv"
OUT_MD = ROOT / "docs" / "era5_vs_trackonly_dev_comparison.md"
OUT_CSV = ROOT / "results" / "era5_vs_trackonly_dev_delta.csv"

SECE = "SECE v2 Phase3"
HORIZONS = ["3h", "12h", "24h", "48h"]


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def main() -> None:
    era5 = _load(ERA5_RAW)
    track = _load(TRACK_RAW)

    rows = []
    print("=== SECE rank-1 (vs 7 competitors, same 1940-2024 rows) ===")
    for label, df in [("track-only", track), ("+ERA5", era5)]:
        print(f"\n{label}:")
        for h in HORIZONS:
            sub = df[(df.horizon == h) & (df.model == SECE)]
            print(f"  {h}: {sub.is_best.sum()}/{sub.seed.nunique()}  mean med km={sub.median_km.mean():.2f}")

    print("\n=== Direct delta (+ERA5 minus track-only SECE mean median km) ===")
    for h in HORIZONS:
        e = era5[(era5.horizon == h) & (era5.model == SECE)].set_index("seed")["median_km"]
        t = track[(track.horizon == h) & (track.model == SECE)].set_index("seed")["median_km"]
        common = sorted(set(e.index) & set(t.index))
        delta = e.loc[common] - t.loc[common]
        improved = int((delta < 0).sum())
        print(
            f"  {h}: track={t.loc[common].mean():.2f}  era5={e.loc[common].mean():.2f}  "
            f"delta={delta.mean():+.2f} km  ({improved}/{len(common)} seeds improved)"
        )
        for seed in common:
            rows.append({
                "horizon": h,
                "seed": seed,
                "track_median_km": t.loc[seed],
                "era5_median_km": e.loc[seed],
                "delta_km": e.loc[seed] - t.loc[seed],
                "era5_better": e.loc[seed] < t.loc[seed],
            })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    lines = [
        "# ERA5 vs track-only dev comparison (controlled)",
        "",
        "Phase 3 SECE, genesis 1940-2024, same CSV rows, 9 DEV seeds.",
        "Only difference: 14 ERA5 feature columns included or excluded.",
        "",
        "## SECE rank-1 per horizon",
        "",
        "| Horizon | Track-only | +ERA5 |",
        "|---------|----------:|------:|",
    ]
    for h in HORIZONS:
        tr = track[(track.horizon == h) & (track.model == SECE)].is_best.sum()
        er = era5[(era5.horizon == h) & (era5.model == SECE)].is_best.sum()
        n = track[(track.horizon == h) & (track.model == SECE)].seed.nunique()
        lines.append(f"| {h} | {tr}/{n} | {er}/{n} |")

    lines.extend([
        "",
        "## SECE mean median km",
        "",
        "| Horizon | Track-only | +ERA5 | Delta (ERA5 − track) |",
        "|---------|----------:|------:|---------------------:|",
    ])
    for h in HORIZONS:
        e = era5[(era5.horizon == h) & (era5.model == SECE)]
        t = track[(track.horizon == h) & (track.model == SECE)]
        d = e.median_km.mean() - t.median_km.mean()
        lines.append(
            f"| {h} | {t.median_km.mean():.2f} | {e.median_km.mean():.2f} | {d:+.2f} |"
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
