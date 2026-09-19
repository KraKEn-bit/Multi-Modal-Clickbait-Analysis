"""Master summary table: SECE rank-1 and mean median km across NEW WAY experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SECE = "SECE v2 Phase3"
HORIZONS = ["3h", "12h", "24h", "48h"]

EXPERIMENTS = [
    ("Track-only (no ERA5)", ROOT / "results" / "sece_trackonly_1940" / "dev_multiseed_phase3_raw.csv"),
    ("Point ERA5", ROOT / "results" / "sece_era5" / "dev_multiseed_phase3_raw.csv"),
    ("Environ subset", ROOT / "results" / "sece_era5_environ" / "dev_multiseed_phase3_raw.csv"),
    ("Spatial 5deg", ROOT / "results" / "sece_era5_spatial_5deg" / "dev_multiseed_phase3_raw.csv"),
    ("Spatial 3deg", ROOT / "results" / "sece_era5_spatial_3deg" / "dev_multiseed_phase3_raw.csv"),
    ("Held-out (locked environ)", ROOT / "results" / "sece_era5_environ_heldout" / "heldout_multiseed_phase3_raw.csv"),
]

OUT_MD = ROOT / "docs" / "full_experiment_summary_table.md"
OUT_CSV = ROOT / "docs" / "full_experiment_summary_table.csv"


def summarize_experiment(name: str, path: Path) -> list[dict]:
    df = pd.read_csv(path)
    sece = df[df.model == SECE]
    rows = []
    for h in HORIZONS:
        sub = sece[sece.horizon == h]
        n = sub.seed.nunique()
        rank1 = int(sub.is_best.sum())
        mean_med = float(sub.median_km.mean()) if len(sub) else float("nan")
        rows.append(
            {
                "experiment": name,
                "horizon": h,
                "sece_rank1": rank1,
                "n_seeds": n,
                "sece_rank1_label": f"{rank1}/{n}",
                "sece_mean_median_km": round(mean_med, 2),
            }
        )
    return rows


def main() -> None:
    all_rows: list[dict] = []
    for name, path in EXPERIMENTS:
        if not path.exists():
            raise SystemExit(f"Missing results: {path}")
        all_rows.extend(summarize_experiment(name, path))

    table = pd.DataFrame(all_rows)
    table.to_csv(OUT_CSV, index=False)

    lines = [
        "# Full NEW WAY experiment summary (SECE v2 Phase 3)",
        "",
        "Master reference for the results section. **SECE rank-1** = count of seeds where SECE "
        "has lowest test median km; **mean median km** = mean of SECE test medians across seeds.",
        "",
        "Dev experiments use seeds `[3,4,5,6,8,9,10,11,14]`; held-out uses `HELDOUT_SEEDS` only.",
        "",
        f"CSV: `{OUT_CSV.relative_to(ROOT)}`",
        "",
        "## SECE rank-1 by experiment",
        "",
        "| Experiment | 3h | 12h | 24h | 48h |",
        "|------------|----|-----|-----|-----|",
    ]
    for name, _ in EXPERIMENTS:
        parts = [name]
        for h in HORIZONS:
            r = table[(table.experiment == name) & (table.horizon == h)].iloc[0]
            parts.append(r["sece_rank1_label"])
        lines.append("| " + " | ".join(parts) + " |")

    lines.extend([
        "",
        "## SECE mean median km by experiment",
        "",
        "| Experiment | 3h | 12h | 24h | 48h |",
        "|------------|---:|----:|----:|----:|",
    ])
    for name, _ in EXPERIMENTS:
        parts = [name]
        for h in HORIZONS:
            r = table[(table.experiment == name) & (table.horizon == h)].iloc[0]
            parts.append(f"{r['sece_mean_median_km']:.2f}")
        lines.append("| " + " | ".join(parts) + " |")

    lines.extend([
        "",
        "## Long format",
        "",
        "| Experiment | Horizon | SECE rank-1 | n seeds | Mean median km |",
        "|------------|---------|------------:|--------:|---------------:|",
    ])
    for _, r in table.iterrows():
        lines.append(
            f"| {r['experiment']} | {r['horizon']} | {r['sece_rank1_label']} | {int(r['n_seeds'])} | {r['sece_mean_median_km']:.2f} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
