"""Final spatial controlled comparison vs point-ERA5 and environ-subset baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SECE = "SECE v2 Phase3"
HORIZONS = ["3h", "12h", "24h", "48h"]

RUNS = {
    "point_era5": ROOT / "results" / "sece_era5" / "dev_multiseed_phase3_raw.csv",
    "environ_subset": ROOT / "results" / "sece_era5_environ" / "dev_multiseed_phase3_raw.csv",
    "spatial_5deg": ROOT / "results" / "sece_era5_spatial_5deg" / "dev_multiseed_phase3_raw.csv",
    "spatial_3deg": ROOT / "results" / "sece_era5_spatial_3deg" / "dev_multiseed_phase3_raw.csv",
}


def _sece_stats(df: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for h in HORIZONS:
        sub = df[(df.horizon == h) & (df.model == SECE)]
        out[h] = {
            "rank1": int(sub.is_best.sum()),
            "n": int(sub.seed.nunique()),
            "mean_km": float(sub.median_km.mean()),
        }
    return out


def _best_baseline(stats: dict[str, dict[str, dict]]) -> str:
    """Pick lowest 24h+48h mean km among point and environ."""
    cands = ["point_era5", "environ_subset"]
    score = {}
    for name in cands:
        score[name] = stats[name]["24h"]["mean_km"] + stats[name]["48h"]["mean_km"]
    return min(score, key=score.get)


def main() -> None:
    join_prog = ROOT / "results" / "era5_spatial_join_progress.json"
    join_h = None
    if join_prog.exists():
        join_h = json.loads(join_prog.read_text()).get("elapsed_hours")

    frames = {k: pd.read_csv(v) for k, v in RUNS.items() if v.exists()}
    stats = {k: _sece_stats(v) for k, v in frames.items()}
    best = _best_baseline(stats) if "point_era5" in stats and "environ_subset" in stats else "point_era5"

    lines = [
        "# Spatial patch final dev comparison",
        "",
        f"Join runtime (spatial extraction): {join_h} h" if join_h else "",
        f"Current best baseline (24h+48h mean km): **{best}**",
        "",
        "## SECE rank-1 per horizon",
        "",
        "| Horizon | Point ERA5 | Environ | Spatial 5deg | Spatial 3deg |",
        "|---------|----------:|--------:|-------------:|-------------:|",
    ]
    for h in HORIZONS:
        row = [h]
        for key in ["point_era5", "environ_subset", "spatial_5deg", "spatial_3deg"]:
            if key in stats:
                row.append(f"{stats[key][h]['rank1']}/{stats[key][h]['n']}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## SECE mean median km",
        "",
        "| Horizon | Point | Environ | Spatial 5deg | Spatial 3deg |",
        "|---------|------:|--------:|-------------:|-------------:|",
    ])
    for h in HORIZONS:
        row = [h]
        for key in ["point_era5", "environ_subset", "spatial_5deg", "spatial_3deg"]:
            row.append(f"{stats[key][h]['mean_km']:.2f}" if key in stats else "-")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", f"## Delta vs current best ({best})", ""])
    for label, key in [("Spatial 5deg", "spatial_5deg"), ("Spatial 3deg", "spatial_3deg")]:
        if key not in stats:
            continue
        lines.append(f"### {label}")
        for h in HORIZONS:
            base_km = stats[best][h]["mean_km"]
            new_km = stats[key][h]["mean_km"]
            d = new_km - base_km
            lines.append(f"- **{h}**: {d:+.2f} km ({new_km:.2f} vs {base_km:.2f})")
        lines.append("")

    out_md = ROOT / "docs" / "era5_spatial_final_dev_comparison.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
