"""Write held-out final report markdown from sece_era5_environ_heldout results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HELD_RAW = ROOT / "results" / "sece_era5_environ_heldout" / "heldout_multiseed_phase3_raw.csv"
DEV_RAW = ROOT / "results" / "sece_era5_environ" / "dev_multiseed_phase3_raw.csv"
OUT_MD = ROOT / "docs" / "era5_environ_heldout_final_report.md"
SECE = "SECE v2 Phase3"
HORIZONS = ["3h", "12h", "24h", "48h"]


def main() -> None:
    if not HELD_RAW.exists():
        raise SystemExit(f"Missing {HELD_RAW}")

    held = pd.read_csv(HELD_RAW)
    lines = [
        "# Held-out final confirm — environ subset (locked architecture)",
        "",
        "Genesis 1940-2024, Phase 3 SECE + point ERA5 + environ expert.",
        "Seeds: held-out only (`eval_protocol.HELDOUT_SEEDS`). Dev seeds were not used.",
        "",
        "## SECE rank-1 (held-out)",
        "",
        "| Horizon | SECE rank-1 | n seeds |",
        "|---------|------------:|--------:|",
    ]
    for h in HORIZONS:
        sub = held[(held.horizon == h) & (held.model == SECE)]
        n = sub.seed.nunique()
        lines.append(f"| {h} | {int(sub.is_best.sum())}/{n} | {n} |")

    lines.extend([
        "",
        "## SECE mean median km (held-out)",
        "",
        "| Horizon | Mean km |",
        "|---------|--------:|",
    ])
    for h in HORIZONS:
        sub = held[(held.horizon == h) & (held.model == SECE)]
        lines.append(f"| {h} | {sub.median_km.mean():.2f} |")

    lines.extend([
        "",
        "## Rank-1 winners per horizon (held-out, all models)",
        "",
    ])
    for h in HORIZONS:
        best = (
            held[held.horizon == h]
            .sort_values("rank")
            .groupby("seed")["model"]
            .first()
            .value_counts()
        )
        parts = ", ".join(f"{k}({v})" for k, v in best.items())
        lines.append(f"- **{h}**: {parts}")

    if DEV_RAW.exists():
        dev = pd.read_csv(DEV_RAW)
        lines.extend([
            "",
            "## Dev reference (architecture selection only — different seeds)",
            "",
            "| Horizon | Dev SECE rank-1 | Dev mean km | Held-out mean km |",
            "|---------|----------------:|------------:|-----------------:|",
        ])
        for h in HORIZONS:
            ds = dev[(dev.horizon == h) & (dev.model == SECE)]
            hs = held[(held.horizon == h) & (held.model == SECE)]
            lines.append(
                f"| {h} | {int(ds.is_best.sum())}/{ds.seed.nunique()} | "
                f"{ds.median_km.mean():.2f} | {hs.median_km.mean():.2f} |"
            )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
