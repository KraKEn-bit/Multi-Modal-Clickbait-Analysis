"""Publication figures for the AMS WAF manuscript.

Every panel is built from an already-computed file. Colour convention used
throughout: blue = kinematic / track-derived; orange = ERA5-derived.
Other encodings (model family, horizon) use the remaining Okabe–Ito colours
so they are not confused with that split.

Outputs (PNG 300 dpi + PDF vector) under Paper Writing/WAF/:
  fig03featimp.{png,pdf}   Figure A  04_environ_expert_feature_importance.csv
  fig04era5hist.{png,pdf}  Figure B  05_seed0_lgb_era5_vs_trackonly_per_storm.csv
  fig05bench4h.{png,pdf}   Figure C  held-out prediction CSVs -> JSON per horizon
  fig06ablation.{png,pdf}  Figure D  full_experiment_summary_table.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate_paper_figures as gpf  # noqa: E402

OUT = ROOT / "Paper Writing" / "WAF"
DATA = ROOT / "docs" / "paper_writeup_data"
FIG_DOCS = ROOT / "docs" / "figures"

# Okabe–Ito; kinematic/ERA5 convention is locked for the whole paper.
BLUE_KIN = "#0072B2"
ORANGE_ERA5 = "#E69F00"
BLACK = "#000000"
GREEN = "#009E73"
VERM = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#5A5A5A"

HORIZON_COLOR = {"3h": BLUE_KIN, "12h": GREEN, "24h": VERM, "48h": PURPLE}
HORIZON_PANEL_TITLE = {"3h": "3 h held-out", "12h": "12 h held-out", "24h": "24 h held-out", "48h": "48 h held-out"}
HORIZON_LABEL = {"3h": "3 h", "12h": "12 h", "24h": "24 h", "48h": "48 h"}

FEATURE_LABELS = {
    "STORM_DIR": "Storm direction",
    "STORM_SPEED": "Storm speed",
    "LANDFALL": "Landfall flag",
    "DIST2LAND": "Distance to land",
    "NEWDELHI_WIND": "Wind estimate",
    "NEWDELHI_WIND_missing": "Wind missing flag",
    "sst": "SST",
    "sst_missing": "SST missing flag",
    "u200": "u 200 hPa",
    "v200": "v 200 hPa",
    "u500": "u 500 hPa",
    "v500": "v 500 hPa",
    "u850": "u 850 hPa",
    "v850": "v 850 hPa",
    "msl": "MSL pressure",
    "steer_u": "Steering u",
    "steer_v": "Steering v",
    "shear_u": "Shear u",
    "shear_v": "Shear v",
    "shear_mag": "Shear magnitude",
}

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _grid(ax) -> None:
    ax.grid(True, ls=":", alpha=0.3, color="#666666")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.8)


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print("wrote", png.name, "and", pdf.name)


def draw_feature_importance() -> None:
    """Figure A: 04_environ_expert_feature_importance.csv."""
    _apply_style()
    df = pd.read_csv(DATA / "04_environ_expert_feature_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 7.2), sharey=False)

    for ax, horizon in zip(axes, ("24h", "48h")):
        sub = df[df["horizon"] == horizon].sort_values("gain_share", ascending=True)
        colors = [ORANGE_ERA5 if bool(v) else BLUE_KIN for v in sub["is_era5"]]
        labels = [FEATURE_LABELS.get(f, f) for f in sub["feature"]]
        ax.barh(labels, sub["gain_share"] * 100.0, color=colors, edgecolor="none", height=0.78)
        ax.set_xlabel("Gain share (%)")
        ax.set_title(f"{HORIZON_LABEL[horizon]} environ expert")
        _grid(ax)
        ax.set_xlim(0, max(55.0, float(sub["gain_share"].max()) * 100.0 * 1.08))

    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor=BLUE_KIN, edgecolor="none", label="Kinematic / track-derived"),
            Patch(facecolor=ORANGE_ERA5, edgecolor="none", label="ERA5-derived"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig03featimp")


def draw_era5_histograms() -> None:
    """Figure B: 05_seed0_lgb_era5_vs_trackonly_per_storm.csv."""
    _apply_style()
    df = pd.read_csv(DATA / "05_seed0_lgb_era5_vs_trackonly_per_storm.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=False)

    for ax, horizon in zip(axes, ("24h", "48h")):
        d = df.loc[df["horizon"] == horizon, "delta_km"].astype(float)
        n = int(d.size)
        n_imp = int((d > 0).sum())
        med = float(d.median())
        bins = np.histogram_bin_edges(d, bins="fd")
        ax.hist(d, bins=bins, color=ORANGE_ERA5, edgecolor="white", linewidth=0.4, alpha=0.92)
        ax.axvline(0.0, color=BLUE_KIN, lw=1.6, label="Zero (parity)")
        ax.axvline(med, color=BLACK, lw=1.6, ls="--", label=f"Median ({med:.1f} km)")
        ax.set_xlabel(r"Per-storm $\Delta$ (km)")
        ax.set_ylabel("Number of storms")
        ax.set_title(f"{HORIZON_LABEL[horizon]}  ({n_imp}/{n} improved)")
        _grid(ax)
        ax.legend(loc="upper right", framealpha=0.92, edgecolor="#CCCCCC")

    fig.tight_layout()
    _save(fig, "fig04era5hist")


def _stats_to_records(rows) -> list[dict]:
    ranked = sorted(rows, key=lambda r: (r[4], r[3]))
    return [
        {
            "rank": i + 1,
            "label": row[0],
            "family": row[1],
            "full": row[2],
            "mean_km": round(row[3], 3),
            "median_km": round(row[4], 3),
        }
        for i, row in enumerate(ranked)
    ]


def _load_or_compute_horizon(horizon: str) -> list[dict]:
    cache = FIG_DOCS / f"benchmark_{horizon}_heldout_stats.json"
    if cache.exists():
        recs = json.loads(cache.read_text(encoding="utf-8"))
        if recs and "mean_km" in recs[0]:
            print("loaded", cache.name)
            return recs
    rows = gpf._benchmark_stats_horizon(horizon)
    recs = _stats_to_records(rows)
    cache.write_text(json.dumps(recs, indent=2), encoding="utf-8")
    print("computed and wrote", cache.name)
    return recs


def draw_benchmark_four_horizon() -> None:
    """Figure C: same label layout as benchmark_3h_heldout.png on each panel."""
    _apply_style()
    stats = {h: _load_or_compute_horizon(h) for h in ("3h", "12h", "24h", "48h")}
    audit = OUT / "fig04bench4h_stats.json"
    audit.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.8))
    legend_handles = []

    for ax, horizon in zip(axes.ravel(), ("3h", "12h", "24h", "48h")):
        rows = gpf._stats_rows_from_json(stats[horizon])
        handles = gpf.plot_benchmark_on_ax(
            ax,
            rows,
            panel_title=HORIZON_PANEL_TITLE[horizon],
            compact=True,
            fixed_3h_limits=(horizon == "3h"),
            fixed_12h_labels=(horizon == "12h"),
            fixed_24h_labels=(horizon == "24h"),
            fixed_48h_labels=(horizon == "48h"),
        )
        if not legend_handles:
            legend_handles = handles

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.01),
        title="Model family",
        fontsize=10,
        title_fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig04bench4h")


def draw_ablation_trend() -> None:
    """Figure D: full_experiment_summary_table.csv."""
    _apply_style()
    df = pd.read_csv(ROOT / "docs" / "full_experiment_summary_table.csv")
    order = [
        "Track-only (no ERA5)",
        "Point ERA5",
        "Environ subset",
        "Spatial 5deg",
        "Spatial 3deg",
        "Held-out (locked environ)",
    ]
    xlabels = [
        "Track-only",
        "Point ERA5",
        "Environ expert",
        r"Spatial 5$^\circ$",
        r"Spatial 3$^\circ$",
        "Held-out confirm",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=True)

    for ax, horizon in zip(axes.ravel(), ("3h", "12h", "24h", "48h")):
        ys = []
        for exp in order:
            row = df[(df["experiment"] == exp) & (df["horizon"] == horizon)]
            if row.empty:
                raise RuntimeError(f"Missing {exp} {horizon}")
            ys.append(float(row["sece_mean_median_km"].iloc[0]))
        ax.plot(
            range(len(order)),
            ys,
            color=HORIZON_COLOR[horizon],
            marker="o",
            ms=6.5,
            lw=2.0,
            markerfacecolor=HORIZON_COLOR[horizon],
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.2f}", (i, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(xlabels, rotation=28, ha="right")
        ax.set_ylabel("Mean of seed medians (km)")
        ax.set_title(HORIZON_LABEL[horizon])
        _grid(ax)
        pad = 0.08 * (max(ys) - min(ys) if max(ys) > min(ys) else max(ys) * 0.02)
        ax.set_ylim(min(ys) - pad - 0.02, max(ys) + pad + 0.15)

    fig.tight_layout()
    _save(fig, "fig06ablation")


def main() -> None:
    only = set(sys.argv[1:])
    run_all = not only
    if run_all or "a" in only:
        draw_feature_importance()
    if run_all or "b" in only:
        draw_era5_histograms()
    if run_all or "d" in only:
        draw_ablation_trend()
    if run_all or "c" in only:
        draw_benchmark_four_horizon()
    print("WAF figures done.")


if __name__ == "__main__":
    main()
