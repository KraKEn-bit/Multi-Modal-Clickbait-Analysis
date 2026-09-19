"""
Generate NEW WAY paper figures (ERA5 + locked Phase 3 held-out).

Outputs under docs/figures/:
  - sece_phase3_architecture.png (+ .drawio source)
  - tsne_era5_63d.png
  - case_study_laila_trajectories.png
  - benchmark_3h_heldout.png
"""
from __future__ import annotations

import json
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / "Final_Cyclone_Pred_Results_P-3"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT))

import pipeline_era5  # noqa: E402
from eval_protocol import HELDOUT_SEEDS  # noqa: E402
from pipeline_common import (  # noqa: E402
    _prepare_eval_frame,
    apply_physics_features,
    apply_qc,
    build_mh_df,
    extract_model_names,
    haversine_km,
    sample_errors_for_model,
    storm_split,
    track_errors_km,
)

FIG = ROOT / "docs" / "figures"
PREDS_DIR = ROOT / "results" / "sece_era5_environ_heldout" / "task_a_test_predictions"
TRAJ_CSV = ROOT / "docs" / "paper_writeup_data" / "05_case_study_trajectories.csv"
HELDOUT_CSV = ROOT / "docs" / "paper_writeup_data" / "01_heldout_comparison_ci_wilcoxon.csv"
CASE_LAILA = "2010137N10090"
REFERENCE = "SECE v2 Phase3"
HORIZONS = ["3h", "12h", "24h", "48h"]
# Journal / WAF: four lead times in a 2×2 grid (see draw_laila_trajectories).
TRAJ_HORIZONS = ["3h", "12h", "24h", "48h"]

# Held-out mean test medians (9 seeds) — architecture output labels
HELDOUT_MEAN_MEDIAN_KM = {"3h": 4.90, "12h": 35.63, "24h": 101.21, "48h": 246.99}

BENCHMARK_MODELS = [
    ("SECE v2 Phase3", "SECE Phase 3", "SECE"),
    ("Stacking Ensemble", "Stacking Ens.", "Ensemble"),
    ("Random Forest", "Random Forest", "Tree-based"),
    ("LightGBM", "LightGBM", "Tree-based"),
    ("XGBoost", "XGBoost", "Tree-based"),
    ("CB+MotionNN", "CB+MotionNN", "Hybrid"),
    ("Persistence Residual Cascade", "PRC", "Hybrid"),
    ("CatBoost", "CatBoost", "Tree-based"),
]

FAMILY_STYLE = {
    "SECE": dict(marker="*", color="#F4C430", s=320, z=6, edge="#C48A00"),
    "Ensemble": dict(marker="s", color="#4C78A8", s=95, z=5, edge="#2F4E6F"),
    "Tree-based": dict(marker="o", color="#E45756", s=88, z=4, edge="#9B2C2C"),
    "Hybrid": dict(marker="^", color="#54A24B", s=115, z=5, edge="#2E6B2A"),
}

# Legacy-style label anchors (x, y, ha). Top cluster: stacked under SECE on the right.
BENCHMARK_LABEL_XY: dict[str, tuple[float, float, str]] = {
    "SECE Phase 3": (6.92, 1.36, "left"),
    "Stacking Ens.": (7.22, 1.04, "left"),
    "Random Forest": (7.22, 0.70, "left"),
    "CB+MotionNN": (8.95, 0.38, "left"),
    "LightGBM": (8.95, 0.02, "left"),
    "XGBoost": (8.95, -0.34, "left"),
    "PRC": (11.12, -0.28, "right"),
    "CatBoost": (9.05, -1.02, "right"),
}

# Tiny display-only shifts so overlapping top-cluster markers stay distinguishable.
BENCHMARK_DISPLAY_OFFSET: dict[str, tuple[float, float]] = {
    "SECE Phase 3": (-0.08, 0.03),
    "Random Forest": (-0.05, -0.04),
    "Stacking Ens.": (0.07, 0.02),
}

# 12 h held-out panel (WAF fig04): manual label anchors in data coordinates (km, accuracy).
BENCHMARK_12H_LABEL_XY: dict[str, tuple[float, float, str]] = {
    "SECE Phase 3": (45.0, 0.16, "left"),
    "Stacking Ens.": (46.0, 0.10, "left"),
    "XGBoost": (46.3, 0.05, "left"),
    "LightGBM": (43.7, 0.01, "left"),
    "PRC": (46.8, 0.02, "left"),
    "Random Forest": (44.5, -0.10, "left"),
    "CB+MotionNN": (48.0, -0.15, "left"),
    "CatBoost": (50.0, -0.30, "left"),
}
_BENCH_12H_XLIM = (42.8, 52.2)
_BENCH_12H_YLIM = (-0.42, 0.24)

# 24 h held-out panel (WAF fig04): manual label anchors (km, accuracy).
BENCHMARK_24H_LABEL_XY: dict[str, tuple[float, float, str]] = {
    "SECE Phase 3": (117.5, 0.20, "left"),
    "Stacking Ens.": (118.5, 0.15, "left"),
    "XGBoost": (119.5, 0.11, "left"),
    "LightGBM": (120.5, 0.05, "left"),
    "PRC": (120.4, 0.02, "left"),
    "Random Forest": (122.0, -0.07, "left"),
    "CB+MotionNN": (122.5, -0.03, "left"),
    "CatBoost": (126.5, -0.15, "left"),
}
_BENCH_24H_XLIM = (116.2, 128.5)
_BENCH_24H_YLIM = (-0.22, 0.24)

# 48 h held-out panel (WAF fig04): manual label anchors (km, accuracy).
BENCHMARK_48H_LABEL_XY: dict[str, tuple[float, float, str]] = {
    # Top cluster (markers ~279–281 km): fan left + right so boxes/lines do not overlap.
    "SECE Phase 3": (279.5, 0.21, "left"),
    "Stacking Ens.": (282.7, 0.08, "left"),
    "XGBoost": (282.0, 0.18, "left"),
    "PRC": (284.0, 0.15, "left"),
    "LightGBM": (280.8, -0.02, "left"),
    "CB+MotionNN": (287.2, 0.03, "left"),
    "Random Forest": (287.0, -0.14, "left"),
    "CatBoost": (292.0, -0.05, "left"),
}
_BENCH_48H_XLIM = (278.0, 294.5)
_BENCH_48H_YLIM = (-0.22, 0.26)

# ── Architecture (matplotlib + draw.io) ─────────────────────────────────────

def iso_box(ax, x, y, w, h, d, face, edge="#333333", lw=0.7, z=3):
    dx, dy = d, d * 0.55
    shade = lambda c, k: tuple(max(0.0, min(1.0, v * k)) for v in mcolors.to_rgb(c))

    ax.add_patch(
        Polygon(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            closed=True,
            facecolor=face,
            edgecolor=edge,
            lw=lw,
            zorder=z,
        )
    )
    ax.add_patch(
        Polygon(
            [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
            closed=True,
            facecolor=shade(face, 0.72),
            edgecolor=edge,
            lw=lw,
            zorder=z,
        )
    )
    ax.add_patch(
        Polygon(
            [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)],
            closed=True,
            facecolor=shade(face, 1.12),
            edgecolor=edge,
            lw=lw,
            zorder=z + 0.1,
        )
    )


def arrow(ax, x1, y1, x2, y2, color="#444", lw=1.1):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10),
        zorder=2,
    )


def draw_architecture_png() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(19.2, 6.8))
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F3F6F4")
    for gx in np.arange(0, 19.3, 0.35):
        ax.axvline(gx, color="#D5DDD8", lw=0.4, zorder=0)
    for gy in np.arange(0, 6.9, 0.35):
        ax.axhline(gy, color="#D5DDD8", lw=0.4, zorder=0)

    def stage(x, title):
        ax.text(x, 6.55, title, ha="center", va="top", fontsize=8.3, color="#4A4A4A", fontstyle="italic")

    # Stage 1
    stage(1.2, "Stage 1")
    ax.text(1.2, 6.25, "INPUT FEATURES", ha="center", fontsize=10, fontweight="bold", color="#2F4A63")
    cols = ["#8FB4D4", "#7BA4C8", "#6A93BA", "#5A84AD"]
    for i, c in enumerate(cols):
        iso_box(ax, 0.55 + i * 0.13, 2.65 + i * 0.13, 1.15, 2.05, 0.28, c, z=2 + i)
    ax.text(1.2, 4.95, "63-D vector", ha="center", fontsize=8, color="white", fontweight="bold", zorder=8)
    ax.text(
        1.2,
        3.55,
        "49 track/physics\n+ 14 ERA5 point",
        ha="center",
        fontsize=6.5,
        color="white",
        zorder=8,
    )
    ax.text(1.2, 1.95, "Per origin (3h QC table\nor multi-horizon row)", ha="center", fontsize=7, color="#444")

    # Stage 2 — horizon-specific subsets
    stage(3.7, "Stage 2")
    ax.text(3.7, 6.25, "SUBSET EXPERTS × 4", ha="center", fontsize=10, fontweight="bold", color="#2F4A63")
    subsets_3h = [
        ("Position", "#4C90C8"),
        ("Motion", "#3D9B8F"),
        ("Environ + ERA5", "#2F8F6B"),
        ("Full 63d", "#5A6A8A"),
    ]
    y0 = 5.05
    for i, (name, col) in enumerate(subsets_3h):
        yy = y0 - i * 0.55
        iso_box(ax, 2.65, yy, 2.05, 0.44, 0.16, col, z=4)
        ax.text(3.68, yy + 0.22, name, ha="center", va="center", fontsize=7.8, color="white", fontweight="bold", zorder=8)
        arrow(ax, 2.05, 3.55, 2.60, yy + 0.22, color="#6A6A6A", lw=0.75)
    ax.text(3.68, 1.55, "12h–48h: motion, physics,\nenviron, full", ha="center", fontsize=7, color="#333")

    # Stage 3 — trees per subset
    stage(6.35, "Stage 3")
    ax.text(6.35, 6.25, "SUBSET TREE BANK", ha="center", fontsize=10, fontweight="bold", color="#2F4A63")
    tree_cols = ["#D94A4A", "#E0673A", "#C43C3C", "#A83232"]
    tree_labs = ["CB", "XGB", "LGB", "RF"]
    for i, (c, lab) in enumerate(zip(tree_cols, tree_labs)):
        iso_box(ax, 5.55, 2.45 + i * 0.42, 0.72, 0.38, 0.14, c, z=4 + i)
        ax.text(5.91, 2.62 + i * 0.42, lab, ha="center", va="center", fontsize=6.5, color="white", zorder=8)
    ax.text(5.95, 2.05, "4 algos × 4 subsets\n= 16 experts / horizon", ha="center", fontsize=7, color="#444")
    ax.text(6.95, 3.55, "NNLS fuse\nper subset", ha="center", fontsize=7.2, color="#222", fontweight="bold")
    arrow(ax, 4.85, 3.55, 5.45, 3.55, color="#555")

    # Stage 4 — champions
    stage(8.85, "Stage 4")
    ax.text(8.85, 6.25, "CHAMPION POOL", ha="center", fontsize=10, fontweight="bold", color="#2F4A63")
    champs = ["Stacking", "PRC", "RF", "LGB", "XGB", "CatBoost", "CB+MotionNN"]
    for i, lab in enumerate(champs):
        yy = 4.95 - i * 0.38
        iso_box(ax, 7.95, yy, 1.75, 0.32, 0.12, "#6A8FBF" if i < 2 else "#5B7FA8", z=4)
        ax.text(8.82, yy + 0.16, lab, ha="center", va="center", fontsize=6.8, color="white", zorder=8)
    ax.text(8.85, 1.55, "7 standalone baselines\n(trained per seed)", ha="center", fontsize=7, color="#444")
    arrow(ax, 7.35, 3.55, 7.90, 3.55, color="#555")

    # Stage 5 — fusion candidates
    stage(11.35, "Stage 5")
    ax.text(11.35, 6.25, "FUSION CANDIDATES", ha="center", fontsize=10, fontweight="bold", color="#2F4A63")
    fusions = [
        "SECE NNLS (7 champs + subset)",
        "NNLS champions only",
        "RF + Stacking NNLS (3h)",
        "Stacking + residual LGB (3h)",
        "km-NNLS (24h/48h)",
        "LGB + PRC NNLS (48h)",
    ]
    iso_box(ax, 10.35, 2.35, 2.05, 2.55, 0.18, "#7FBF7A", z=4)
    for i, t in enumerate(fusions):
        ax.text(11.38, 4.55 - i * 0.32, t, ha="center", fontsize=6.5, color="#163816", zorder=8)
    ax.text(11.38, 2.05, "Val-set only", ha="center", fontsize=7, color="#333")
    arrow(ax, 9.85, 3.2, 10.30, 3.55, color="#555")

    # Stage 6 — router
    stage(14.0, "Stage 6")
    ax.text(14.0, 6.25, "PHASE 3 ROUTER", ha="center", fontsize=10, fontweight="bold", color="#2F4A63")
    iso_box(ax, 12.85, 2.55, 2.35, 2.35, 0.22, "#E3C14A", z=5)
    ax.text(
        14.02,
        4.05,
        "Validation-best\nmedian km",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#3A2E00",
        zorder=8,
    )
    ax.text(14.02, 3.35, "Pick lowest val\nHaversine median", ha="center", fontsize=7, color="#3A2E00", zorder=8)
    ax.text(14.02, 2.85, "Per horizon, per seed", ha="center", fontsize=6.8, color="#3A2E00", zorder=8)
    arrow(ax, 12.55, 3.55, 12.80, 3.55, color="#555")

    # Stage 7 — outputs
    stage(16.85, "Stage 7")
    ax.text(16.85, 6.25, "OUTPUTS", ha="center", fontsize=10, fontweight="bold", color="#2F4A63")
    outs = [
        ("3h", HELDOUT_MEAN_MEDIAN_KM["3h"]),
        ("12h", HELDOUT_MEAN_MEDIAN_KM["12h"]),
        ("24h", HELDOUT_MEAN_MEDIAN_KM["24h"]),
        ("48h", HELDOUT_MEAN_MEDIAN_KM["48h"]),
    ]
    y_out = 4.85
    for lab, km in outs:
        iso_box(ax, 16.15, y_out, 0.72, 0.62, 0.16, "#E07A6A", z=5)
        ax.text(17.05, y_out + 0.31, f"{lab}  {km:.1f} km", ha="left", va="center", fontsize=8, fontweight="bold")
        y_out -= 0.72
    arrow(ax, 15.35, 3.55, 16.10, 3.85, color="#C48A00", lw=1.2)
    ax.text(16.85, 1.75, "Held-out mean test median (9 seeds)", ha="center", fontsize=7, color="#444")
    ax.text(16.85, 1.45, "ΔLat + ΔLon → Haversine km", ha="center", fontsize=7, color="#666")

    legend_items = [
        ("#7BA4C8", "63-D input (ERA5 point)"),
        ("#4C90C8", "4 subset channels"),
        ("#D94A4A", "16 tree experts / horizon"),
        ("#5B7FA8", "7 champion models"),
        ("#7FBF7A", "Fusion candidate pool"),
        ("#E3C14A", "Val-best router"),
        ("#E07A6A", "4 lead-time outputs"),
    ]
    ax.add_patch(
        FancyBboxPatch(
            (0.22, 0.10),
            10.5,
            1.05,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="white",
            edgecolor="#BBBBBB",
            lw=0.7,
            zorder=6,
        )
    )
    lx, ly = 0.38, 0.95
    for i, (c, lab) in enumerate(legend_items):
        row, col = divmod(i, 4)
        xx = lx + col * 2.55
        yy = ly - row * 0.42
        ax.add_patch(mpatches.Rectangle((xx, yy), 0.22, 0.22, facecolor=c, edgecolor="#333", lw=0.4, zorder=7))
        ax.text(xx + 0.28, yy + 0.10, lab, fontsize=6.6, va="center", zorder=7)

    ax.text(
        0.35,
        6.55,
        "SECE v2 Phase 3 — locked environ-subset + ERA5 (1940–2024 BoB)",
        fontsize=9.5,
        fontweight="bold",
        color="#2F4A63",
    )
    fig.tight_layout(pad=0.12)
    out = FIG / "sece_phase3_architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print("wrote", out)


def write_architecture_drawio() -> None:
    """Editable draw.io source aligned to the PNG layout."""
    mxfile = ET.Element("mxfile", host="app.diagrams.net", agent="generate_paper_figures.py")
    diagram = ET.SubElement(mxfile, "diagram", name="SECE Phase 3", id="sece-p3")
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1200",
        dy="800",
        grid="1",
        gridSize="10",
        guides="1",
        page="1",
        pageWidth="1600",
        pageHeight="600",
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    stages = [
        (40, 40, 180, 520, "Stage 1\nINPUT\n63-D\n(49 + ERA5)"),
        (240, 40, 200, 520, "Stage 2\nSUBSETS ×4\n3h: pos/mot/env/full\n12h+: mot/phy/env/full"),
        (460, 40, 200, 520, "Stage 3\n16 trees\n4×4 NNLS/subset"),
        (680, 40, 200, 520, "Stage 4\n7 champions\nStack/PRC/RF/LGB/XGB/CB/CB+NN"),
        (900, 40, 220, 520, "Stage 5\nFusion candidates\nNNLS / km-NNLS / residual"),
        (1140, 40, 200, 520, "Stage 6\nVal-best router\n(lowest val median km)"),
        (1360, 40, 200, 520, "Stage 7\n3h/12h/24h/48h\nΔLat, ΔLon"),
    ]
    cid = 2
    prev = "1"
    for x, y, w, h, text in stages:
        cell_id = str(cid)
        style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;align=center;"
        cell = ET.SubElement(
            root,
            "mxCell",
            id=cell_id,
            value=text,
            style=style,
            vertex="1",
            parent="1",
        )
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), as_="geometry")
        if prev != "1":
            edge_id = str(cid + 100)
            edge = ET.SubElement(
                root,
                "mxCell",
                id=edge_id,
                style="endArrow=block;html=1;",
                edge="1",
                parent="1",
                source=str(int(prev)),
                target=cell_id,
            )
            ET.SubElement(edge, "mxGeometry", relative="1", as_="geometry")
        prev = cell_id
        cid += 1

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    out = FIG / "sece_phase3_architecture.drawio"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print("wrote", out)


# ── t-SNE ───────────────────────────────────────────────────────────────────

def draw_tsne() -> None:
    cache = FIG / "tsne_era5_63d_cache.npz"
    meta_path = FIG / "tsne_era5_63d_meta.json"
    n_sample = 10_000
    perplexity = 40
    seed = 42

    df_raw = pd.read_csv(pipeline_era5.DATA_PATH)
    df = apply_physics_features(apply_qc(df_raw))
    mh = build_mh_df(df)
    train, _, _ = storm_split(mh, seed=0)
    train_df = mh[mh["SID"].isin(train)].copy()
    feat_cols = pipeline_era5.get_feature_cols(train_df)
    X = train_df[feat_cols].values
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)
    X = StandardScaler().fit_transform(X)

    rng = np.random.RandomState(seed)
    if len(train_df) > n_sample:
        idx = rng.choice(len(train_df), size=n_sample, replace=False)
    else:
        idx = np.arange(len(train_df))
    sub = train_df.iloc[idx].copy()
    Xs = X[idx]

    if cache.exists() and meta_path.exists():
        z = np.load(cache)
        emb = z["emb"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kl = meta.get("kl", float("nan"))
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            max_iter=1000,
        )
        emb = tsne.fit_transform(Xs)
        kl = float(tsne.kl_divergence_)
        np.savez_compressed(cache, emb=emb)
        meta_path.write_text(json.dumps({"kl": kl, "n": int(len(sub)), "perplexity": perplexity}), encoding="utf-8")

    speed = sub["geo_speed_kmh"].fillna(0).values
    lat = sub["LAT"].values
    if "dLAT" in sub.columns:
        disp = np.sqrt(sub["dLAT"].fillna(0).values ** 2 + sub["dLON"].fillna(0).values ** 2)
    else:
        disp = np.sqrt(sub["dLAT_3h"].fillna(0).values ** 2 + sub["dLON_3h"].fillna(0).values ** 2)
    nature = sub["NATURE"].fillna("NR").astype(str)

    def speed_cat(v):
        if v < 5:
            return "Slow (<5 kts)"
        if v < 12:
            return "Moderate (5–11 kts)"
        if v < 22:
            return "Fast (12–21 kts)"
        return "Very fast (>22 kts)"

    def lat_cat(v):
        if v < 10:
            return "0–10°N"
        if v < 15:
            return "10–15°N"
        if v < 20:
            return "15–20°N"
        if v < 25:
            return "20–25°N"
        return "25°N+"

    speed_labels = np.array([speed_cat(v) for v in speed])
    lat_labels = np.array([lat_cat(v) for v in lat])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    })
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"t-SNE Projection of {len(feat_cols)}-Dimensional Training Feature Space\n"
        f"(n = {len(sub):,} training samples | KL = {kl:.3f} | perplexity = {perplexity})",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )

    nat_u = nature.str.upper().str.strip()
    nature_colors = {"TS": "#4C78A8", "NR": "#E45756", "DS": "#54A24B", "MX": "#F58518", "ET": "#B279A2", "SS": "#9D755D"}
    ax = axes[0, 0]
    plotted = np.zeros(len(sub), dtype=bool)
    for code in ["TS", "NR", "DS", "MX", "ET", "SS"]:
        m = nat_u.eq(code)
        if not m.any():
            continue
        plotted |= m.values
        ax.scatter(
            emb[m, 0],
            emb[m, 1],
            s=8,
            alpha=0.55,
            c=nature_colors.get(code, "#888"),
            label=f"{code} (n={int(m.sum())})",
        )
    other = ~plotted
    if other.any():
        ax.scatter(emb[other, 0], emb[other, 1], s=8, alpha=0.4, c="#AAAAAA", label=f"Other (n={other.sum()})")
    ax.set_title("(a) Storm nature classification", fontsize=17, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right", markerscale=2.2)
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[0, 1]
    speed_order = ["Slow (<5 kts)", "Moderate (5–11 kts)", "Fast (12–21 kts)", "Very fast (>22 kts)"]
    sc = ["#2166AC", "#92C5DE", "#F4A582", "#B2182B"]
    for lab, col in zip(speed_order, sc):
        m = speed_labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.55, c=col, label=lab)
    ax.set_title("(b) Storm translation speed", fontsize=17, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right", markerscale=2.2)
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1, 0]
    lat_order = ["0–10°N", "10–15°N", "15–20°N", "20–25°N", "25°N+"]
    lc = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(lat_order)))
    for lab, col in zip(lat_order, lc):
        m = lat_labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.55, c=[col], label=lab)
    ax.set_title("(c) Latitude band", fontsize=17, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right", markerscale=2.2)
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1, 1]
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=disp, s=8, alpha=0.65, cmap="viridis")
    ax.set_title(
        r"(d) Step displacement magnitude $\sqrt{\Delta lat^2 + \Delta lon^2}$",
        fontsize=17,
        fontweight="bold",
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Displacement magnitude [°]", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    ax.grid(True, ls=":", alpha=0.4)

    for ax in axes.ravel():
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "tsne_era5_63d.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ── LAILA trajectories ──────────────────────────────────────────────────────

def draw_laila_trajectories(*, layout: str = "grid") -> None:
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    class ESRIWorldImagery(cimgt.GoogleWTS):
        def _image_url(self, tile):
            x, y, z = tile
            return (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"
                f"{z}/{y}/{x}"
            )

    traj = pd.read_csv(TRAJ_CSV)
    traj = traj[(traj["SID"] == CASE_LAILA) & (traj["horizon"].isin(TRAJ_HORIZONS))].copy()
    traj["ISO_TIME"] = pd.to_datetime(traj["ISO_TIME"])
    traj = traj.sort_values(["horizon", "ISO_TIME"])
    storm_name = traj["NAME"].iloc[0] if len(traj) else "LAILA (2010)"

    models = [
        ("sece_v2_phase3", "SECE Phase 3", "#E53935"),
        ("persistence_residual_cascade", "PRC", "#FDD835"),
        ("cbplusmotionnn", "CB+MotionNN", "#1E88E5"),
    ]

    n_panels = len(TRAJ_HORIZONS)
    if layout == "grid" and n_panels == 4:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10.4, 8.6),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        axes = list(axes.ravel())
        suptitle_y = 0.98
    else:
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(8.5, 4.8 * n_panels),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        if n_panels == 1:
            axes = [axes]
        suptitle_y = 0.995

    fig.suptitle(
        f"Trajectory Validation — {storm_name}: Held-out seed 0 vs IBTrACS",
        fontsize=13 if layout == "grid" else 14,
        fontweight="bold",
        y=suptitle_y,
    )

    # LAILA landfall sector; west edge 77.5 so 48h preds are not clipped (lon can be ~78).
    extent = [77.5, 93.5, 8.5, 19.0]
    imagery = ESRIWorldImagery()
    panel_fs = 9 if layout == "grid" else 10
    title_fs = 10 if layout == "grid" else 11

    for ax, horizon in zip(axes, TRAJ_HORIZONS):
        sub = traj[traj["horizon"] == horizon]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_image(imagery, 6)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="white", alpha=0.55, linestyle="--")
        gl.top_labels = gl.right_labels = False

        olat, olon = sub["origin_lat"].values, sub["origin_lon"].values
        true_lat, true_lon = sub["true_lat"].values, sub["true_lon"].values

        # Storm center at each forecast time (IBTrACS).
        ax.plot(
            olon,
            olat,
            transform=ccrs.PlateCarree(),
            color="white",
            ls="-",
            lw=1.6,
            alpha=0.85,
            zorder=3,
        )
        # Verified position H hours ahead (direct multi-horizon target).
        ax.plot(
            true_lon,
            true_lat,
            transform=ccrs.PlateCarree(),
            color="white",
            ls="--",
            lw=2.0,
            marker="o",
            ms=3,
            markevery=max(1, len(sub) // 12),
            zorder=5,
        )

        legend_lines = []
        ytrue = np.column_stack([true_lat - olat, true_lon - olon])
        for slug, label, color in models:
            plat, plon = sub[f"{slug}_lat"].values, sub[f"{slug}_lon"].values
            ax.plot(
                plon,
                plat,
                transform=ccrs.PlateCarree(),
                color=color,
                lw=2.0,
                marker="o",
                ms=3,
                markevery=max(1, len(sub) // 10),
                zorder=4,
            )
            ypred = np.column_stack([plat - olat, plon - olon])
            err = track_errors_km(olat, olon, ytrue, ypred)
            med = float(np.median(err)) if len(err) else float("nan")
            legend_lines.append(f"{label}: {med:.1f} km")

        ax.text(
            0.02,
            0.02,
            "\n".join(legend_lines),
            transform=ax.transAxes,
            fontsize=panel_fs,
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
            zorder=10,
        )
        ax.text(
            0.02,
            0.98,
            f"{chr(ord('a') + TRAJ_HORIZONS.index(horizon))}. {horizon} lead time",
            transform=ax.transAxes,
            fontsize=title_fs,
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEEEE", alpha=0.9),
            zorder=10,
        )

    rect = [0, 0, 1, 0.96] if layout == "grid" else [0, 0, 1, 0.98]
    fig.tight_layout(rect=rect)
    out_png = FIG / "case_study_laila_trajectories.png"
    out_pdf = FIG / "case_study_laila_trajectories.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out_png)
    print("wrote", out_pdf)

    waf = ROOT / "Paper Writing" / "WAF"
    if waf.is_dir():
        for name in ("fig07laila.png", "fig07laila.pdf"):
            dest = waf / name
            src = out_png if name.endswith(".png") else out_pdf
            dest.write_bytes(src.read_bytes())
            print("wrote", dest)


# Reference axes for BENCHMARK_LABEL_XY (tuned on 3 h held-out figure).
_BENCH_REF_XLIM = (5.35, 11.72)
_BENCH_REF_YLIM = (-1.82, 1.68)


def _bench_label_norm(short: str) -> tuple[float, float, str]:
    lx, ly, ha = BENCHMARK_LABEL_XY[short]
    rx = (_BENCH_REF_XLIM[1] - _BENCH_REF_XLIM[0]) or 1.0
    ry = (_BENCH_REF_YLIM[1] - _BENCH_REF_YLIM[0]) or 1.0
    return (lx - _BENCH_REF_XLIM[0]) / rx, (ly - _BENCH_REF_YLIM[0]) / ry, ha


def _resolve_label_ys(target_ys: list[float], ylim: tuple[float, float], min_sep: float) -> list[float]:
    """Keep label y near each point but enforce minimum vertical separation."""
    lo, hi = ylim
    ys = list(target_ys)
    for _ in range(8):
        changed = False
        for i in range(1, len(ys)):
            if ys[i - 1] - ys[i] < min_sep:
                ys[i] = ys[i - 1] - min_sep
                changed = True
        for i in range(len(ys) - 2, -1, -1):
            if ys[i] - ys[i + 1] < min_sep:
                ys[i] = ys[i + 1] + min_sep
                changed = True
        if not changed:
            break
    margin = 0.04 * (hi - lo)
    for i, y in enumerate(ys):
        ys[i] = min(hi - margin, max(lo + margin, y))
    return ys


def _compact_benchmark_label_layout(
    ann_rows: list[tuple[str, str, float, float, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[tuple[str, str, float, float, float], float, float, str]]:
    """
    Horizontal leaders only (label y = point y) so lines do not cross in tight x-clusters.
    Uses up to three right-margin columns when accuracy values are too close vertically.
    """
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]
    min_sep = 0.052 * span_y
    px_vals = [r[3] for r in ann_rows]
    xr_data = max(px_vals) - min(px_vals) or 1.0
    cx = float(np.median(px_vals))

    main: list[tuple[str, str, float, float, float]] = []
    outliers: list[tuple[str, str, float, float, float]] = []
    for row in ann_rows:
        if abs(row[3] - cx) > 0.38 * xr_data:
            outliers.append(row)
        else:
            main.append(row)

    main_sorted = sorted(main, key=lambda r: r[4], reverse=True)
    columns: list[list[tuple[str, str, float, float, float]]] = []
    for row in main_sorted:
        placed = False
        for col in columns:
            if all(abs(row[4] - other[4]) >= min_sep for other in col):
                col.append(row)
                placed = True
                break
        if not placed:
            columns.append([row])

    col_x = [xlim[1] - (0.07 + 0.19 * i) * span_x for i in range(len(columns))]
    out: list[tuple[tuple[str, str, float, float, float], float, float, str]] = []
    for col_idx, col in enumerate(columns):
        lx = col_x[min(col_idx, len(col_x) - 1)]
        for row in col:
            _short, _fam, _mean, _px, py = row
            out.append((row, lx, py, "right"))

    for row in sorted(outliers, key=lambda r: r[4], reverse=True):
        _short, _fam, _mean, px, py = row
        if px >= cx:
            lx = min(xlim[1] - 0.04 * span_x, px + 0.06 * xr_data)
            ha = "left"
        else:
            lx = max(xlim[0] + 0.04 * span_x, px - 0.06 * xr_data)
            ha = "right"
        out.append((row, lx, py, ha))

    return out


def _stats_rows_from_json(recs: list[dict]) -> list[tuple[str, str, str, float, float]]:
    return [
        (r["label"], r["family"], r.get("full", r["label"]), r["mean_km"], r["median_km"])
        for r in recs
    ]


def _annotate_benchmark_point(
    ax,
    short: str,
    fam: str,
    mean: float,
    px: float,
    py: float,
    lx: float,
    ly: float,
    ha: str,
    fs: float,
    compact: bool,
    error_mult,
) -> None:
    bbox_fc = "#FFF6D8" if fam == "SECE" else "white"
    bbox_ec = "#C48A00" if fam == "SECE" else "#7A7A7A"
    ax.annotate(
        f"{short} ({error_mult(mean)})",
        xy=(px, py),
        xytext=(lx, ly),
        textcoords="data",
        fontsize=fs,
        ha=ha,
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.38" if compact else "round,pad=0.42",
            fc=bbox_fc,
            ec=bbox_ec,
            lw=0.7,
        ),
        clip_on=False,
        arrowprops=dict(
            arrowstyle="-",
            color="#5A5A5A",
            lw=0.7 if compact else 0.8,
            shrinkA=5,
            shrinkB=6,
        ),
        zorder=10 + (1 if fam == "SECE" else 0),
    )


def plot_benchmark_on_ax(
    ax,
    stats_rows: list[tuple[str, str, str, float, float]],
    *,
    panel_title: str | None = None,
    compact: bool = False,
    fixed_3h_limits: bool = False,
    fixed_12h_labels: bool = False,
    fixed_24h_labels: bool = False,
    fixed_48h_labels: bool = False,
) -> list:
    """Same layout as benchmark_3h_heldout.png; label slots scale with panel limits."""
    y_scale = 2.5
    group_med = float(np.median([r[4] for r in stats_rows if r[1] == "Tree-based"]))
    sece_mean = next(r[3] for r in stats_rows if r[0] == "SECE Phase 3")

    def accuracy(median: float) -> float:
        return y_scale * (group_med / median - 1.0)

    def error_mult(mean: float) -> str:
        return f"{mean / sece_mean:.2f}x"

    scale = 0.38 if compact else 1.0
    fs = 8.2 if compact else 17.0
    legend_handles = []
    seen: set[str] = set()

    draw_order = sorted(
        stats_rows,
        key=lambda r: {"Random Forest": 0, "Stacking Ens.": 1, "SECE Phase 3": 2}.get(r[0], 3),
    )

    xs, ys = [], []
    for short, fam, _full, mean, med in draw_order:
        st = FAMILY_STYLE[fam]
        x, y = mean, accuracy(med)
        ox, oy = BENCHMARK_DISPLAY_OFFSET.get(short, (0.0, 0.0))
        px, py = x + ox, y + oy
        xs.append(px)
        ys.append(py)
        z_mark = {"SECE Phase 3": 8, "Stacking Ens.": 7, "Random Forest": 6}.get(short, st["z"])
        ax.scatter(
            [px],
            [py],
            marker=st["marker"],
            s=st["s"] * scale,
            c=st["color"],
            edgecolors=st["edge"],
            linewidths=0.85 if not compact else 0.65,
            zorder=z_mark,
        )
        if fam not in seen:
            seen.add(fam)
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=st["marker"],
                    color="none",
                    markerfacecolor=st["color"],
                    markeredgecolor=st["edge"],
                    markersize=(9 if st["marker"] != "*" else 14) * (0.55 if compact else 1.0),
                    label=fam,
                )
            )

    if fixed_3h_limits:
        xlim = _BENCH_REF_XLIM
        ylim = _BENCH_REF_YLIM
    elif fixed_12h_labels:
        xlim = _BENCH_12H_XLIM
        ylim = _BENCH_12H_YLIM
    elif fixed_24h_labels:
        xlim = _BENCH_24H_XLIM
        ylim = _BENCH_24H_YLIM
    elif fixed_48h_labels:
        xlim = _BENCH_48H_XLIM
        ylim = _BENCH_48H_YLIM
    else:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xr = xmax - xmin or 1.0
        yr = ymax - ymin or 0.12
        yr = max(yr, 0.38)
        ymid = 0.5 * (ymin + ymax)
        ymin, ymax = ymid - yr / 2, ymid + yr / 2
        right_pad = 0.42 * xr if compact else 0.24 * xr
        xlim = (xmin - 0.06 * xr, xmax + right_pad)
        ylim = (ymin - 0.12 * yr, ymax + 0.12 * yr)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]

    ann_rows: list[tuple[str, str, float, float, float]] = []
    for short, fam, _full, mean, med in draw_order:
        ox, oy = BENCHMARK_DISPLAY_OFFSET.get(short, (0.0, 0.0))
        px = mean + ox
        py = accuracy(med) + oy
        ann_rows.append((short, fam, mean, px, py))

    if fixed_3h_limits:
        for short, fam, mean, px, py in ann_rows:
            xn, yn, ha = _bench_label_norm(short)
            lx = xlim[0] + xn * span_x
            ly = ylim[0] + yn * span_y
            _annotate_benchmark_point(
                ax, short, fam, mean, px, py, lx, ly, ha, fs, compact, error_mult
            )
    elif fixed_12h_labels:
        for short, fam, mean, px, py in ann_rows:
            lx, ly, ha = BENCHMARK_12H_LABEL_XY[short]
            _annotate_benchmark_point(
                ax, short, fam, mean, px, py, lx, ly, ha, fs, compact, error_mult
            )
    elif fixed_24h_labels:
        for short, fam, mean, px, py in ann_rows:
            lx, ly, ha = BENCHMARK_24H_LABEL_XY[short]
            _annotate_benchmark_point(
                ax, short, fam, mean, px, py, lx, ly, ha, fs, compact, error_mult
            )
    elif fixed_48h_labels:
        for short, fam, mean, px, py in ann_rows:
            lx, ly, ha = BENCHMARK_48H_LABEL_XY[short]
            _annotate_benchmark_point(
                ax, short, fam, mean, px, py, lx, ly, ha, fs, compact, error_mult
            )
    else:
        layouts = _compact_benchmark_label_layout(ann_rows, xlim, ylim)
        for row, lx, ly, ha in layouts:
            short, fam, mean, px, py = row
            _annotate_benchmark_point(
                ax,
                short,
                fam,
                mean,
                px,
                py,
                lx,
                ly,
                ha,
                fs,
                compact,
                error_mult,
            )

    ax.axhline(0.0, color="#888888", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel("Mean prediction error (km)" if compact else "Mean Prediction Error (km)")
    ax.set_ylabel("Accuracy index (normalised)" if compact else "Accuracy (normalised)")
    if panel_title:
        ax.set_title(panel_title, fontsize=11 if compact else 22, fontweight="bold", pad=6 if compact else 12)
    ax.grid(True, ls=":", alpha=0.55 if not compact else 0.35)
    ax.set_axisbelow(True)
    return legend_handles


# ── 3h benchmark scatter ────────────────────────────────────────────────────

def draw_benchmark_scatter(horizon: str, out_name: str) -> None:
    stats = _benchmark_stats_horizon(horizon)
    ranked = sorted(stats, key=lambda r: (r[4], r[3]))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
    })
    fig, ax = plt.subplots(figsize=(14.8, 8.4))

    h_title = "3-Hour" if horizon == "3h" else horizon
    legend_handles = plot_benchmark_on_ax(
        ax,
        stats,
        panel_title=(
            f"{h_title} Horizon: Accuracy vs. Mean Prediction Error\n"
            "Across All Benchmarked Architectures (held-out ERA5 Phase 3)"
        ),
        compact=False,
        fixed_3h_limits=(horizon == "3h"),
    )
    ax.legend(
        handles=legend_handles,
        title="Model Family",
        loc="upper right",
        framealpha=0.95,
        fontsize=17,
        title_fontsize=18,
    )
    fig.tight_layout()
    out = FIG / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.10)
    plt.close(fig)
    print("wrote", out)

    summary = [
        {
            "rank": i + 1,
            "label": row[0],
            "family": row[1],
            "mean_km": round(row[3], 3),
            "median_km": round(row[4], 3),
        }
        for i, row in enumerate(ranked)
    ]
    stats_path = FIG / out_name.replace(".png", "_stats.json")
    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _benchmark_stats_horizon(horizon: str) -> list[tuple[str, str, str, float, float]]:
    merge_keys = ["SID", "ISO_TIME", "LAT", "LON"]
    parts = []
    df_raw = pd.read_csv(pipeline_era5.DATA_PATH)
    df = apply_physics_features(apply_qc(df_raw))
    mh_df = build_mh_df(df)
    for seed in HELDOUT_SEEDS:
        _, _, test_storms = storm_split(mh_df, seed=seed)
        base = df[df["SID"].isin(test_storms)] if horizon == "3h" else mh_df[mh_df["SID"].isin(test_storms)]
        pred = pd.read_csv(PREDS_DIR / f"seed{seed}_predictions_{horizon}.csv")
        parts.append(_prepare_eval_frame(base, pred, merge_keys))
    frame = pd.concat(parts, ignore_index=True)
    names = extract_model_names(frame.columns)
    rows = []
    for full, short, fam in BENCHMARK_MODELS:
        if full not in names:
            raise RuntimeError(f"Missing model {full} at {horizon}")
        err, _ = sample_errors_for_model(frame, horizon, full)
        rows.append((short, fam, full, float(np.mean(err)), float(np.median(err))))
    return rows


def draw_benchmark_3h() -> None:
    draw_benchmark_scatter("3h", "benchmark_3h_heldout.png")


def write_readme() -> None:
    text = textwrap.dedent(
        """\
        # NEW WAY paper figures (generated)

        | File | Description |
        |------|-------------|
        | `sece_phase3_architecture.png` | Locked SECE v2 Phase 3 pipeline (63-D ERA5 point) |
        | `sece_phase3_architecture.drawio` | Editable diagrams.net source (align/export in draw.io) |
        | `tsne_era5_63d.png` | t-SNE of 63-D training features (seed 0 train, n=10k) |
        | `case_study_laila_trajectories.png` | LAILA (2010) case study — seed 0 held-out preds |
        | `benchmark_3h_heldout.png` | 3h mean error (x) vs median-based accuracy — from task_a_test_predictions |
        | `benchmark_3h_heldout_stats.json` | Audit: mean/median km per system (matches figure labels) |

        Regenerate: `python scripts/generate_paper_figures.py`
        """
    )
    (FIG / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["arch", "tsne", "traj", "bench", "drawio"], action="append")
    args = parser.parse_args()
    only = set(args.only or [])

    FIG.mkdir(parents=True, exist_ok=True)
    run_all = not only

    if run_all or "drawio" in only or "arch" in only:
        write_architecture_drawio()
    if run_all or "arch" in only:
        draw_architecture_png()
    if run_all or "tsne" in only:
        draw_tsne()
    if run_all or "traj" in only:
        draw_laila_trajectories()
    if run_all or "bench" in only:
        draw_benchmark_3h()
    write_readme()
    print("Done.")


if __name__ == "__main__":
    main()
