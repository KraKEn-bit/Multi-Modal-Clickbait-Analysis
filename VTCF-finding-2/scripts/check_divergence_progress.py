"""Mann-Whitney semantic_divergence check on Phase 1 training-ready subset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._paths import load_config, resolve_path
from scripts.phase0_spike import compute_divergence_stats, label_to_binary

PHASE0_MANN_WHITNEY_P = 0.300
PHASE0_N = 50
MIN_WORDS = 5


def _word_count(text: str) -> int:
    return len(text.split()) if text and text.strip() else 0


def load_training_ready_rows(
    verified_csv: Path,
    transcripts_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter gemini rows with transcript_word_count > MIN_WORDS from metadata."""
    verified = pd.read_csv(verified_csv)
    verified["video_id"] = verified["video_id"].astype(str)

    gemini = verified[verified["summary_source"].astype(str).str.lower() == "gemini"].copy()

    transcript_counts: list[int] = []
    summary_counts: list[int] = []
    for video_id in gemini["video_id"]:
        meta_path = transcripts_dir / video_id / "metadata.json"
        transcript_count = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            transcript_count = int(meta.get("transcript_word_count", 0) or 0)

        summary_path = transcripts_dir / video_id / "summary.txt"
        summary_text = (
            summary_path.read_text(encoding="utf-8").strip()
            if summary_path.exists()
            else ""
        )
        summary_count = _word_count(summary_text)

        transcript_counts.append(transcript_count)
        summary_counts.append(summary_count)

    gemini["transcript_word_count"] = transcript_counts
    gemini["summary_word_count"] = summary_counts

    clean = gemini[gemini["transcript_word_count"] > MIN_WORDS].copy()
    return gemini, clean


def _correlation_report(
    df: pd.DataFrame,
    *,
    length_col: str,
    label: str,
) -> dict[str, float | int | None]:
    subset = df[df["human_label"] == label].copy()
    subset = subset[
        subset["semantic_divergence"].notna()
        & subset[length_col].notna()
    ]
    n = len(subset)
    if n < 3:
        return {"n": n, "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}

    x = subset[length_col].astype(float)
    y = subset["semantic_divergence"].astype(float)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "n": n,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def _print_correlation_block(
    title: str,
    df: pd.DataFrame,
    length_col: str,
) -> None:
    print(f"--- {title} vs semantic_divergence ---")
    for label in ("clickbait", "non_clickbait"):
        stats = _correlation_report(df, length_col=length_col, label=label)
        n = stats["n"]
        if stats["pearson_r"] is None:
            print(f"  {label}: n={n} (insufficient data)")
            continue
        print(
            f"  {label} (n={n}): "
            f"Pearson r={stats['pearson_r']:.4f} p={stats['pearson_p']:.4f} | "
            f"Spearman rho={stats['spearman_r']:.4f} p={stats['spearman_p']:.4f}"
        )
    print()


def main() -> None:
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    config = load_config()
    verified_csv = resolve_path(config["data"]["finding2_verified_csv"])
    transcripts_dir = resolve_path(config["data"]["transcripts_dir"])

    gemini_all, clean = load_training_ready_rows(verified_csv, transcripts_dir)

    rows = [
        {
            "video_id": str(row["video_id"]),
            "label": str(row["human_label"]),
            "semantic_divergence": row["semantic_divergence"],
            "transcript_word_count": int(row["transcript_word_count"]),
            "summary_word_count": int(row["summary_word_count"]),
        }
        for _, row in clean.iterrows()
        if pd.notna(row["semantic_divergence"])
    ]

    stats = compute_divergence_stats(rows)
    cb_vals = [
        float(r["semantic_divergence"])
        for r in rows
        if label_to_binary(r["label"]) == 1
    ]
    nc_vals = [
        float(r["semantic_divergence"])
        for r in rows
        if label_to_binary(r["label"]) == 0
    ]

    excluded_empty = len(gemini_all) - len(clean)

    print("\n=== Phase 1 Divergence Progress (training-ready) ===")
    print(f"verified_csv:              {verified_csv}")
    print(f"gemini rows:               {len(gemini_all)}")
    print(f"excluded (word_count<={MIN_WORDS}): {excluded_empty}")
    print(f"clean training-ready n:    {len(clean)}")
    print(f"  clickbait:               {len(cb_vals)}")
    print(f"  non-clickbait:           {len(nc_vals)}")
    print()
    print(f"Mean divergence clickbait:     {stats['mean_divergence_clickbait']:.4f}")
    print(f"Mean divergence non-clickbait: {stats['mean_divergence_non_clickbait']:.4f}")
    print(
        f"Direction (clickbait > non-clickbait): "
        f"{'YES' if stats['direction_correct'] else 'NO'}"
    )
    p_val = stats.get("mann_whitney_p")
    p_str = f"{p_val:.6f}" if p_val is not None else "N/A"
    print(f"Mann-Whitney U (alternative=greater): p = {p_str}")
    print()
    print("=== vs Phase 0 baseline (n=50) ===")
    print(f"Phase 0 Mann-Whitney p:    {PHASE0_MANN_WHITNEY_P:.3f}")
    print(f"Phase 1 Mann-Whitney p:    {p_str}")
    if p_val is not None:
        delta = p_val - PHASE0_MANN_WHITNEY_P
        trend = "more significant" if p_val < PHASE0_MANN_WHITNEY_P else "less significant"
        print(f"Change:                    {delta:+.6f} ({trend})")
    print()

    print("=== Length confound check (topical drift vs honesty) ===")
    for label in ("clickbait", "non_clickbait"):
        subset = clean[clean["human_label"] == label]
        print(f"{label}:")
        print(
            f"  mean summary_word_count:     "
            f"{subset['summary_word_count'].mean():.1f} "
            f"(median {subset['summary_word_count'].median():.0f})"
        )
        print(
            f"  mean transcript_word_count:  "
            f"{subset['transcript_word_count'].mean():.1f} "
            f"(median {subset['transcript_word_count'].median():.0f})"
        )
        print(
            f"  mean semantic_divergence:    "
            f"{subset['semantic_divergence'].mean():.4f}"
        )
    print()
    _print_correlation_block("summary_word_count", clean, "summary_word_count")
    _print_correlation_block("transcript_word_count", clean, "transcript_word_count")

    cb = clean[clean["human_label"] == "clickbait"]
    nc = clean[clean["human_label"] == "non_clickbait"]
    summary_gap = nc["summary_word_count"].mean() - cb["summary_word_count"].mean()
    div_gap = nc["semantic_divergence"].mean() - cb["semantic_divergence"].mean()
    nc_summary_corr = _correlation_report(clean, length_col="summary_word_count", label="non_clickbait")
    print("=== Confound interpretation hint ===")
    print(
        f"Non-clickbait summaries are {summary_gap:+.1f} words longer on average; "
        f"divergence is {div_gap:+.4f} higher for non-clickbait."
    )
    if (
        summary_gap > 0
        and div_gap > 0
        and nc_summary_corr.get("pearson_r") is not None
        and nc_summary_corr["pearson_r"] > 0
        and nc_summary_corr["pearson_p"] is not None
        and nc_summary_corr["pearson_p"] < 0.05
    ):
        print(
            "LIKELY LENGTH CONFOUND: longer non-clickbait summaries correlate with "
            "higher divergence — metric may track topical drift, not factual contradiction."
        )
    elif summary_gap > 0 and div_gap > 0:
        print(
            "PARTIAL CONFOUND SIGNAL: non-clickbait is longer and more divergent, "
            "but summary-length correlation is weak — length alone may not explain reversal."
        )
    else:
        print("Length pattern does not cleanly explain the divergence reversal.")
    print()


if __name__ == "__main__":
    main()
