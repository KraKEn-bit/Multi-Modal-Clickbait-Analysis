"""Build the AMS WAF manuscript from Journal_Paper.tex."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Paper Writing" / "Journal_Paper.tex"
DST = ROOT / "Paper Writing" / "WAF" / "waf_manuscript.tex"
CSV_ARCH = ROOT / "docs" / "paper_writeup_data" / "02_architecture_hyperparameters.csv"

BIB_LABELS = {
    "knapp2010ibtracs": "Knapp et al.(2010)",
    "gahtan2024ibtracs": "Gahtan et al.(2024)",
    "worldbank2007sidr": "World Bank(2007)",
    "paul2013post": "Paul(2013)",
    "ifrc2020amphan": "IFRC(2020)",
    "worldbank2023mocha": "World Bank(2023)",
    "gomez2026tcbench": "Gomez et al.(2026)",
    "lian2020novel": "Lian et al.(2020)",
    "alemany2019predicting": "Alemany et al.(2019)",
    "giffard2020tropical": "Giffard-Roisin et al.(2020)",
    "kar2021tropical": "Kar and Banerjee(2021)",
    "goerss2004tropical": "Goerss(2004)",
    "ganaie2022ensemble": "Ganaie et al.(2022)",
    "grinsztajn2022tree": "Grinsztajn et al.(2022)",
    "neumann1972alternative": "Neumann(1972)",
    "dube2009storm": "Dube et al.(2009)",
    "aberson1998five": "Aberson(1998)",
    "goerss2000tropical": "Goerss(2000)",
    "hossain2021track": "Hossain et al.(2021)",
    "rahman2025tropical": "Rahman et al.(2025)",
    "kumar2024machine": "Kumar et al.(2024)",
    "qu2025accurate": "Qu et al.(2025)",
    "chen2016xgboost": "Chen and Guestrin(2016)",
    "ke2017lightgbm": "Ke et al.(2017)",
    "prokhorenkova2018catboost": "Prokhorenkova et al.(2018)",
    "hersbach2023era5": "Hersbach et al.(2023)",
    "krishnamurti2010ann": "Krishnamurti et al.(2010)",
    "pan2024guangdong": "Pan(2024)",
    "breiman2001rf": "Breiman(2001)",
    "maaten2008tsne": "van der Maaten and Hinton(2008)",
    "hochreiter1997lstm": "Hochreiter and Schmidhuber(1997)",
    "cho2014gru": "Cho et al.(2014)",
    "graves2005blstm": "Graves and Schmidhuber(2005)",
    "vaswani2017attention": "Vaswani et al.(2017)",
    "cawley2010": "Cawley and Talbot(2010)",
    "efron1979": "Efron(1979)",
    "wilcoxon1945": "Wilcoxon(1945)",
    "bonferroni1936": "Bonferroni(1936)",
    "sinnott1984": "Sinnott(1984)",
    "gneiting2007": "Gneiting and Raftery(2007)",
}

PREAMBLE = r"""% AMS Weather and Forecasting manuscript (submission / peer-review format)
% Class: official AMS LaTeX Template v6.1 (draft mode: 12 pt, ~1.5 line spacing,
% 1-inch-class margins, line numbers). Do not use the [twocol] option for submission.
%
% Compile twice: pdflatex waf_manuscript.tex
%
% Figure files sit in this folder (no directory paths in \includegraphics).
\documentclass{ametsocv6.1}

% Packages already loaded by ametsocv6.1.cls: graphicx, amsmath, natbib, url,
% setspace, lineno, xcolor, newtxtext/newtxmath. Do not reload them.
\usepackage{booktabs}
\usepackage{multirow}

\newcommand{\sece}{SECE}
\newcommand{\bob}{Bay of Bengal}
\newcommand{\era}{ERA5}
\newcommand{\ibtracs}{IBTrACS}

\title{ERA5-Augmented Ensemble Learning for Multi-Horizon Tropical Cyclone
Track Prediction over the Bay of Bengal: A Leakage-Aware Benchmark}

\authors{
Author One,\aff{a}\correspondingauthor{Author One, [email]}
and Author Two\aff{a}
}

\affiliation{
\aff{a} [Department, Institution, City, Country]
}

\abstract{
Accurate short- to medium-range cyclone track forecasting is a foundational input
to early warning over the Bay of Bengal, a basin that combines complex monsoon-season
steering with some of the highest coastal population exposure in the world. We
present a rigorously validated benchmark of eight track-prediction systems, a
proposed Subset-Expert Context-Aware Ensemble (SECE v2 Phase~3) and seven
baselines spanning tree and hybrid families, evaluated at 3, 12, 24, and 48\,h
lead times on North Indian Ocean IBTrACS storms with genesis in 1940--2024.
After quality control the modelling table contains 852 storms and 27{,}728
three-hourly forecast origins, of which 583 storms and 15{,}605 origins support
multi-horizon supervision. Forty-nine kinematic and derived predictors are
augmented with 14 ERA5 reanalysis fields collocated at the storm centre,
comprising winds at 850, 500, and 200\,hPa, mean sea level pressure, sea surface
temperature, a derived deep-layer steering wind, and vertical wind shear.
Critically, we identify and correct a subtle form of test-set leakage present in
an earlier iteration of our own development, in which architectural decisions
were guided by repeated inspection of test-set performance; we replace it with a
two-tier protocol in which nine development seeds carry all design decisions and
nine disjoint held-out seeds are evaluated exactly once. On the held-out confirm,
pooled over 442 distinct test storms, SECE attains median great-circle errors
of 4.895, 35.538, 101.186, and 246.331\,km. Paired Wilcoxon signed-rank tests
with Bonferroni correction across seven comparisons show that SECE is
significantly better than every baseline at 3 and 12\,h ($p<0.0071$),
statistically indistinguishable from the strongest stacks at 24 and 48\,h, and
never significantly worse than any baseline at any horizon. Controlled ablations
show that ERA5 features deliver genuine but horizon-dependent value,
concentrated at 48\,h on a per-storm basis, while 5$^\circ$ and 3$^\circ$ spatial
patch statistics were tested and not adopted.
}

\begin{document}
\maketitle

\statement
This study provides a leakage-corrected, statistically tested benchmark of
tree-based tropical cyclone track models for the Bay of Bengal at 3 to 48\,h
lead times. It shows where a routed ensemble is genuinely better than strong
baselines, where it is not, and where ERA5 environmental fields do and do not
reduce track error, so agencies without a full numerical weather prediction
suite can interpret machine-learning guidance honestly.

"""

LIMITATIONS = r"""
\section{Limitations}
\label{sec:limits}

\textbf{Environmental fields.} The locked model uses \era{} only at the storm
centre (14 point variables). We tested 5$^\circ$ and 3$^\circ$ spatial-patch
statistics in development; they did not improve 24--48\,h mean error enough to
justify adoption (Table~\ref{tab:abl}; Fig.~\ref{fig:ablation}), remaining
inside seed noise relative to the environ expert. Gridded or attention-based
encoders therefore remain outside this benchmark.

\textbf{Deterministic outputs.} Every system returns a single track point
without forecast spread. That omission matters most at 24--48\,h, where
Table~\ref{tab:heldout} shows overlapping leading medians and right-skewed
errors. A probabilistic formulation trained under a proper scoring
rule~\citep{gneiting2007} is future work, not a result claimed here.

\textbf{Baseline scope.} CNN-GRU~\citep{lian2020novel} and bidirectional
LSTM~\citep{graves2005blstm} were explored in development but were \emph{not}
re-trained on the locked \era{}-augmented nine-seed held-out protocol. The
confirm table is an eight-way tree comparison only; deep models are a scoping
boundary, not a measured defeat.

\textbf{Historical mixing.} Genesis 1940--2024 spans pre-satellite and modern
eras. A fixed modern holdout (section~\ref{sec:window}) supports the wide window
for track error, yet pre-1979 wind-observation sparsity limits
intensity-adjacent interpretation, and residual confounding between sample size
and observational era cannot be fully excluded.

\textbf{Attribution.} Section~\ref{sec:fi} and Fig.~\ref{fig:featimp} report
LightGBM \emph{gain} on the environ expert for held-out seed~0 only. This is
illustrative, not multi-seed SHAP or permutation testing, and not mechanistic
proof of how atmospheric fields cause track-error reductions.
"""

ACK_AND_DATA = r"""
\acknowledgments
ERA5 reanalysis fields were obtained from the Copernicus Climate Data Store
\citep{hersbach2023era5}. IBTrACS best-track data were obtained from NOAA's
National Centers for Environmental Information \citep{knapp2010ibtracs,gahtan2024ibtracs}.
The authors declare no conflicts of interest.

\datastatement
Processed modelling tables, locked held-out predictions, bootstrap and Wilcoxon
outputs, case-study trajectories, seed lists, and split counts are versioned with
the project. ERA5 is available from the Copernicus Climate Data Store
(https://doi.org/10.24381/cds.adbb2d47). IBTrACS version 4r01 is available from
NOAA NCEI (https://doi.org/10.25921/82ty-9e16). Every new figure in this
manuscript regenerates from those frozen files.

"""

FIG_TSNE = r"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.98\textwidth]{fig01tsne}
\caption{t-SNE projection \citep{maaten2008tsne} of the 63-dimensional feature
space ($n=10{,}000$ training samples, held-out seed 0). Colouring by independent
meteorological attributes. Coherent separation indicates that the engineered
features encode physically distinct regimes, which is the premise of the
subset-expert design.}
\label{fig:tsne}
\end{figure}
"""

FIG_ARCH = r"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.98\textwidth]{fig02arch}
\caption{Locked SECE v2 Phase~3 pipeline. Input is a 63-dimensional vector
(49 track/physics features plus 14 point \era{} fields). Four subset experts,
each an NNLS fusion of four tree algorithms, enter a champion pool with seven
standalone baselines; a validation-only router selects the fusion candidate with
the lowest validation median error at each horizon.}
\label{fig:arch}
\end{figure}
"""

FIG_ABLATION = r"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.98\textwidth]{fig03ablation}
\caption{Development ablation trend from
\texttt{full\_experiment\_summary\_table.csv}. The $y$-axis is the mean of SECE
test medians across nine seeds (km). Configurations on the $x$-axis are sequential
interventions on the development seeds except the last point, which is the
independent held-out confirm on nine disjoint seeds (not a sixth architecture).
Each panel uses its own vertical scale so that 3\,h changes of a few tenths of a
kilometre remain visible. Point \era{} helps at 48\,h; the environ expert is the
locked 24\,h compromise; spatial 5$^\circ$/3$^\circ$ patches do not justify
adoption over the environ expert.}
\label{fig:ablation}
\end{figure}
"""

FIG_BENCH = r"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.98\textwidth]{fig04bench4h}
\caption{Held-out four-horizon benchmark for the eight locked systems.
Each point is computed from the pooled nine-seed held-out prediction export
(\texttt{results/sece\_era5\_environ\_heldout/task\_a\_test\_predictions/}),
audited in \texttt{benchmark\_*h\_heldout\_stats.json}. The horizontal axis is
\emph{mean} great-circle error (km) over all scored test origins; parenthetical
multipliers are that system's mean divided by the SECE mean. The vertical axis is
an accuracy index derived from \emph{median} error, normalised so that the
tree-family median sits at zero. Table~\ref{tab:heldout} reports the corresponding
medians with bootstrap confidence intervals. Mean and median therefore answer
different questions and are not interchangeable.}
\label{fig:bench}
\end{figure}
"""

FIG_ERA5HIST = r"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.98\textwidth]{fig05era5hist}
\caption{Per-storm LightGBM diagnostic comparing track-only error with
point-\era{} error on held-out seed 0
(\texttt{05\_seed0\_lgb\_era5\_vs\_trackonly\_per\_storm.csv}).
$\Delta$ is track-only median error minus \era{}-informed median error, so
positive values mean \era{} improved that storm. Blue: zero (parity); black
dashed: median $\Delta$. Panel titles report $n_{\mathrm{improved}}/n_{\mathrm{total}}$.
This is a single-model, single-seed diagnostic, not the full SECE stack.}
\label{fig:era5hist}
\end{figure}
"""

FIG_FEATIMP = r"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.98\textwidth]{fig06featimp}
\caption{LightGBM gain share for every feature in the environ-subset expert at
24 and 48\,h, from \texttt{04\_environ\_expert\_feature\_importance.csv}.
Blue: kinematic / track-derived features; orange: \era{}-derived features,
the colour convention used throughout this manuscript. Bars are sorted by gain
share. This is held-out seed 0 only, gain-based (not SHAP or permutation
importance), and environ-expert-only (not the full SECE stack).}
\label{fig:featimp}
\end{figure}
"""

FIG_LAILA = r"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.92\textwidth]{fig07laila}
\caption{Cyclone Laila (2010) on a held-out seed. White denotes the \ibtracs{}
verified track. Each marker is a prediction issued from a distinct origin time
along the track and offset forward by the panel's lead time, so panels show direct
multi-horizon endpoints rather than a recursively rolled-out trajectory.}
\label{fig:laila}
\end{figure}
"""

ARCH_TABLE = r"""
\begin{table}[ht]
\centering
\begin{tabular}{@{}p{0.42\textwidth}p{0.52\textwidth}@{}}
\toprule
Component & Setting \\
\midrule
Locked architecture & SECE v2 Phase 3 + point ERA5 + environ expert \\
Genesis window & 1940--2024 IBTrACS NI + ERA5 point features \\
Held-out seeds & 0, 1, 2, 7, 13, 99, 123, 2024, 2026 \\
Development seeds (architecture lock only) & 3, 4, 5, 6, 8, 9, 10, 11, 14 \\
Split & storm-wise 70:15:15, seed-specific \\
3\,h SECE subsets & position, motion, environ, full \\
12/24/48\,h SECE subsets & motion, physics, environ, full \\
Subset tree algorithms & CatBoost, XGBoost, LightGBM, Random Forest $\times$ each subset \\
Subset experts & 4 subsets $\times$ 4 algorithms $=$ 16 trees, NNLS-fused to 1 subset signal \\
Champion / standalone bases & Stacking, PRC, RF, LightGBM, XGBoost, CatBoost, CB+MotionNN \\
NNLS full pool & 7 champions + 1 subset-NNLS signal (degree space, separate lat/lon weights) \\
Router (Phase 3) & hard val-best: lowest validation median km \\
3\,h extra fusion candidates & SECE NNLS full, NNLS champions, RF+STK NNLS, STK+residual LGB \\
24\,h extra fusion candidates & SECE NNLS full, NNLS champions, km-NNLS (champions, persistence-anchored) \\
48\,h extra fusion candidates & as 24\,h plus LGB+PRC NNLS and LGB+PRC km-NNLS \\
Global learning rate & 0.01 \\
Tree $n_{\mathrm{estimators}}$ (RF/XGB/LGB/CB) & 300 \\
Tree max depth / CatBoost depth & 6 \\
LightGBM \texttt{num\_leaves} & 63 \\
XGBoost subsample / colsample by tree & 0.8 / 0.8 \\
Stacking STK\_N (RF+XGB+LGB Ridge) & 150 \\
MotionNN & MLP 64-32-16, ReLU, lr 0.01, max\_iter 100, early stopping \\
PRC & persistence displacement + LightGBM residual in km, mapped back to degrees \\
km-NNLS & NNLS on (pred$-$persist)$\times$(111,111) km, then add persist \\
ERA5 environ keys & $u,v$ at 850/500/200\,hPa, MSL, SST, steer $u,v$, shear $u,v$, shear magnitude, SST missing flag \\
Environ subset also includes & DIST2LAND, LANDFALL, NEWDELHI, STORM\_SPEED, STORM\_DIR + ERA5 keys \\
Not evaluated on held-out & CNN-GRU, BLSTM, other deep models \\
\bottomrule
\end{tabular}
\caption{Locked architecture and hyperparameters from
\texttt{02\_architecture\_hyperparameters.csv}, applied uniformly across systems
unless noted. CNN-GRU and BLSTM were not re-trained on this held-out protocol.}
\label{tab:hyp}
\end{table}
"""


def convert_cites(text: str) -> str:
    # Textual "Author et al.~\cite{key}" -> \citet{key}
    text = re.sub(
        r"([A-Z][A-Za-z\-]+ et al\.)~\\cite\{([^}]+)\}",
        r"\\citet{\2}",
        text,
    )
    # Remaining parenthetical IEEE cites -> \citep
    text = text.replace(r"\cite{", r"\citep{")
    # Undo if we accidentally wrapped citet (should not happen)
    text = text.replace(r"\citepitet{", r"\citet{")
    return text


def relabel_bibitems(text: str) -> str:
    def repl(m):
        key = m.group(1)
        label = BIB_LABELS.get(key)
        if label:
            return f"\\bibitem[{label}]{{{key}}}"
        return m.group(0)

    return re.sub(r"\\bibitem\{([^}]+)\}", repl, text)


def fix_cawley(text: str) -> str:
    old = (
        r"N. J. Cawley and G. C. L. Talbot, ``On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,'' "
        r"\textit{Journal of Machine Learning Research}, vol. 11, pp. 2079--2107, 2010."
    )
    new = (
        r"G. C. Cawley and N. L. C. Talbot, ``On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,'' "
        r"\textit{Journal of Machine Learning Research}, vol. 11, no. 70, pp. 2079--2107, 2010."
    )
    if old not in text:
        raise SystemExit("Cawley bibitem text not found for replacement")
    return text.replace(old, new)


def strip_old_figure(text: str, label: str) -> str:
    lab = f"\\label{{{label}}}"
    i = text.find(lab)
    if i < 0:
        raise SystemExit(f"Could not strip figure {label}: label missing")
    start = text.rfind("\\begin{figure", 0, i)
    if start < 0:
        raise SystemExit(f"Could not strip figure {label}: begin missing")
    end = text.find("\\end{figure", i)
    if end < 0:
        raise SystemExit(f"Could not strip figure {label}: end missing")
    end = text.find("}", end) + 1
    return text[:start] + text[end:]


def replace_labeled_block(text: str, env: str, label: str, replacement: str) -> str:
    lab = f"\\label{{{label}}}"
    i = text.find(lab)
    if i < 0:
        raise SystemExit(f"Could not find label {label}")
    start = text.rfind(f"\\begin{{{env}", 0, i)
    if start < 0:
        raise SystemExit(f"Could not find begin {env} for {label}")
    end = text.find(f"\\end{{{env}", i)
    if end < 0:
        raise SystemExit(f"Could not find end {env} for {label}")
    end = text.find("}", end) + 1
    return text[:start] + replacement.strip() + text[end:]


def main() -> None:
    raw = SRC.read_text(encoding="utf-8")

    # Body starts at Introduction; keep through end.
    body = raw.split("\\section{Introduction}", 1)[1]
    body = "\\section{Introduction}\n" + body

    # Drop old two-column figure/table stars.
    body = body.replace(r"\begin{table*}[!t]", r"\begin{table}[ht]")
    body = body.replace(r"\begin{table*}[!t]", r"\begin{table}[ht]")
    body = body.replace(r"\end{table*}", r"\end{table}")
    body = body.replace(r"\begin{figure*}[!t]", r"\begin{figure}[ht]")
    body = body.replace(r"\begin{figure}[!t]", r"\begin{figure}[ht]")
    body = body.replace(r"\end{figure*}", r"\end{figure}")

    body = convert_cites(body)
    body = replace_labeled_block(body, "table", "tab:hyp", ARCH_TABLE)

    # Replace existing figures with AMS-safe filenames / new panels.
    body = strip_old_figure(body, "fig:tsne")
    body = strip_old_figure(body, "fig:arch")
    body = strip_old_figure(body, "fig:bench")
    body = strip_old_figure(body, "fig:laila")

    # Insert t-SNE after the sentence that first cites it.
    needle = "which is the premise of the subset-expert design."
    # After conversion that sentence may still be nearby; insert after first fig cite paragraph.
    mark = r"Fig.~\ref{fig:tsne}"
    idx = body.find(mark)
    if idx < 0:
        raise SystemExit("fig:tsne citation missing")
    # insert figure after the paragraph containing the cite
    para_end = body.find("\n\n", idx)
    body = body[:para_end] + "\n" + FIG_TSNE + body[para_end:]

    mark = r"Fig.~\ref{fig:arch}"
    idx = body.find(mark)
    if idx < 0:
        # architecture may be cited as figure in methodology opener
        raise SystemExit("fig:arch citation missing")
    para_end = body.find("\n\n", idx)
    body = body[:para_end] + "\n" + FIG_ARCH + body[para_end:]

    # Ablation figure after Table tab:abl
    mark = r"\label{tab:abl}"
    idx = body.find(mark)
    end_tab = body.find(r"\end{table}", idx)
    end_tab = body.find("\n", end_tab)
    body = body[: end_tab + 1] + "\n" + FIG_ABLATION + body[end_tab + 1 :]

    insert_ablation_sentence = (
        "Fig.~\\ref{fig:ablation} shows the same mean-median series as a trend "
        "across configurations so that the horizon at which each intervention "
        "helps, or fails to help, is visible at a glance.\n\n"
    )
    held = body.find(r"\subsection{Held-out evaluation}")
    body = body[:held] + insert_ablation_sentence + body[held:]

    # Four-horizon benchmark: after the sentence that currently cites fig:bench
    mark = r"Fig.~\ref{fig:bench}"
    idx = body.find(mark)
    if idx < 0:
        raise SystemExit("fig:bench citation missing")
    para_end = body.find("\n\n", idx)
    body = body[:para_end] + "\n" + FIG_BENCH + body[para_end:]

    # ERA5 histogram after diagnostic subsection intro numbers
    mark = r"\label{sec:diag}"
    idx = body.find(mark)
    # after the paragraph ending "valuable..."
    para = body.find("giffard2020tropical,goerss2004tropical}", idx)
    if para < 0:
        para = body.find(r"\citep{giffard2020tropical,goerss2004tropical}", idx)
    para_end = body.find("\n\n", para)
    body = body[:para_end] + "\n" + FIG_ERA5HIST + body[para_end:]

    # Feature importance figure before the FI table
    mark = r"\label{sec:fi}"
    idx = body.find(mark)
    para_end = body.find("\n\n", idx)
    body = body[:para_end] + "\n" + FIG_FEATIMP + body[para_end:]

    # Laila at case studies
    mark = r"\label{sec:cases}"
    idx = body.find(mark)
    para_end = body.find("\n\n", idx)
    body = body[:para_end] + "\n" + FIG_LAILA + body[para_end:]

    # Replace limitations through acknowledgments
    lim = body.find(r"\section{Limitations and Future Work}")
    if lim < 0:
        lim = body.find(r"\section{Limitations}")
    conc = body.find(r"\section{Conclusion}")
    if lim < 0 or conc < 0:
        raise SystemExit("Could not find limitations/conclusion")
    body = body[:lim] + LIMITATIONS + "\n" + body[conc:]

    # Replace reproducibility + acknowledgments with AMS commands; keep conclusion
    repro = body.find(r"\section*{Reproducibility Statement}")
    if repro < 0:
        raise SystemExit("Reproducibility block missing")
    refs = body.find(r"% REFERENCES")
    if refs < 0:
        refs = body.find(r"\begin{thebibliography}")
    body = body[:repro] + ACK_AND_DATA + body[refs:]

    # Clean old footnotesize bibliography wrapper
    body = body.replace(r"\footnotesize" + "\n" + r"\setlength{\itemsep}{0pt plus 0.3pt}" + "\n", "")
    body = body.replace(r"\setlength{\itemsep}{0pt plus 0.3pt}" + "\n", "")

    body = relabel_bibitems(body)
    body = fix_cawley(body)

    # Remove commented gomez duplicate if present is fine.
    # Drop leftover graphicspath mentions — none in body.

    # AMS: no \section*{Acknowledgments} leftover
    if "[Funding sources" in body:
        raise SystemExit("Placeholder acknowledgment still present")

    out = PREAMBLE + body
    # Ensure documentclass-safe figure names (no extra periods).
    DST.write_text(out, encoding="utf-8")
    print("wrote", DST, "chars", len(out))
    if "[?]" in out:
        print("WARNING: remaining [?] placeholders")
    else:
        print("no [?] placeholders")


if __name__ == "__main__":
    main()
