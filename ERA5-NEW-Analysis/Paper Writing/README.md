# Paper writing (NEW WAY)

| File | Use |
|------|-----|
| **`Journal_Paper.tex`** | Single-file **journal** (`article`, 10pt two-column); ~**13 pp**, figures + tables + refs |
| **`Conference_Paper.tex`** | Single-file **conference** (`IEEEtran`); target **$\le$6 pp**, 4 figures + 3 tables |
| **`WAF/waf_manuscript.tex`** | **AMS WAF** (`ametsocv6.1`); upload **entire `WAF/` folder** to Overleaf (see below) |

Both compile clean under `pdflatex` (two passes, no errors). The wide architecture
diagram is a `figure*` spanning both columns; float limits are raised in each
preamble so figures settle next to the text that discusses them instead of piling
up at the end. The only warnings are the intentional `NEEDS-VERIFICATION-*`
undefined citations.
| `Journal_Paper_Draft.md` | Prose source (markdown) |
| `Conference_Paper_Draft.md` | Short prose source (markdown) |

## Overleaf setup

### Journal or IEEE conference

1. Paste `Journal_Paper.tex` or `Conference_Paper.tex` as `main.tex`.
2. Create folder **`figures`** and upload from `NEW WAY/docs/figures/` (or `Paper Writing/figures/`):
   - `sece_phase3_architecture.png`
   - `benchmark_3h_heldout.png`
   - `case_study_laila_trajectories.png`
   - `tsne_era5_63d.png` *(journal only)*

### AMS Weather and Forecasting (`WAF/`)

1. **Upload Project** → zip the whole **`NEW WAY/Paper Writing/WAF/`** directory (flat root, no subfolders for figures).
2. Set **Main document** to **`waf_manuscript.tex`**; compiler **pdfLaTeX**; recompile twice.
3. Required in project root: `ametsocV6.1.cls`, `ametsocV6.bst`, `waf_manuscript.tex`, and figure files `fig01tsne` … `fig07laila` (`.png` / `.pdf` as generated).

Journal and conference files are self-contained with embedded bibliographies; WAF uses the same reference block in `waf_manuscript.tex`.

New citations use `\cite{NEEDS-VERIFICATION-...}` placeholders only — see **`CITATION_PLACEHOLDERS.md`**.

Numbers and claims must match `docs/paper_writeup_data/` and `docs/full_experiment_summary_table.md`.

Fill in `[Author]`, affiliations, and acknowledgments before submission.
