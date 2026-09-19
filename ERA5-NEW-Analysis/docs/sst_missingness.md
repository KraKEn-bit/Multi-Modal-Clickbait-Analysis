# ERA5 SST missingness (tracks_era5.csv)

Generated during `prepare_tracks_era5.py` setup. Diagnostic for modeling — not a download defect.

## Cause

ERA5 `sst` is **NaN over land**. Track points over Bangladesh/eastern India or near the coast often sit on land grid cells while u/v/msl still interpolate fine.

## Scale (pre-fill, on joined 1940–2024 rows)

| Metric | Value |
|--------|------:|
| Rows with missing raw SST | ~38% |
| All other ERA5 variables | 0% missing |

## Proximity to land (operational relevance)

Using `DIST2LAND` from the research-ready tracks:

| Subset | SST missing rate |
|--------|----------------:|
| Closest 25% to land | ~97% |
| DIST2LAND ≤ 50 km | ~83% |
| Near-land BoB (lat 15–25°N, D2L ≤ 30 km) | ~91% |
| Open ocean (DIST2LAND > 200 km) | ~11% |

Missing SST **correlates with land proximity** (~0.48 with inverse distance-to-land). This is expected for a Bangladesh landfall-focused study: the operationally important near-land cases are exactly where SST is often unavailable from ERA5 at the track point.

## Handling in NEW WAY

1. **`sst_missing`** — binary flag (same idea as `NEWDELHI_WIND_missing` in the old pipeline).
2. **`sst` fill** — nearest valid **ocean** SST cell at the same ERA5 time; month-ocean median if no finite neighbor.
3. **Ablation / modeling** — `sst_missing` included as a feature; non-SST columns median-imputed on train only.

Narrowing the date window (e.g. 1990–2024) improves IBTrACS wind quality more than SST completeness; SST gaps remain ~29% even in the modern window.
