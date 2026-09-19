# Position vs wind quality by era

Generated 2026-09-11T12:45:01Z. BoB-genesis storms (SUBBASIN=BB).

## Summary

The pre-1979 **wind/intensity** gap is large; **position** proxies are much closer after QC.

| Metric | Pre-satellite (1940–1978) | Modern (1990–2024) |
|--------|-------------------------:|-------------------:|
| IBTrACS WMO wind missing (point) | 100.0% | 48.8% |
| IBTrACS NEWDELHI wind missing (point) | 100.0% | 33.0% |
| IFLAG interpolated position (point) | 0.0% | 0.0% |
| Research-ready irregular dt (pre-QC) | 0.00% | 0.04% |
| Research-ready lat/lon mismatch (pre-QC) | 0.00% | 0.03% |
| QC-passing std(dLAT) deg | 0.172 | 0.218 |
| QC-passing std(dLON) deg | 0.310 | 0.332 |

## Interpretation

- **Wind/intensity observations** are far sparser pre-1979 (expected; already in `ibtracs_quality_by_window.csv`).
- **Position/timing proxies** (IFLAG interpolation, irregular dt, lat/lon mismatch, QC kinematic spread) are **much closer** between eras.
- **ERA5 along-track wind spread** differs by era because of **meteorology**, not reanalysis quality (ERA5 is homogeneous 1940–2024).
- Extra pre-1979 storms mainly add **trajectory + ERA5 environment** examples for long-lead track prediction,
  even when IBTrACS wind reports are missing.

Full table: `position_vs_wind_quality_by_era.csv`
