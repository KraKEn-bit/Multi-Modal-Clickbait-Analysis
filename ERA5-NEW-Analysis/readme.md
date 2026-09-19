# NEW WAY — ERA5 track (frozen old pipeline)

All new code, datasets, and results live here.
Do not modify `Final_Cyclone_Pred_Results_P-3/` datasets in place.

## Sequence (lock date window before architecture)

0. IBTrACS quality-by-era (no ERA5)
1. ERA5 download (year files) + join to tracks
2. Date-window ablation (simple LGB, DEV seeds only)
3. Lock window in `docs/date_window_decision.md`
4. Then architecture / SECE-style work

## Date windows (proposed; end year = last track year)

| Window | Genesis years |
|--------|----------------|
| Full ERA5 overlap | 1940–2024 |
| Satellite-era | 1979–2024 |
| Modern (paper-like) | 1990–2024 |

Held-out seeds from Task B are **not** used for this comparison.

## Power cut / sleep

Leave the laptop **plugged in**. Run:

```text
NEW WAY\run_overnight.bat
```

- Completed ERA5 **year** files in `datasets/era5_raw/` are never re-downloaded.
- Incomplete year `.nc.part` files are discarded and that year is retried.
- Join appends to `datasets/tracks_era5.csv` and skips SID+time already written.
- After logon (power restored), scheduled task `Research1_NEWWAY_ERA5` restarts this script if not finished.

**CDS key required** before any NetCDF arrives. Put `url` + `key` in `NEW WAY\\.cdsapirc` (see `docs/era5_cds_setup.md`). The runner waits and retries; it does not start over.
