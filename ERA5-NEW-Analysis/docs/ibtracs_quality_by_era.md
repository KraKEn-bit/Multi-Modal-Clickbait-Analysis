# IBTrACS quality by era (before ERA5 join)

Generated 2026-09-08T23:25:39Z. Diagnostic only — no modeling.

Windows use **genesis season** on the research-ready file after the same QC as the old pipeline.

## Counts

       window  year_start  year_end  n_storms  n_track_points_ready  frac_storms_bob_genesis  frac_storms_majority_missing_wmo_wind  frac_storms_majority_missing_usa_wind  frac_storms_majority_missing_newdelhi_wind  mean_frac_missing_wmo_wind  mean_frac_missing_newdelhi_wind_ready  mean_frac_irregular_dt  mean_frac_latnext_mismatch  mean_frac_track_not_main  mean_frac_nature_nr
    full_era5        1940      2024       852                 27728                 0.894366                               0.825117                               0.731221                                    0.731221                    0.817906                               0.753003                0.003323                    0.003070                  0.001174             0.067063
    satellite        1979      2024       353                 12370                 0.864023                               0.577904                               0.439093                                    0.351275                    0.562317                               0.403848                0.008021                    0.007411                  0.000000             0.161864
       modern        1990      2024       269                  9446                 0.869888                               0.453532                               0.423792                                    0.286245                    0.500208                               0.331774                0.000544                    0.000258                  0.000000             0.207934
pre_satellite        1940      1978       499                 15358                 0.915832                               1.000000                               0.937876                                    1.000000                    0.998713                               1.000000                0.000000                    0.000000                  0.002004             0.000000
     pre_era5        1842      1939       667                 18325                 0.997001                               1.000000                               1.000000                                    1.000000                    1.000000                               1.000000                0.000000                    0.000000                  0.000000             0.002999

## How to read this

- **pre_era5 (1842–1939)** cannot get ERA5. Large missing-wind fractions are expected.
- **pre_satellite (1940–1978)** has ERA5 but sparse ocean obs + weaker IBTrACS.
- **satellite (1979–2024)** is the usual climate-quality ERA5/IBTrACS era.
- **modern (1990–2024)** matches a paper-like scope; smallest sample.

IBTrACS fields used: `WMO_WIND`, `USA_WIND`, `NEWDELHI_WIND`, `TRACK_TYPE`, `NATURE`, `IFLAG` (raw),
plus research-ready `NEWDELHI_WIND_missing`, `flag_irregular_dt`, `flag_latnext_mismatch`.

A storm is “majority missing wind” if ≥50% of its IBTrACS points lack that wind field.

Date-window **model** ablation still waits on `datasets/*_era5.csv`.
