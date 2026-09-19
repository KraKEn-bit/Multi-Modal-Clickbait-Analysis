# Copernicus CDS setup (ERA5)

1. Create an account: https://cds.climate.copernicus.eu
2. Accept ERA5 dataset licences on the dataset pages.
3. Create **either** `NEW WAY\\.cdsapirc` **or** `%USERPROFILE%\\.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-personal-access-token>
```

4. `pip install cdsapi xarray netCDF4`

5. **Accept dataset licences** (profile ticks are not enough). Open both pages, Download tab, accept licences:

- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download

New CDS (2024+) uses a personal access token as `key`, not `uid:key`.

If download fails with 401, the `.cdsapirc` is missing or the key is old.
