# Data Quality Report

_report_schema_version: `1.0.0`, script_version: `1.0.0`_

## 1. Summary

| Metric | train | test |
|---|---:|---:|
| rows | 99955 | 57185 |
| columns | 21 | 20 |
| polygon_count | 39 | 78 |
| primary_ndvi_finite | 30520 | 17641 |
| duplicate_keys | 0 | 0 |
| gaps (is_synthetic_gap==True) | — | 3112 |

## 2. Schema

| file | column | dtype |
|---|---|---|
| train | `anon_polygon_id` | `string` |
| train | `date` | `string` |
| train | `s2_ndvi` | `float64` |
| train | `s2_evi` | `float64` |
| train | `s2_ndwi` | `float64` |
| train | `landsat_ndvi` | `float64` |
| train | `landsat_evi` | `float64` |
| train | `landsat_ndwi` | `float64` |
| train | `modis_ndvi` | `float64` |
| train | `modis_evi` | `float64` |
| train | `era5_temp_c` | `float64` |
| train | `era5_precip_mm` | `float64` |
| train | `year` | `int64` |
| train | `primary_ndvi` | `float64` |
| train | `doy` | `int64` |
| train | `ndvi_climatology_mean` | `float64` |
| train | `ndvi_climatology_std` | `float64` |
| train | `ndvi_zscore` | `float64` |
| train | `n_reference_years` | `int64` |
| train | `status` | `object` |
| train | `crop_type` | `string` |
| test | `anon_polygon_id` | `string` |
| test | `date` | `string` |
| test | `s2_ndvi` | `float64` |
| test | `s2_evi` | `float64` |
| test | `s2_ndwi` | `float64` |
| test | `landsat_ndvi` | `float64` |
| test | `landsat_evi` | `float64` |
| test | `landsat_ndwi` | `float64` |
| test | `modis_ndvi` | `float64` |
| test | `modis_evi` | `float64` |
| test | `era5_temp_c` | `float64` |
| test | `era5_precip_mm` | `float64` |
| test | `year` | `float64` |
| test | `primary_ndvi` | `float64` |
| test | `doy` | `float64` |
| test | `ndvi_climatology_mean` | `float64` |
| test | `ndvi_climatology_std` | `float64` |
| test | `n_reference_years` | `float64` |
| test | `is_synthetic_gap` | `bool` |
| test | `crop_type` | `string` |

## 3. Missingness

Top-15 columns by missing rate (test):

| column | missing | missing_rate | finite | finite_rate |
|---|---:|---:|---:|---:|
| `modis_ndvi` | 53228 | 0.9308 | 3957 | 0.0692 |
| `modis_evi` | 53228 | 0.9308 | 3957 | 0.0692 |
| `s2_ndvi` | 49370 | 0.8633 | 7815 | 0.1367 |
| `s2_evi` | 49370 | 0.8633 | 7815 | 0.1367 |
| `s2_ndwi` | 49370 | 0.8633 | 7815 | 0.1367 |
| `landsat_evi` | 48943 | 0.8559 | 8242 | 0.1441 |
| `landsat_ndvi` | 48942 | 0.8559 | 8243 | 0.1441 |
| `landsat_ndwi` | 48942 | 0.8559 | 8243 | 0.1441 |
| `primary_ndvi` | 39544 | 0.6915 | 17641 | 0.3085 |
| `ndvi_climatology_mean` | 39544 | 0.6915 | 17641 | 0.3085 |
| `ndvi_climatology_std` | 39544 | 0.6915 | 17641 | 0.3085 |
| `era5_temp_c` | 9183 | 0.1606 | 48002 | 0.8394 |
| `era5_precip_mm` | 9183 | 0.1606 | 48002 | 0.8394 |
| `year` | 3112 | 0.0544 | 54073 | 0.9456 |
| `doy` | 3112 | 0.0544 | 54073 | 0.9456 |

## 4. Gaps

| run length | count |
|---|---:|
| 1 | 2827 |
| 2 | 136 |
| 3 | 3 |
| 4+ | 1 |

Columns **always missing** in gap rows: `s2_ndvi`, `s2_evi`, `s2_ndwi`, `landsat_ndvi`, `landsat_evi`, `landsat_ndwi`, `modis_ndvi`, `modis_evi`, `era5_temp_c`, `era5_precip_mm`, `year`, `primary_ndvi`, `doy`, `ndvi_climatology_mean`, `ndvi_climatology_std`, `n_reference_years`

## 5. Polygon overlap

- train polygons: **39**
- test polygons (known): **39**
- test polygons (unseen): **39**
- gaps on known polygons: **464**
- gaps on unseen polygons: **2648**

## 6. Target-source hierarchy (S2 → Landsat → MODIS)

**train**: checked=30520, match=30520, match_rate=1.0000, max_abs_mismatch=0.0
**test**: checked=17641, match=17641, match_rate=1.0000, max_abs_mismatch=0.0

## 7. Warnings

_No warnings. All counts match the reference._
