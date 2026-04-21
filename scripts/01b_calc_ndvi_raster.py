"""
calc_ndvi_raster.py
===================
Computes a full NDVI raster from LAZ point cloud NIR and Red values.
Rasterizes point cloud colour values onto a regular grid and computes
NDVI = (NIR - Red) / (NIR + Red) per pixel.

Usage:
    python scripts/01b_calc_ndvi_raster.py
    python scripts/01b_calc_ndvi_raster.py --resolution 0.5 --class-filter 5
    python scripts/01b_calc_ndvi_raster.py --resolution 0.25 --class-filter 0

Arguments:
    --resolution    float, output pixel size in metres (default: 0.5)
    --class-filter  int, only use points of this class (default: 5 = high veg)
                    use 0 to include all points
"""

import os
import sys
import glob
import argparse
import numpy as np
import laspy
from osgeo import gdal, osr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR   = os.path.join(BASE_DIR, "data", "processed")
INPUT_NDSM   = os.path.join(BASE_DIR, "data", "processed", "nDSM.tif")
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute NDVI raster from LAZ point cloud."
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Output pixel size in metres (default: 0.5)"
    )
    parser.add_argument(
        "--class-filter",
        type=int,
        default=5,
        help="Only use points of this LiDAR class (default: 5 = high veg). "
             "Use 0 for all points."
    )
    return parser.parse_args()


def find_laz_files(directory, logger):
    files = sorted([
        f for f in glob.glob(os.path.join(directory, "*.laz"))
        if ".copc." not in f
    ])
    if not files:
        msg = f"No LAZ files found in {directory}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    logger.info(f"  Found {len(files)} LAZ files:")
    for f in files:
        logger.info(f"    {os.path.basename(f)}")
    return files


def get_reference_grid(ndsm_path, resolution, logger):
    """
    Get output grid parameters from reference nDSM.
    Resamples to requested resolution if different.
    """
    logger.info(f"Setting up output grid ({resolution} m resolution)...")
    ds   = gdal.Open(ndsm_path)
    if ds is None:
        msg = f"Cannot open reference raster: {ndsm_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ds   = None

    # Compute extent from original grid
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + cols * gt[1]
    y_min = y_max + rows * gt[5]

    # New grid at requested resolution
    out_cols = int(np.ceil((x_max - x_min) / resolution))
    out_rows = int(np.ceil((y_max - y_min) / resolution))
    out_gt   = (x_min, resolution, 0.0, y_max, 0.0, -resolution)

    logger.info(f"  Extent     : ({x_min:.1f}, {y_min:.1f}) - "
                f"({x_max:.1f}, {y_max:.1f})")
    logger.info(f"  Dimensions : {out_cols} x {out_rows} pixels")
    logger.info(f"  Pixel size : {resolution} m")
    return out_gt, proj, out_cols, out_rows


def load_points(laz_paths, class_filter, logger):
    """Load points from all LAZ tiles with optional class filter."""
    logger.info(f"Loading points "
                f"({'class ' + str(class_filter) if class_filter > 0 else 'all classes'})...")

    all_xs, all_ys = [], []
    all_red, all_nir = [], []
    total_all = 0

    for laz_path in laz_paths:
        logger.info(f"  Reading: {os.path.basename(laz_path)}")
        las = laspy.read(laz_path)

        classification = np.array(las.classification)
        total_all += len(classification)

        if class_filter > 0:
            mask = classification == class_filter
        else:
            mask = np.ones(len(classification), dtype=bool)

        xs  = np.array(las.x)[mask]
        ys  = np.array(las.y)[mask]
        red = np.array(las.red).astype(float)[mask]
        nir = np.array(las.nir).astype(float)[mask]

        all_xs.append(xs)
        all_ys.append(ys)
        all_red.append(red)
        all_nir.append(nir)

        logger.info(f"    Points loaded: {np.sum(mask):,} of {len(mask):,}")
        las = None

    xs  = np.concatenate(all_xs)
    ys  = np.concatenate(all_ys)
    red = np.concatenate(all_red)
    nir = np.concatenate(all_nir)

    # Normalize from 16bit to 0-1
    if red.max() > 255:
        red = red / 65535.0
        nir = nir / 65535.0

    logger.info(f"  Total points loaded : {len(xs):,}")
    logger.info(f"  Red range           : {red.min():.3f} - {red.max():.3f}")
    logger.info(f"  NIR range           : {nir.min():.3f} - {nir.max():.3f}")
    return xs, ys, red, nir


def rasterize_ndvi(xs, ys, red, nir, out_gt, out_cols, out_rows, logger):
    """
    Rasterize point cloud NIR and Red values onto output grid,
    then compute NDVI per pixel.

    For pixels with multiple points, uses the mean value.
    NoData (-9999) where no points fall in a pixel.
    """
    logger.info("Rasterizing NIR and Red values...")

    x_origin     = out_gt[0]
    y_origin     = out_gt[3]
    pixel_width  = out_gt[1]
    pixel_height = out_gt[5]  # negative

    # Convert point coordinates to pixel indices
    col_idx = ((xs - x_origin) / pixel_width).astype(int)
    row_idx = ((ys - y_origin) / pixel_height).astype(int)

    # Filter to points within grid extent
    valid = (
        (col_idx >= 0) & (col_idx < out_cols) &
        (row_idx >= 0) & (row_idx < out_rows)
    )
    col_idx = col_idx[valid]
    row_idx = row_idx[valid]
    red_v   = red[valid]
    nir_v   = nir[valid]

    logger.info(f"  Points within grid extent: {np.sum(valid):,}")

    # Accumulate NIR and Red sums per pixel using numpy
    # Use flat index for efficiency
    flat_idx = row_idx * out_cols + col_idx

    nir_sum   = np.zeros(out_rows * out_cols, dtype=np.float64)
    red_sum   = np.zeros(out_rows * out_cols, dtype=np.float64)
    count     = np.zeros(out_rows * out_cols, dtype=np.int32)

    np.add.at(nir_sum, flat_idx, nir_v)
    np.add.at(red_sum, flat_idx, red_v)
    np.add.at(count,   flat_idx, 1)

    # Reshape to 2D
    nir_sum = nir_sum.reshape(out_rows, out_cols)
    red_sum = red_sum.reshape(out_rows, out_cols)
    count   = count.reshape(out_rows, out_cols)

    # Mean per pixel
    has_data = count > 0
    nir_mean = np.where(has_data, nir_sum / count, 0.0)
    red_mean = np.where(has_data, red_sum / count, 0.0)

    logger.info(f"  Pixels with data  : {np.sum(has_data):,} of "
                f"{out_rows * out_cols:,} "
                f"({100*np.sum(has_data)/(out_rows*out_cols):.1f}%)")

    # Compute NDVI
    logger.info("Computing NDVI...")
    denom = nir_mean + red_mean
    ndvi  = np.where(
        has_data & (denom > 0),
        (nir_mean - red_mean) / denom,
        -9999.0
    )

    valid_ndvi = ndvi[ndvi != -9999.0]
    logger.info(f"  NDVI range : {valid_ndvi.min():.3f} - "
                f"{valid_ndvi.max():.3f}")
    logger.info(f"  NDVI mean  : {valid_ndvi.mean():.3f}")

    return ndvi.astype(np.float32)


def save_raster(arr, out_gt, proj, output_path, nodata, logger):
    """Save array as compressed GeoTIFF."""
    rows, cols = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        output_path, cols, rows, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    ds_out.SetGeoTransform(out_gt)
    ds_out.SetProjection(proj)
    band = ds_out.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(nodata)
    ds_out.FlushCache()
    ds_out = None
    logger.info(f"  Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("calc_ndvi_raster")

    try:
        logger.info(f"Parameters:")
        logger.info(f"  resolution   = {args.resolution} m")
        logger.info(f"  class-filter = {args.class_filter} "
                    f"({'high veg only' if args.class_filter == 5 else 'all points' if args.class_filter == 0 else 'custom'})")

        # Build output filename from parameters
        class_str   = (f"class{args.class_filter}"
                       if args.class_filter > 0 else "allpoints")
        output_path = os.path.join(
            OUTPUT_DIR,
            f"ndvi_{class_str}_{args.resolution}m.tif"
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1. Find LAZ files
        laz_files = find_laz_files(RAW_DATA_DIR, logger)

        # 2. Get output grid from reference nDSM
        out_gt, proj, out_cols, out_rows = get_reference_grid(
            INPUT_NDSM, args.resolution, logger
        )

        # 3. Load points
        xs, ys, red, nir = load_points(laz_files, args.class_filter, logger)

        # 4. Rasterize and compute NDVI
        ndvi = rasterize_ndvi(
            xs, ys, red, nir, out_gt, out_cols, out_rows, logger
        )

        # 5. Save
        logger.info(f"Saving NDVI raster...")
        save_raster(ndvi, out_gt, proj, output_path, -9999.0, logger)

        logger.info(f"Done!")
        logger.info(f"  Output : {output_path}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise