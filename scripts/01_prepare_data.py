"""
Merges multiple LiDAR-derived raster tiles (MDS or MDT) into a single
mosaic GeoTIFF, and computes the nDSM (MDS - MDT) from the merged tiles.
Generates vegetation mask from LAZ classification (class 5 = high vegetation only)

Usage:
    python scripts/01_prepare_data.py --minheight 2.0

Arguments:
    --minheight    float, minimum nDSM height to keep (default: 2.0)

Output:
    - data/processed/MDS_merged.tif
    - data/processed/MDT_merged.tif
    - data/processed/nDSM.tif
    - data/processed/vegetation_mask.tif
"""

import os
import sys
import glob
import argparse
import numpy as np
from osgeo import gdal, ogr, osr
import laspy
from scipy.ndimage import binary_dilation

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
RAW_DATA_DIR   = r"data\raw"
OUTPUT_DIR     = r"data\processed"
MDS_PATTERN    = "MDS-50cm-*.tif"
MDT_PATTERN    = "MDT-50cm-*.tif"
LAZ_PATTERN  = "*.laz"
OUTPUT_MDS     = "MDS_merged.tif"
OUTPUT_MDT     = "MDT_merged.tif"
OUTPUT_NDSM    = "nDSM.tif"
OUTPUT_MASK  = os.path.join(OUTPUT_DIR, "vegetation_mask.tif")
NDSM_MIN_HEIGHT = 0.0   # clip nDSM below this value to 0 (removes negatives)
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare raster data and vegetation mask for tree mapping."
    )
    parser.add_argument(
        "--minheight",
        type=float,
        default=1.0,
        help="Minimum nDSM height in metres to include in mask (default: 1.0)"
    )
    return parser.parse_args()

def find_files(directory, pattern):
    """Find all files matching pattern in directory."""
    search = os.path.join(directory, pattern)
    files = sorted(glob.glob(search))
    if not files:       
        msg = f"No files found matching '{pattern}' in '{directory}'"
        logger.error(msg)
        raise FileNotFoundError(msg)
    logger.info(f"  Found {len(files)} files for pattern '{pattern}':")
    for f in files:
        logger.info(f"    {os.path.basename(f)}")
    return files

def merge_tiles(tile_paths, output_path):
    """
    Merge a list of raster tiles into a single GeoTIFF using GDAL VRT.
    Uses bilinear resampling where tiles overlap.
    """
    logger.info(f"\nMerging {len(tile_paths)} tiles -> {output_path}")

    # Build a virtual mosaic first (fast, no data copy)
    vrt_path = output_path.replace(".tif", ".vrt")
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg="bilinear",
        addAlpha=False
    )
    vrt = gdal.BuildVRT(vrt_path, tile_paths, options=vrt_options)
    if vrt is None:
        msg = f"Failed to build VRT for {output_path}"
        logger.error(msg)
        raise RuntimeError(msg)
    vrt.FlushCache()
    vrt = None

    # Translate VRT to actual GeoTIFF
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=[
            "COMPRESS=DEFLATE",
            "TILED=YES",
            "BIGTIFF=IF_SAFER"
        ]
    )
    ds = gdal.Translate(output_path, vrt_path, options=translate_options)
    if ds is None:
        msg = f"Failed to translate VRT to GeoTIFF: {output_path}"
        logger.error(msg)
        raise RuntimeError(msg)
    ds.FlushCache()
    ds = None

    # Clean up VRT
    if os.path.exists(vrt_path):
        os.remove(vrt_path)

    # Report output info
    ds_out = gdal.Open(output_path)
    gt = ds_out.GetGeoTransform()
    cols = ds_out.RasterXSize
    rows = ds_out.RasterYSize
    logger.info(f"  Output dimensions: {cols} x {rows} pixels")
    logger.info(f"  Pixel size: {gt[1]:.2f} m")
    logger.info(f"  Extent: ({gt[0]:.1f}, {gt[3] + rows*gt[5]:.1f}) - "
          f"({gt[0] + cols*gt[1]:.1f}, {gt[3]:.1f})")
    ds_out = None
    logger.info(f"  Saved: {output_path}")

def compute_ndsm(mds_path, mdt_path, output_path, min_height=0.0):
    """
    Compute nDSM = MDS - MDT, clamp negatives to min_height.
    Saves as compressed GeoTIFF.
    """
    logger.info(f"\nComputing nDSM -> {output_path}")

    ds_mds = gdal.Open(mds_path)
    ds_mdt = gdal.Open(mdt_path)

    if ds_mds is None:
        msg = f"Cannot open MDS: {mds_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if ds_mdt is None:
        msg = f"Cannot open MDT: {mdt_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Check dimensions match
    if (ds_mds.RasterXSize != ds_mdt.RasterXSize or
            ds_mds.RasterYSize != ds_mdt.RasterYSize):
        msg = f"MDS and MDT dimensions don't match: \nMDS={ds_mds.RasterXSize}x{ds_mds.RasterYSize}, \nMDT={ds_mdt.RasterXSize}x{ds_mdt.RasterYSize}"
        logger.error(msg)
        raise ValueError(msg)

    band_mds = ds_mds.GetRasterBand(1)
    band_mdt = ds_mdt.GetRasterBand(1)

    mds_arr = band_mds.ReadAsArray().astype(float)
    mdt_arr = band_mdt.ReadAsArray().astype(float)

    # Handle NoData
    mds_nd = band_mds.GetNoDataValue()
    mdt_nd = band_mdt.GetNoDataValue()
    nodata_mask = np.zeros(mds_arr.shape, dtype=bool)
    if mds_nd is not None:
        nodata_mask |= (mds_arr == mds_nd)
    if mdt_nd is not None:
        nodata_mask |= (mdt_arr == mdt_nd)

    # Compute nDSM
    ndsm = mds_arr - mdt_arr

    # Clamp negatives
    ndsm = np.where(ndsm < min_height, min_height, ndsm)

    # Restore NoData areas
    ndsm[nodata_mask] = -9999.0

    # Write output
    gt  = ds_mds.GetGeoTransform()
    proj = ds_mds.GetProjection()
    rows, cols = ndsm.shape

    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        output_path, cols, rows, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj)
    band_out = ds_out.GetRasterBand(1)
    band_out.WriteArray(ndsm)
    band_out.SetNoDataValue(-9999.0)
    ds_out.FlushCache()
    ds_out = None
    ds_mds = None
    ds_mdt = None

    # Report stats
    ds_check = gdal.Open(output_path)
    band_check = ds_check.GetRasterBand(1)
    arr = band_check.ReadAsArray()
    valid = arr[arr != -9999.0]
    logger.info(f"  nDSM height range: {valid.min():.2f} - {valid.max():.2f} m")
    logger.info(f"  Mean height above ground: {valid.mean():.2f} m")
    ds_check = None
    logger.info(f"  Saved: {output_path}")

def get_reference_raster_info(raster_path, logger):
    """Get geotransform, projection, dimensions from reference raster."""
    ds = gdal.Open(raster_path)
    if ds is None:
        msg = f"Cannot open reference raster: {raster_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    gt    = ds.GetGeoTransform()
    proj  = ds.GetProjection()
    cols  = ds.RasterXSize
    rows  = ds.RasterYSize
    ds    = None
    return gt, proj, cols, rows


def laz_points_to_mask(laz_paths, ref_gt, ref_proj, ref_cols, ref_rows,
                       minheight, output_path, logger):
    """
    Rasterize class 4+5 (medium+high vegetation) points from all LAZ files
    into a binary mask matching the reference raster grid.

    Mask values:
        1 = high vegetation above min_height
        0 = everything else (buildings, ground, low veg etc.)
    """
    logger.info("Generating vegetation mask from LAZ classification...")
    logger.info(f"  Reference grid: {ref_cols} x {ref_rows} pixels")
    logger.info(f"  Pixel size: {ref_gt[1]:.2f} m")

    # Initialise empty mask
    mask = np.zeros((ref_rows, ref_cols), dtype=np.uint8)

    x_origin    = ref_gt[0]
    y_origin    = ref_gt[3]
    pixel_width = ref_gt[1]
    pixel_height = ref_gt[5]  # negative

    total_points   = 0
    class4_5_points = 0
    masked_in      = 0

    for laz_path in laz_paths:
        logger.info(f"  Processing: {os.path.basename(laz_path)}")
        las = laspy.read(laz_path)

        xs             = np.array(las.x)
        ys             = np.array(las.y)
        zs             = np.array(las.z)
        classification = np.array(las.classification)

        total_points += len(xs)

        # Filter to class 4,5 only
        class4_5_filter = (classification == 4) | (classification == 5)
        xs45 = xs[class4_5_filter]
        ys45 = ys[class4_5_filter]
        class4_5_points += len(xs45)

        if len(xs45) == 0:
            logger.info(f"    No class 4 5 points found in this tile")
            continue

        # Convert coordinates to pixel indices
        cols_idx = ((xs45 - x_origin) / pixel_width).astype(int)
        rows_idx = ((ys45 - y_origin) / pixel_height).astype(int)

        # Keep only points within raster extent
        valid = (
            (cols_idx >= 0) & (cols_idx < ref_cols) &
            (rows_idx >= 0) & (rows_idx < ref_rows)
        )
        cols_idx = cols_idx[valid]
        rows_idx = rows_idx[valid]
        masked_in += len(cols_idx)

        # Burn class 4 5 points into mask
        mask[rows_idx, cols_idx] = 1

        logger.info(f"    Total points     : {len(xs):,}")
        logger.info(f"    Class 5 points   : {len(xs45):,} "
                    f"({100*len(xs45)/len(xs):.1f}%)")
        logger.info(f"    Points in extent : {len(cols_idx):,}")
        las = None

    logger.info(f"Summary across all tiles:")
    logger.info(f"  Total points processed : {total_points:,}")
    logger.info(f"  Class 4+5 points         : {class4_5_points:,} "
                f"({100*class4_5_points/total_points:.1f}%)")
    logger.info(f"  Mask pixels set        : {masked_in:,}")
    logger.info(f"  Mask coverage          : "
                f"{100*np.sum(mask>0)/(ref_rows*ref_cols):.1f}% of raster")

    # Dilate mask to fill gaps and buffer crown edges
    # Each pixel = 0.5m, so buffer_pixels=2 gives ~1m buffer
    buffer_pixels = 2
    logger.info(f"  Applying morphological dilation "
                f"(buffer={buffer_pixels} pixels = "
                f"{buffer_pixels * ref_gt[1]:.1f} m)...")
    mask_dilated = binary_dilation(
        mask,
        iterations=buffer_pixels
    ).astype(np.uint8)

    pixels_added = np.sum(mask_dilated) - np.sum(mask)
    logger.info(f"  Pixels added by dilation: {pixels_added:,}")

    # Save dilated mask
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        output_path, ref_cols, ref_rows, 1, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    ds_out.SetGeoTransform(ref_gt)
    ds_out.SetProjection(ref_proj)
    band_out = ds_out.GetRasterBand(1)
    band_out.WriteArray(mask_dilated)
    band_out.SetNoDataValue(255)
    ds_out.FlushCache()
    ds_out = None
    logger.info(f"  Saved: {output_path}")
    return mask_dilated

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("01_prepare_data")
    try:
        # 1 Find tiles
        logger.info("\nLocating input tiles...")
        mds_tiles = find_files(RAW_DATA_DIR, MDS_PATTERN)
        mdt_tiles = find_files(RAW_DATA_DIR, MDT_PATTERN)
        laz_files = find_files(RAW_DATA_DIR, LAZ_PATTERN)

        # 2 Merge
        logger.info("\nMerging tiles...")
        mds_out = os.path.join(OUTPUT_DIR, OUTPUT_MDS)
        mdt_out = os.path.join(OUTPUT_DIR, OUTPUT_MDT)
        merge_tiles(mds_tiles, mds_out)
        merge_tiles(mdt_tiles, mdt_out)

        # 3 Compute nDSM
        logger.info("\nComputing nDSM...")
        ndsm_out = os.path.join(OUTPUT_DIR, OUTPUT_NDSM)
        compute_ndsm(mds_out, mdt_out, ndsm_out, min_height=NDSM_MIN_HEIGHT)

        # 4 Generate vegetation mask from LAZ
        logger.info("\nGenerating vegetation mask...")
        logger.info(f"  minheight = {args.minheight} m")
        ref_gt, ref_proj, ref_cols, ref_rows = get_reference_raster_info(
            ndsm_out, logger
        )
        laz_points_to_mask(
            laz_files, ref_gt, ref_proj, ref_cols, ref_rows,
            args.minheight, OUTPUT_MASK, logger
        )

        logger.info(f"Done! Outputs written to: {OUTPUT_DIR}")
        logger.info(f"  MDS merged       : {mds_out}")
        logger.info(f"  MDT merged       : {mdt_out}")
        logger.info(f"  nDSM             : {ndsm_out}")
        logger.info(f"  Vegetation mask  : {OUTPUT_MASK}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise
