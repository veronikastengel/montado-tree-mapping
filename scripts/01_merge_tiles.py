"""
Merges multiple LiDAR-derived raster tiles (MDS or MDT) into a single
mosaic GeoTIFF, and computes the nDSM (MDS - MDT) from the merged tiles.

Output:
    - data/processed/MDS_merged.tif
    - data/processed/MDT_merged.tif
    - data/processed/nDSM.tif
"""

import os
import glob
import numpy as np
from osgeo import gdal

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from utils import get_logger
logger = get_logger("01_merge_tiles")

# ============================================================
# CONFIG
# ============================================================
RAW_DATA_DIR   = r"data\raw"
OUTPUT_DIR     = r"data\processed"
MDS_PATTERN    = "MDS-50cm-*.tif"
MDT_PATTERN    = "MDT-50cm-*.tif"
OUTPUT_MDS     = "MDS_merged.tif"
OUTPUT_MDT     = "MDT_merged.tif"
OUTPUT_NDSM    = "nDSM.tif"
NDSM_MIN_HEIGHT = 0.0   # clip nDSM below this value to 0 (removes negatives)
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_tiles(directory, pattern):
    """Find all tiles matching pattern in directory."""
    search = os.path.join(directory, pattern)
    tiles = sorted(glob.glob(search))
    if not tiles:       
        msg = f"No files found matching '{pattern}' in '{directory}'"
        logger.error(msg)
        raise FileNotFoundError(msg)
    logger.info(f"  Found {len(tiles)} tiles for pattern '{pattern}':")
    for t in tiles:
        logger.info(f"    {os.path.basename(t)}")
    return tiles

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

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logger = get_logger("01_merge_tiles")
    try:
        # Find tiles
        logger.info("\nLocating MDS tiles...")
        mds_tiles = find_tiles(RAW_DATA_DIR, MDS_PATTERN)
        logger.info("\nLocating MDT tiles...")
        mdt_tiles = find_tiles(RAW_DATA_DIR, MDT_PATTERN)

        # Merge
        mds_out = os.path.join(OUTPUT_DIR, OUTPUT_MDS)
        mdt_out = os.path.join(OUTPUT_DIR, OUTPUT_MDT)
        merge_tiles(mds_tiles, mds_out)
        merge_tiles(mdt_tiles, mdt_out)

        # Compute nDSM
        ndsm_out = os.path.join(OUTPUT_DIR, OUTPUT_NDSM)
        compute_ndsm(mds_out, mdt_out, ndsm_out, min_height=NDSM_MIN_HEIGHT)

        logger.info(f"Done! Outputs written to: {OUTPUT_DIR}")
        logger.info(f"  MDS merged : {mds_out}")
        logger.info(f"  MDT merged : {mdt_out}")
        logger.info(f"  nDSM       : {ndsm_out}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise
