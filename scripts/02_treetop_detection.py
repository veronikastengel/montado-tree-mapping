"""
02_treetop_detection.py
=======================
Detects individual tree top points from a normalized DSM (nDSM).

Pipeline:
    1. Threshold nDSM to remove ground (default > 2m)
    2. Smooth with average neighbourhood filter
    3. Find local maxima using max neighbourhood filter
    4. Compare smoothed vs max to get binary treetop mask
    5. Remove ground pixels from mask
    6. Vectorize mask to polygons
    7. Compute centroids (one point per tree top)
    8. Sample nDSM height at each centroid
    9. Save to GeoPackage

Usage:
    python scripts/02_treetop_detection.py --window 15 --minheight 3.0 --tolerance 0.01

Arguments, all optional:
    --window        integer, neighbourhood window size in pixels (default: 9)
    --minheight    float, minimum height in metres to be considered a tree (default: 2.0)
    --tolerance     float, comparison tolerance for local maxima detection (default: 0.001)
"""

import os
import sys
import argparse
import numpy as np
from osgeo import gdal, ogr, osr
from scipy.ndimage import uniform_filter, maximum_filter

# Add scripts folder to path for utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_NDSM  = os.path.join(BASE_DIR, "data", "processed", "nDSM.tif")
INPUT_MASK  = os.path.join(BASE_DIR, "data", "processed", "vegetation_mask.tif")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_GPKG = os.path.join(OUTPUT_DIR, "treetops.gpkg")
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect tree top points from nDSM raster."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=9,
        help="Neighbourhood window size in pixels (default: 9). "
             "Use odd numbers only. Larger = fewer detections."
    )
    parser.add_argument(
        "--minheight",
        type=float,
        default=2.0,
        help="Minimum height above ground in metres (default: 2.0)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="Tolerance for float comparison of local maxima (default: 0.001)"
    )
    return parser.parse_args()


def load_ndsm(path, logger):
    """Load nDSM raster, return array + geotransform + projection."""
    logger.info(f"Loading nDSM: {path}")
    ds = gdal.Open(path)
    if ds is None:
        msg = f"Cannot open nDSM: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    band = ds.GetRasterBand(1)
    arr  = band.ReadAsArray().astype(np.float32)
    nd   = band.GetNoDataValue()
    if nd is not None:
        arr[arr == nd] = 0.0
    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    logger.info(f"  Dimensions: {ds.RasterXSize} x {ds.RasterYSize} pixels")
    logger.info(f"  Pixel size: {gt[1]:.2f} m")
    logger.info(f"  Height range: {arr.min():.2f} - {arr.max():.2f} m")
    ds = None
    return arr, gt, proj


def threshold_ndsm(arr, minheight, logger):
    """Zero out all pixels below min height."""
    logger.info(f"Thresholding nDSM at {minheight} m...")
    thresholded = np.where(arr >= minheight, arr, 0.0)
    veg_pixels  = np.sum(thresholded > 0)
    total       = arr.size
    logger.info(f"  Vegetation pixels: {veg_pixels:,} of {total:,} "
                f"({100*veg_pixels/total:.1f}%)")
    return thresholded

def apply_vegetation_mask(arr, mask_path, logger):
    """Apply vegetation mask — zero out anything not class 4 5 medium / high vegetation."""
    logger.info("Applying vegetation mask...")
    ds_mask  = gdal.Open(mask_path)
    if ds_mask is None:
        msg = f"Cannot open vegetation mask: {mask_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    mask_arr = ds_mask.GetRasterBand(1).ReadAsArray()
    ds_mask  = None
    masked   = np.where(mask_arr == 1, arr, 0.0)
    removed  = np.sum((arr > 0) & (mask_arr != 1))
    logger.info(f"  Pixels removed by mask: {removed:,}")
    return masked

def detect_local_maxima(arr, window, tolerance, logger):
    """
    Detect local maxima:
    1. Smooth with average filter
    2. Compute max filter on smoothed result
    3. Pixels where smoothed == max are local peaks
    """
    logger.info(f"Detecting local maxima (window={window})...")

    # Ensure window is odd
    if window % 2 == 0:
        window += 1
        logger.info(f"  Window adjusted to {window} (must be odd)")

    # Step 1: smooth
    logger.info(f"  Applying average filter (window={window})...")
    smoothed = uniform_filter(arr, size=window)

    # Step 2: max filter on smoothed
    logger.info(f"  Applying max filter (window={window})...")
    max_filtered = maximum_filter(smoothed, size=window)

    # Step 3: local maxima
    logger.info(f"  Finding local maxima (tolerance={tolerance})...")
    local_max = np.abs(smoothed - max_filtered) < tolerance

    # Step 4: remove ground pixels
    local_max[arr == 0] = False

    n_maxima = np.sum(local_max)
    logger.info(f"  Local maxima found: {n_maxima:,}")
    return local_max.astype(np.uint8)


def save_raster(arr, gt, proj, path, nodata=0, logger=None):
    """Save a numpy array as a GeoTIFF."""
    rows, cols = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path, cols, rows, 1, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None
    if logger:
        logger.info(f"  Saved raster: {path}")


def vectorize_to_polygons(raster_path, logger):
    """Convert binary raster to polygon layer, keep only value=1."""
    logger.info("Vectorizing local maxima to polygons...")
    ds_rast = gdal.Open(raster_path)
    band    = ds_rast.GetRasterBand(1)

    # Create in-memory vector layer
    mem_driver = ogr.GetDriverByName("MEM")
    mem_ds     = mem_driver.CreateDataSource("memdata")
    srs        = osr.SpatialReference()
    srs.ImportFromWkt(ds_rast.GetProjection())
    mem_layer  = mem_ds.CreateLayer("polys", srs=srs)

    field = ogr.FieldDefn("value", ogr.OFTInteger)
    mem_layer.CreateField(field)

    gdal.Polygonize(band, band, mem_layer, 0, [], callback=None)
    ds_rast = None

    # Filter to value == 1 only
    mem_layer.SetAttributeFilter("value = 1")
    n_polys = mem_layer.GetFeatureCount()
    logger.info(f"  Polygons after filtering (value=1): {n_polys:,}")

    return mem_ds, mem_layer, srs


def compute_centroids(mem_layer, ndsm_arr, gt, proj, output_gpkg,
                      layer_name, logger):
    """
    Compute centroid of each polygon, sample nDSM height, save to GeoPackage.
    Opens existing GeoPackage if present, deletes layer if it already exists.
    """
    logger.info("Computing centroids and sampling heights...")

    gpkg_driver = ogr.GetDriverByName("GPKG")

    # Open existing or create new GeoPackage
    if os.path.exists(output_gpkg):
        ds_out = gpkg_driver.Open(output_gpkg, 1)  # 1 = read/write
        if ds_out is None:
            msg = f"Could not open existing GeoPackage: {output_gpkg}"
            logger.error(msg)
            raise RuntimeError(msg)
        # Delete layer if it already exists
        for i in range(ds_out.GetLayerCount()):
            if ds_out.GetLayerByIndex(i).GetName() == layer_name:
                ds_out.DeleteLayer(i)
                logger.info(f"  Deleted existing layer: {layer_name}")
                break
        logger.info(f"  Opened existing GeoPackage: {output_gpkg}")
    else:
        ds_out = gpkg_driver.CreateDataSource(output_gpkg)
        logger.info(f"  Created new GeoPackage: {output_gpkg}")

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    out_layer = ds_out.CreateLayer(
        layer_name, srs=srs, geom_type=ogr.wkbPoint
    )

    # Add fields
    out_layer.CreateField(ogr.FieldDefn("tree_id",  ogr.OFTInteger))
    out_layer.CreateField(ogr.FieldDefn("height_m", ogr.OFTReal))
    out_layer.CreateField(ogr.FieldDefn("crown_px", ogr.OFTInteger))

    def sample_height(x, y):
        col = int((x - gt[0]) / gt[1])
        row = int((y - gt[3]) / gt[5])
        rows_total, cols_total = ndsm_arr.shape
        if 0 <= col < cols_total and 0 <= row < rows_total:
            return float(ndsm_arr[row, col])
        return 0.0

    mem_layer.ResetReading()
    tree_id = 1
    skipped = 0

    for feature in mem_layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            skipped += 1
            continue

        centroid = geom.Centroid()
        cx, cy   = centroid.GetX(), centroid.GetY()
        height   = sample_height(cx, cy)
        crown_px = int(feature.GetGeometryRef().GetArea() / (gt[1] ** 2))

        out_feat = ogr.Feature(out_layer.GetLayerDefn())
        out_feat.SetGeometry(centroid)
        out_feat.SetField("tree_id",  tree_id)
        out_feat.SetField("height_m", round(height, 3))
        out_feat.SetField("crown_px", crown_px)
        out_layer.CreateFeature(out_feat)
        tree_id += 1

    n_trees = out_layer.GetFeatureCount()
    ds_out.Destroy()

    logger.info(f"  Tree tops saved: {n_trees:,}")
    logger.info(f"  Skipped (no geometry): {skipped}")
    logger.info(f"  Layer '{layer_name}' saved to: {output_gpkg}")
    return n_trees


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("02_treetop_detection")

    try:
        logger.info(f"  window     = {args.window}")
        logger.info(f"  minheight = {args.minheight} m")
        logger.info(f"  tolerance  = {args.tolerance}")

        param_str   = f"w{args.window}_mh{args.minheight}_t{args.tolerance}"
        LAYER_NAME  = f"treetops_{param_str}"
        binary_path = os.path.join(OUTPUT_DIR, f"treetops_binary_{param_str}.tif")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1 Load nDSM
        ndsm_arr, gt, proj = load_ndsm(INPUT_NDSM, logger)

        # 2 Threshold
        ndsm_thresh = threshold_ndsm(ndsm_arr, args.minheight, logger)

        # 3 Apply vegetation mask
        ndsm_thresh = apply_vegetation_mask(ndsm_thresh, INPUT_MASK, logger)

        # 4 Detect local maxima
        local_max = detect_local_maxima(
            ndsm_thresh, args.window, args.tolerance, logger
        )

        # 5 Save binary raster (for inspection)
        save_raster(local_max, gt, proj, binary_path, logger=logger)

        # 6 Vectorize
        mem_ds, mem_layer, srs = vectorize_to_polygons(binary_path, logger)

        # 7 Centroids + heights -> GeoPackage
        n_trees = compute_centroids(
            mem_layer, ndsm_arr, gt, proj,
            OUTPUT_GPKG, LAYER_NAME, logger
        )

        logger.info(f"Done! {n_trees:,} tree tops detected.")
        logger.info(f"  Binary raster : {binary_path}")
        logger.info(f"  Tree tops     : {OUTPUT_GPKG} / layer: {LAYER_NAME}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise