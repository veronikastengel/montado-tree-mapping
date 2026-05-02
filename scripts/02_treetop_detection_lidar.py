"""
02b_treetop_detection_lidar.py
==============================
Detects individual tree top points directly from LAZ point cloud data,
without rasterization. Uses the highest class 5 (high vegetation) point
within a 3D local neighbourhood cylinder as the tree top candidate.

This is the point-cloud equivalent of the CHM local maxima method in
02_treetop_detection.py, avoiding rasterization artefacts by working
directly with the raw 3D point positions.

Scientific basis:
    Analogous to Popescu & Wynne (2004) variable window local maxima,
    applied directly to 3D point cloud data rather than a rasterized CHM.
    The highest point within a local cylinder is a well-established
    treetop proxy in airborne LiDAR literature.

Pipeline:
    1. Load all LAZ tiles, filter to class 5 (high vegetation)
    2. Normalize heights above ground using merged DTM
    3. Remove points below minimum height threshold
    4. For each point, check if it is the highest point within
       a cylinder of radius R metres (local maximum in 3D)
    5. Apply minimum distance filter to remove duplicate candidates
       from multi-peaked crowns
    6. Save treetop points to GeoPackage

Usage:
    python scripts/02_treetop_detection_lidar.py
    python scripts/02_treetop_detection_lidar.py --radius 1.25 --minheight 2.0
    python scripts/02_treetop_detection_lidar.py --radius 2.5 --minheight 3.0 --mindist 2.0
    python scripts/02_treetop_detection_lidar.py --radius_a 1.5 --radius_b 0.1 --minheight 2.0 --mindist 2.5

Arguments:
    --radius_a      float, Base radius (a) for variable window (default: 0.5)
    --radius_b      float, Scaling factor (b) for variable window (default: 0.05)
    --minheight     float, minimum height above ground in metres to
                    be considered a tree top candidate (default: 2.0)
    --mindist       float, minimum distance between 2 tree tops to count 
                    them both, otherwise only the higher one stays (default: 0.0)
"""

import os
import sys
import glob
import argparse
import numpy as np
import laspy
from osgeo import gdal, ogr, osr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
INPUT_DTM      = os.path.join(BASE_DIR, "data", "processed", "MDT_merged.tif")
OUTPUT_DIR     = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_GPKG    = os.path.join(OUTPUT_DIR, "treetops_lidar.gpkg")
LAYER_NAME_TPL = "treetops_lidar_ra{radius_a}_rb{radius_b}_mh{minheight}_md_{mindist}"
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect tree tops directly from LAZ point cloud."
    )
    parser.add_argument(
        "--radius_a",
        type=float,
        default=0.5,
        help="Base radius (a) for variable window (default: 0.5)"
    )
    parser.add_argument(
        "--radius_b",
        type=float,
        default=0.05,
        help="Scaling factor (b) for variable window (default: 0.05)"
    )
    parser.add_argument(
        "--minheight",
        type=float,
        default=2.0,
        help="Minimum height above ground in metres (default: 2.0)"
    )
    parser.add_argument(
        "--mindist",
        type=float,
        default=0.0,
        help="Minimum distance to keep separate tree tops (default: 0.0)"
    )
    return parser.parse_args()


def find_laz_files(directory, logger):
    """Find all LAZ files excluding COPC variants."""
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

def filter_outliers(xs, ys, zs, logger, upper_percentile=99.9):
    """
    Remove extreme high outliers from LiDAR data.
    """
    logger.info(f"Filtering outliers (>{upper_percentile} percentile)...")

    z_max = np.percentile(zs, upper_percentile)
    mask = zs <= z_max

    xs_f, ys_f, zs_f = xs[mask], ys[mask], zs[mask]

    logger.info(f"  Removed {len(zs) - len(zs_f):,} outliers")
    logger.info(f"  New max height: {zs_f.max():.2f} m")

    return xs_f, ys_f, zs_f

def load_class5_points(laz_paths, logger):
    """
    Load all LAZ tiles and return class 5 (high vegetation) point
    coordinates and attributes concatenated across all tiles.
    """
    logger.info("Loading class 5 points from LAZ tiles...")

    all_xs, all_ys, all_zs = [], [], []

    for laz_path in laz_paths:
        logger.info(f"  Reading: {os.path.basename(laz_path)}")
        las            = laspy.read(laz_path)
        classification = np.array(las.classification)
        mask           = classification == 5

        all_xs.append(np.array(las.x)[mask])
        all_ys.append(np.array(las.y)[mask])
        all_zs.append(np.array(las.z)[mask])

        n_total = len(las.x)
        n_class5 = np.sum(mask)
        logger.info(f"    Total points : {n_total:,}")
        logger.info(f"    Class 5      : {n_class5:,} "
                    f"({100*n_class5/n_total:.1f}%)")
        las = None

    xs = np.concatenate(all_xs)
    ys = np.concatenate(all_ys)
    zs = np.concatenate(all_zs)

    logger.info(f"  Total class 5 points: {len(xs):,}")
    return xs, ys, zs


def normalize_heights(xs, ys, zs, dtm_path, logger):
    """Subtract DTM elevation at each point to get height above ground."""
    logger.info("Normalizing heights above ground using DTM...")
    ds_dtm   = gdal.Open(dtm_path)
    if ds_dtm is None:
        msg = f"Cannot open DTM: {dtm_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    dtm_band = ds_dtm.GetRasterBand(1)
    dtm_arr  = dtm_band.ReadAsArray().astype(float)
    dtm_nd   = dtm_band.GetNoDataValue()
    if dtm_nd is not None:
        dtm_arr[dtm_arr == dtm_nd] = np.nan
    dtm_gt   = ds_dtm.GetGeoTransform()
    ds_dtm   = None

    col = ((xs - dtm_gt[0]) / dtm_gt[1]).astype(int)
    row = ((ys - dtm_gt[3]) / dtm_gt[5]).astype(int)
    nrows, ncols = dtm_arr.shape

    valid    = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    dtm_vals = np.full(len(xs), np.nan)
    dtm_vals[valid] = dtm_arr[row[valid], col[valid]]

    zs_norm = zs - dtm_vals
    good    = ~np.isnan(zs_norm)

    xs      = xs[good]
    ys      = ys[good]
    zs_norm = zs_norm[good]

    logger.info(f"  Points after normalization : {len(xs):,}")
    logger.info(f"  Height range               : "
                f"{zs_norm.min():.1f} - {zs_norm.max():.1f} m")
    return xs, ys, zs_norm


def filter_by_height(xs, ys, zs, minheight, logger):
    """Remove points below minimum height threshold."""
    logger.info(f"Filtering to points above {minheight} m...")
    mask = zs >= minheight
    xs, ys, zs = xs[mask], ys[mask], zs[mask]
    logger.info(f"  Points remaining: {len(xs):,}")
    return xs, ys, zs


def build_grid_index(xs, ys, cell_size, logger):
    """
    Build a simple grid spatial index for fast neighbourhood queries.
    Returns index dict and origin coordinates.
    """
    logger.info(f"Building spatial index (cell={cell_size} m)...")
    x_min, y_min = xs.min(), ys.min()
    col_idx = ((xs - x_min) / cell_size).astype(int)
    row_idx = ((ys - y_min) / cell_size).astype(int)
    cell_keys = col_idx * 1000000 + row_idx

    index = {}
    for i, key in enumerate(cell_keys):
        if key not in index:
            index[key] = []
        index[key].append(i)
    index = {k: np.array(v) for k, v in index.items()}

    logger.info(f"  Index cells: {len(index):,}")
    return index, x_min, y_min, cell_size


def query_grid_index(index, x_min, y_min, cell_size, cx, cy, radius):
    """Return point indices within radius of (cx, cy)."""
    col_min = int((cx - radius - x_min) / cell_size)
    col_max = int((cx + radius - x_min) / cell_size)
    row_min = int((cy - radius - y_min) / cell_size)
    row_max = int((cy + radius - y_min) / cell_size)

    candidates = []
    for c in range(col_min, col_max + 1):
        for r in range(row_min, row_max + 1):
            key = c * 1000000 + r
            if key in index:
                candidates.extend(index[key])

    return np.array(candidates, dtype=int) if candidates else np.array(
        [], dtype=int)


def detect_local_maxima_3d(xs, ys, zs, radius_a, radius_b, logger):
    """
    For each point, check if it is the highest point within a
    horizontal cylinder of given radius. Vectorized batch processing
    for speed.
    """
    logger.info(f"Detecting 3D local maxima (radius_a={radius_a}, radius_b={radius_b})...")
    logger.info(f"  Processing {len(xs):,} points...")

    max_radius = radius_a + radius_b * zs.max() if radius_a + radius_b * zs.max() < 10 else 10
    index, x_min, y_min, cell_size = build_grid_index(
        xs, ys, max_radius, logger
    )

    is_local_max = np.ones(len(xs), dtype=bool)
    report_every = max(1, len(xs) // 10)

    for i in range(len(xs)):
        if i % report_every == 0:
            logger.info(f"  Progress: {100*i/len(xs):.0f}%")

        radius = radius_a + radius_b * zs[i]

        candidates = query_grid_index(
            index, x_min, y_min, cell_size,
            xs[i], ys[i], radius
        )
        if len(candidates) == 0:
            continue

        # Vectorized distance calculation
        dx   = xs[candidates] - xs[i]
        dy   = ys[candidates] - ys[i]
        mask = (dx**2 + dy**2) <= radius**2
        in_radius = candidates[mask]

        # If any neighbour is strictly higher, this is not a local max
        if np.any(zs[in_radius] > zs[i]):
            is_local_max[i] = False

    n_maxima = np.sum(is_local_max)
    logger.info(f"  Local maxima found: {n_maxima:,}")
    return is_local_max


def save_treetops(xs, ys, zs, proj, output_gpkg, layer_name, logger):
    """
    Save treetop points to GeoPackage.
    Opens existing file or creates new, replaces layer if exists.
    Output format matches 02_treetop_detection.py for compatibility
    with script 03.
    """
    logger.info(f"Saving treetops to GeoPackage...")

    gpkg_driver = ogr.GetDriverByName("GPKG")

    if os.path.exists(output_gpkg):
        ds_out = gpkg_driver.Open(output_gpkg, 1)
        if ds_out is None:
            msg = f"Could not open: {output_gpkg}"
            logger.error(msg)
            raise RuntimeError(msg)
        for i in range(ds_out.GetLayerCount()):
            if ds_out.GetLayerByIndex(i).GetName() == layer_name:
                ds_out.DeleteLayer(i)
                logger.info(f"  Deleted existing layer: '{layer_name}'")
                break
        logger.info(f"  Opened existing GeoPackage")
    else:
        ds_out = gpkg_driver.CreateDataSource(output_gpkg)
        logger.info(f"  Created new GeoPackage: {output_gpkg}")

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    out_layer = ds_out.CreateLayer(
        layer_name, srs=srs, geom_type=ogr.wkbPoint
    )

    # Same fields as script 02 for compatibility with script 03
    out_layer.CreateField(ogr.FieldDefn("tree_id",  ogr.OFTInteger))
    out_layer.CreateField(ogr.FieldDefn("height_m", ogr.OFTReal))
    out_layer.CreateField(ogr.FieldDefn("crown_px", ogr.OFTInteger))

    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint(float(x), float(y))
        feat = ogr.Feature(out_layer.GetLayerDefn())
        feat.SetGeometry(pt)
        feat.SetField("tree_id",  i + 1)
        feat.SetField("height_m", round(float(z), 3))
        feat.SetField("crown_px", 0)  # not applicable for point cloud method
        out_layer.CreateFeature(feat)

    n_saved = out_layer.GetFeatureCount()
    ds_out.Destroy()
    logger.info(f"  Treetops saved: {n_saved:,}")
    logger.info(f"  Layer '{layer_name}' in: {output_gpkg}")
    return n_saved


def get_projection_from_dtm(dtm_path, logger):
    """Get projection string from DTM raster."""
    ds = gdal.Open(dtm_path)
    if ds is None:
        msg = f"Cannot open DTM: {dtm_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    proj = ds.GetProjection()
    ds   = None
    return proj

def apply_min_distance_filter(xs, ys, zs, mindist, logger):
    """
    Enforce minimum distance between treetops.
    Keeps highest point when multiple are closer than mindist.
    """
    if mindist <= 0:
        logger.info("Skipping min distance filter (mindist <= 0)")
        return xs, ys, zs

    logger.info(f"Applying minimum distance filter ({mindist} m)...")

    # Sort by height descending
    order = np.argsort(-zs)
    xs, ys, zs = xs[order], ys[order], zs[order]

    keep = np.ones(len(xs), dtype=bool)

    for i in range(len(xs)):
        if not keep[i]:
            continue

        dx = xs[i+1:] - xs[i]
        dy = ys[i+1:] - ys[i]
        dist2 = dx**2 + dy**2

        too_close = dist2 < mindist**2
        keep[i+1:][too_close] = False

    xs_f = xs[keep]
    ys_f = ys[keep]
    zs_f = zs[keep]

    logger.info(f"  Remaining after filter: {len(xs_f):,}")
    return xs_f, ys_f, zs_f


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("02b_treetop_detection_lidar")

    try:
        logger.info(f"Parameters:")
        logger.info(f"  radius_a    = {args.radius_a}")
        logger.info(f"  radius_b    = {args.radius_b}")
        logger.info(f"  minheight = {args.minheight} m")

        layer_name = LAYER_NAME_TPL.format(
            radius_a=args.radius_a,
            radius_b=args.radius_b,
            minheight=args.minheight,
            mindist=args.mindist
        )
        logger.info(f"  output layer = {layer_name}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1 Find LAZ files
        laz_files = find_laz_files(RAW_DATA_DIR, logger)

        # 2 Load class 5 points
        xs, ys, zs = load_class5_points(laz_files, logger)

        # 3 Normalize heights and remove outliers
        xs, ys, zs = normalize_heights(xs, ys, zs, INPUT_DTM, logger)
        xs, ys, zs = filter_outliers(xs, ys, zs, logger)

        # 4 Filter by minimum height
        xs, ys, zs = filter_by_height(xs, ys, zs, args.minheight, logger)

        # 5 Detect 3D local maxima
        is_local_max = detect_local_maxima_3d(xs, ys, zs, args.radius_a, args.radius_b, logger)

        # Extract treetop candidates
        tx = xs[is_local_max]
        ty = ys[is_local_max]
        tz = zs[is_local_max]
        logger.info(f"  Treetop candidates: {len(tx):,}")

        # keep only the treetops with more than minimum distance
        tx, ty, tz = apply_min_distance_filter(
            tx, ty, tz, args.mindist, logger
        )

        # 6 Get projection from DTM
        proj = get_projection_from_dtm(INPUT_DTM, logger)

        # 7 Save to GeoPackage
        n_saved = save_treetops(
            tx, ty, tz, proj, OUTPUT_GPKG, layer_name, logger
        )

        logger.info(f"Done! {n_saved:,} tree tops saved.")
        logger.info(f"  Output : {OUTPUT_GPKG}")
        logger.info(f"  Layer  : {layer_name}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise