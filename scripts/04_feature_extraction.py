"""
04_feature_extraction.py
========================
Extracts per-crown metrics from LAZ point cloud data.

Metrics computed:
    - mean_height, max_height       : height above ground (from DTM normalization)
    - crown_area_m2, crown_width_m  : crown geometry
    - height_width_ratio            : crown shape indicator
    - mean_intensity                : laser return strength
    - mean_ndvi                     : vegetation index from NIR + Red
    - point_density                 : points per m2
    - rugosity                      : crown surface roughness
    - vert_dist_top25/mid50/bot25   : vertical point distribution
    - n_points                      : class 5 points inside crown
    - n_points_all                  : all points inside crown
    - pct_high_veg                  : % class 5 points (use to filter buildings)

Usage:
    python scripts/04_feature_extraction.py --crowns-layer crowns_treetops_w9_mh2.0_t0.001_c0.001_mh2.0

Arguments:
    --crowns-layer    name of layer inside crowns.gpkg to enrich
                      (default: uses first layer found)
"""

import os
import sys
import argparse
import numpy as np
import laspy
from osgeo import gdal, ogr, osr
from shapely import wkb, vectorized

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
INPUT_DTM      = os.path.join(BASE_DIR, "data", "processed", "MDT_merged.tif")
INPUT_CROWNS   = os.path.join(BASE_DIR, "data", "processed", "crowns.gpkg")
LAZ_PATTERN    = "*.laz"
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-crown metrics from LAZ point cloud."
    )
    parser.add_argument(
        "--crowns-layer",
        type=str,
        default=None,
        help="Layer name inside crowns.gpkg to enrich. If not specified, uses first layer found."
    )
    return parser.parse_args()


def find_laz_files(directory, logger):
    """Find all LAZ files in directory."""
    import glob
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


def load_crowns_layer(gpkg_path, layer_name, logger):
    """Open crowns GeoPackage in read/write mode, return layer."""
    logger.info(f"Loading crowns: {os.path.basename(gpkg_path)}")
    ds = ogr.Open(gpkg_path, 1)  # 1 = read/write
    if ds is None:
        msg = f"Cannot open crowns GeoPackage: {gpkg_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    available = [ds.GetLayerByIndex(i).GetName()
                 for i in range(ds.GetLayerCount())]
    logger.info(f"  Available layers: {available}")

    if layer_name is None:
        layer_name = available[0]
        logger.info(f"  No layer specified, using: '{layer_name}'")
    elif layer_name not in available:
        msg = (f"Layer '{layer_name}' not found. Available: {available}")
        logger.error(msg)
        raise ValueError(msg)
    else:
        logger.info(f"  Using layer: '{layer_name}'")

    layer = ds.GetLayerByName(layer_name)
    logger.info(f"  Crown polygons: {layer.GetFeatureCount():,}")
    return ds, layer, layer_name


def load_point_cloud(laz_paths, logger):
    """
    Load all LAZ files, return arrays of coordinates and attributes.
    Keeps ALL points (for pct_high_veg) and class 5 filtered points
    (for metrics).
    """
    logger.info("Loading point cloud data...")

    all_xs, all_ys, all_zs = [], [], []
    all_cls = []
    all_intensity = []
    all_red, all_green, all_blue, all_infrared = [], [], [], []
    all_ret_number, all_n_returns = [], []

    for laz_path in laz_paths:
        logger.info(f"  Reading: {os.path.basename(laz_path)}")
        las = laspy.read(laz_path)

        all_xs.append(np.array(las.x))
        all_ys.append(np.array(las.y))
        all_zs.append(np.array(las.z))
        all_cls.append(np.array(las.classification))
        all_intensity.append(np.array(las.intensity).astype(float))
        all_red.append(np.array(las.red).astype(float))
        all_green.append(np.array(las.green).astype(float))
        all_blue.append(np.array(las.blue).astype(float))
        all_infrared.append(np.array(las.nir).astype(float))
        all_ret_number.append(np.array(las.return_number).astype(float))
        all_n_returns.append(np.array(las.number_of_returns).astype(float))
        logger.info(f"    Points: {len(las.x):,}")
        las = None

    # Concatenate all tiles
    xs_all         = np.concatenate(all_xs)
    ys_all         = np.concatenate(all_ys)
    zs_all         = np.concatenate(all_zs)
    classification = np.concatenate(all_cls)
    intensity_all  = np.concatenate(all_intensity)
    red_all        = np.concatenate(all_red)
    green_all      = np.concatenate(all_green)
    blue_all       = np.concatenate(all_blue)
    infrared_all   = np.concatenate(all_infrared)
    ret_number_all = np.concatenate(all_ret_number)
    n_returns_all  = np.concatenate(all_n_returns)

    logger.info(f"  Total points loaded: {len(xs_all):,}")

    # Normalize RGB+IR from 16bit to 0-1
    if red_all.max() > 255:
        red_all      = red_all / 65535.0
        green_all    = green_all / 65535.0
        blue_all     = blue_all / 65535.0
        infrared_all = infrared_all / 65535.0

    # NDVI per point
    ndvi_all = np.where(
        (infrared_all + red_all) > 0,
        (infrared_all - red_all) / (infrared_all + red_all),
        0.0
    )

    # Class 5 filter
    logger.info("Filtering to class 5 (high vegetation)...")
    veg_filter = classification == 5
    xs        = xs_all[veg_filter]
    ys        = ys_all[veg_filter]
    zs        = zs_all[veg_filter]
    intensity = intensity_all[veg_filter]
    ndvi      = ndvi_all[veg_filter]
    ret_number = ret_number_all[veg_filter]
    n_returns  = n_returns_all[veg_filter]

    logger.info(f"  Class 5 points: {len(xs):,} of {len(xs_all):,} "
                f"({100*len(xs)/len(xs_all):.1f}%)")

    return (xs_all, ys_all, classification,
            xs, ys, zs, intensity, ndvi, ret_number, n_returns)

def build_spatial_index(xs, ys, cell_size=50.0, logger=None):
    """
    Build a simple grid spatial index for fast point lookup.
    cell_size in metres -> larger = fewer cells but more points per cell.
    """
    if logger:
        logger.info(f"Building spatial index (cell size={cell_size}m)...")
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Assign each point to a grid cell
    col_idx = ((xs - x_min) / cell_size).astype(int)
    row_idx = ((ys - y_min) / cell_size).astype(int)
    cell_keys = col_idx * 100000 + row_idx  # unique cell ID per point

    # Build dict: cell_key -> array of point indices
    index = {}
    for i, key in enumerate(cell_keys):
        if key not in index:
            index[key] = []
        index[key].append(i)
    # Convert to arrays for faster lookup
    index = {k: np.array(v) for k, v in index.items()}

    if logger:
        logger.info(f"  Index cells: {len(index):,}")
    return index, x_min, y_min, cell_size


def query_spatial_index(index, x_min, y_min, cell_size,
                        xmin, xmax, ymin, ymax):
    """Return point indices whose cell overlaps the query bbox."""
    col_min = int((xmin - x_min) / cell_size)
    col_max = int((xmax - x_min) / cell_size)
    row_min = int((ymin - y_min) / cell_size)
    row_max = int((ymax - y_min) / cell_size)

    candidates = []
    for c in range(col_min, col_max + 1):
        for r in range(row_min, row_max + 1):
            key = c * 100000 + r
            if key in index:
                candidates.append(index[key])

    if candidates:
        return np.concatenate(candidates)
    return np.array([], dtype=int)

def normalize_heights(xs, ys, zs, dtm_path, logger):
    """Subtract DTM elevation at each point to get height above ground."""
    logger.info("Normalizing heights above ground...")
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
    dtm_gt = ds_dtm.GetGeoTransform()
    ds_dtm = None

    col = ((xs - dtm_gt[0]) / dtm_gt[1]).astype(int)
    row = ((ys - dtm_gt[3]) / dtm_gt[5]).astype(int)
    nrows, ncols = dtm_arr.shape
    valid = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    dtm_vals = np.full(len(xs), np.nan)
    dtm_vals[valid] = dtm_arr[row[valid], col[valid]]

    zs_norm = zs - dtm_vals
    good    = ~np.isnan(zs_norm)
    logger.info(f"  Points after normalization: {np.sum(good):,}")
    logger.info(f"  Height range: {zs_norm[good].min():.1f} - "
                f"{zs_norm[good].max():.1f} m")
    return zs_norm, good


def add_fields(layer, logger):
    """Add metric fields to layer if not already present."""
    logger.info("Adding metric fields to layer...")

    def add(name, field_type):
        if layer.FindFieldIndex(name, True) == -1:
            layer.CreateField(ogr.FieldDefn(name, field_type))

    add("mean_height",        ogr.OFTReal)
    add("max_height",         ogr.OFTReal)
    add("crown_area_m2",      ogr.OFTReal)
    add("crown_width_m",      ogr.OFTReal)
    add("height_width_ratio", ogr.OFTReal)
    add("mean_intensity",     ogr.OFTReal)
    add("mean_ndvi",          ogr.OFTReal)
    add("point_density",      ogr.OFTReal)
    add("rugosity",           ogr.OFTReal)
    add("vert_dist_top25",    ogr.OFTReal)
    add("vert_dist_mid50",    ogr.OFTReal)
    add("vert_dist_bot25",    ogr.OFTReal)
    add("n_points",           ogr.OFTInteger)
    add("n_points_all",       ogr.OFTInteger)
    add("pct_high_veg",       ogr.OFTReal)
    logger.info("  Fields ready.")


def compute_rugosity(z_vals):
    """Crown surface roughness -> higher = more irregular (cork oak tendency)."""
    if len(z_vals) < 3:
        return 0.0
    return float(np.std(z_vals) / (np.ptp(z_vals) + 0.001))


def compute_vertical_distribution(z_vals):
    """Fraction of points in top 25%, middle 50%, bottom 25% of crown."""
    if len(z_vals) < 3:
        return 0.0, 1.0, 0.0
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min
    if z_range < 0.1:
        return 0.0, 1.0, 0.0
    top25 = float(np.sum(z_vals >= z_min + 0.75 * z_range) / len(z_vals))
    bot25 = float(np.sum(z_vals <= z_min + 0.25 * z_range) / len(z_vals))
    return top25, 1.0 - top25 - bot25, bot25


def process_crowns(layer, xs_all, ys_all,
                   xs, ys, zs_norm, good,
                   intensity, ndvi, logger):
    """
    Iterate over crown polygons, compute metrics from points inside each,
    write results back to GeoPackage layer.
    """
    n_features = layer.GetFeatureCount()
    logger.info(f"Processing {n_features:,} crown polygons...")

    # Apply good mask to class 5 arrays upfront
    xs_v   = xs[good]
    ys_v   = ys[good]
    zs_v   = zs_norm[good]
    int_v  = intensity[good]
    ndvi_v = ndvi[good]

    # Build spatial indices
    logger.info("Building spatial indices...")
    idx_veg, xmin_v, ymin_v, cs_v = build_spatial_index(
        xs_v, ys_v, cell_size=50.0, logger=logger
    )
    idx_all, xmin_a, ymin_a, cs_a = build_spatial_index(
        xs_all, ys_all, cell_size=50.0, logger=logger
    )

    layer.ResetReading()
    processed = 0
    skipped   = 0

    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            skipped += 1
            continue

        xmin, xmax, ymin, ymax = geom.GetEnvelope()

        # Use Index for pre-filter
        cands     = query_spatial_index(idx_veg, xmin_v, ymin_v, cs_v,
                                xmin, xmax, ymin, ymax)
        cands_all = query_spatial_index(idx_all, xmin_a, ymin_a, cs_a,  
                                xmin, xmax, ymin, ymax)

        if len(cands) == 0 and len(cands_all) == 0:
            skipped += 1
            continue

        poly = wkb.loads(bytes(geom.ExportToWkb()))

        # Class 5 points inside polygon
        if len(cands) > 0:
            inside = vectorized.contains(poly, xs_v[cands], ys_v[cands])
            idx    = cands[inside]
        else:
            idx = np.array([], dtype=int)

        # All points inside polygon
        if len(cands_all) > 0:
            inside_all = vectorized.contains(
                poly, xs_all[cands_all], ys_all[cands_all]
            )
            idx_all = cands_all[inside_all]
        else:
            idx_all = np.array([], dtype=int)

        n_all        = len(idx_all)
        n_veg        = len(idx)
        pct_high_veg = float(n_veg / n_all * 100.0) if n_all > 0 else 0.0

        # Always write classification stats even if skipping metrics
        feature.SetField("pct_high_veg", pct_high_veg)
        feature.SetField("n_points_all", n_all)
        feature.SetField("n_points",     n_veg)

        if n_veg < 3:
            layer.SetFeature(feature)
            skipped += 1
            continue

        # Extract point values
        z_in    = zs_v[idx]
        int_in  = int_v[idx]
        ndvi_in = ndvi_v[idx]

        # Compute metrics
        area_m2  = float(poly.area)
        width_m  = float(max(
            poly.bounds[2] - poly.bounds[0],
            poly.bounds[3] - poly.bounds[1]
        ))
        max_h    = float(np.max(z_in))
        top25, mid50, bot25 = compute_vertical_distribution(z_in)

        feature.SetField("mean_height",        float(np.mean(z_in)))
        feature.SetField("max_height",         max_h)
        feature.SetField("crown_area_m2",      area_m2)
        feature.SetField("crown_width_m",      width_m)
        feature.SetField("height_width_ratio",
                         float(max_h / width_m) if width_m > 0 else 0.0)
        feature.SetField("mean_intensity",     float(np.mean(int_in)))
        feature.SetField("mean_ndvi",          float(np.mean(ndvi_in)))
        feature.SetField("point_density",
                         float(n_veg / area_m2) if area_m2 > 0 else 0.0)
        feature.SetField("rugosity",           compute_rugosity(z_in))
        feature.SetField("vert_dist_top25",    top25)
        feature.SetField("vert_dist_mid50",    mid50)
        feature.SetField("vert_dist_bot25",    bot25)
        layer.SetFeature(feature)

        processed += 1
        if processed % 200 == 0:
            logger.info(f"  Processed {processed:,} / {n_features:,} crowns...")

    logger.info(f"  Processed : {processed:,} crowns")
    logger.info(f"  Skipped   : {skipped:,} crowns (no points or < 3)")
    return processed, skipped


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("04_feature_extraction")

    try:
        # 1 Open crowns layer
        ds_crowns, layer, layer_name = load_crowns_layer(
            INPUT_CROWNS, args.crowns_layer, logger
        )
        logger.info(f"  crowns layer = {layer_name}")

        # 2 Find and load LAZ files
        laz_files = find_laz_files(RAW_DATA_DIR, logger)
        (xs_all, ys_all, classification,
         xs, ys, zs, intensity, ndvi,
         ret_number, n_returns) = load_point_cloud(laz_files, logger)

        # 3 Normalize heights using DTM
        zs_norm, good = normalize_heights(xs, ys, zs, INPUT_DTM, logger)

        # 4 Add fields to layer
        add_fields(layer, logger)

        # 5 Process each crown
        processed, skipped = process_crowns(
            layer,
            xs_all, ys_all,
            xs, ys, zs_norm, good,
            intensity, ndvi,
            logger
        )

        ds_crowns.Destroy()

        logger.info(f"Done!")
        logger.info(f"  Crowns processed : {processed:,}")
        logger.info(f"  Crowns skipped   : {skipped:,}")
        logger.info(f"  Results in       : {INPUT_CROWNS}")
        logger.info(f"  Layer            : {layer_name}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise