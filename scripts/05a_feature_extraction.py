"""
05a_feature_extraction.py
=========================
Extracts per-crown metrics from LAZ point cloud data, and samples
landscape classification context from Sentinel-2 based landscape
classification raster.

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
    - is_edge                       : 1 if crown touches raster edge
    - landscape_class               : majority landscape class within crown polygon
                                      (from Sentinel-2 RF classification)
                                      -1 if no valid pixels found
    - landscape_conf                : mean confidence of landscape class within crown
                                      0.0 if no valid pixels found

Landscape class IDs (from 04b_landscape_classification.py):
    1 = Olive grove
    2 = Cork/Holm oak montado
    3 = Eucalyptus
    4 = Other broadleaf
    5 = Maritime pine
    6 = Shrubland
   -1 = No valid landscape pixels within crown

Usage:
    python scripts/05a_feature_extraction.py
    python scripts/05a_feature_extraction.py --crowns-layer crowns_treetops_lidar_ra2.0_rb0.15_mh2.0_md_4.5_c0.05_mh2.0_ma4

Arguments:
    --crowns-layer      name of layer inside crowns.gpkg (default: first layer)
    --edge-buffer       distance in metres from raster edge to flag as is_edge (default: 1.0)
    --landscape-classif path to landscape classification raster (default: see CONFIG)
    --landscape-conf    path to landscape confidence raster (default: see CONFIG)
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

# Default landscape raster paths — override with --landscape-classif
# and --landscape-conf if you have multiple runs of 04b
DEFAULT_LANDSCAPE_CLASSIF = os.path.join(
    BASE_DIR, "data", "processed",
    "landscape_classification_ne500_mdNone_msl10_mt0.1.tif"
)
DEFAULT_LANDSCAPE_CONF = os.path.join(
    BASE_DIR, "data", "processed",
    "landscape_confidence_ne500_mdNone_msl10_mt0.1.tif"
)

LANDSCAPE_CONF_NODATA = -9999.0  # nodata value in confidence raster
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-crown metrics from LAZ point cloud."
    )
    parser.add_argument(
        "--crowns-layer",
        type=str,
        default=None,
        help="Layer name inside crowns.gpkg to enrich. "
             "If not specified, uses first layer found."
    )
    parser.add_argument(
        "--edge-buffer",
        type=float,
        default=1.0,
        help="Distance in metres from raster edge to flag crown "
             "as is_edge (default: 1.0)"
    )
    parser.add_argument(
        "--landscape-classif",
        type=str,
        default=DEFAULT_LANDSCAPE_CLASSIF,
        help="Path to landscape classification raster from 04b "
             "(default: see CONFIG)"
    )
    parser.add_argument(
        "--landscape-conf",
        type=str,
        default=DEFAULT_LANDSCAPE_CONF,
        help="Path to landscape confidence raster from 04b "
             "(default: see CONFIG)"
    )
    return parser.parse_args()


def find_laz_files(directory, logger):
    """Find all LAZ files in directory excluding COPC variants."""
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


def get_raster_extent(raster_path, logger):
    """Get geographic extent of raster as (x_min, x_max, y_min, y_max)."""
    ds   = gdal.Open(raster_path)
    if ds is None:
        msg = f"Cannot open raster: {raster_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    gt   = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ds   = None
    x_min = gt[0]
    x_max = gt[0] + cols * gt[1]
    y_max = gt[3]
    y_min = gt[3] + rows * gt[5]
    logger.info(f"  Raster extent: ({x_min:.1f}, {y_min:.1f}) - "
                f"({x_max:.1f}, {y_max:.1f})")
    return x_min, x_max, y_min, y_max


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


def load_landscape_rasters(classif_path, conf_path, logger):
    """
    Load landscape classification and confidence rasters into numpy arrays.
    Returns arrays, geotransforms, and dimensions for pixel sampling.
    """
    logger.info("Loading landscape rasters...")

    # Classification raster
    ds_cls = gdal.Open(classif_path)
    if ds_cls is None:
        msg = f"Cannot open landscape classification: {classif_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    cls_arr = ds_cls.GetRasterBand(1).ReadAsArray().astype(np.int16)
    cls_nd  = ds_cls.GetRasterBand(1).GetNoDataValue()
    cls_gt  = ds_cls.GetGeoTransform()
    cls_rows = ds_cls.RasterYSize
    cls_cols = ds_cls.RasterXSize
    ds_cls  = None
    logger.info(f"  Classification: {cls_cols} x {cls_rows} px, "
                f"pixel={cls_gt[1]:.1f}m, nodata={cls_nd}")

    # Confidence raster
    ds_conf = gdal.Open(conf_path)
    if ds_conf is None:
        msg = f"Cannot open landscape confidence: {conf_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    conf_arr  = ds_conf.GetRasterBand(1).ReadAsArray().astype(np.float32)
    conf_nd   = ds_conf.GetRasterBand(1).GetNoDataValue()
    conf_gt   = ds_conf.GetGeoTransform()
    conf_rows = ds_conf.RasterYSize
    conf_cols = ds_conf.RasterXSize
    ds_conf   = None
    logger.info(f"  Confidence    : {conf_cols} x {conf_rows} px, "
                f"pixel={conf_gt[1]:.1f}m, nodata={conf_nd}")

    # Warn if grids don't match — they should since both come from 04b
    if cls_gt != conf_gt or cls_rows != conf_rows or cls_cols != conf_cols:
        logger.info("  WARNING: classification and confidence rasters "
                    "have different grids — check they are from the "
                    "same 04b run")

    # Replace nodata with sentinel values for safe arithmetic
    if cls_nd is not None:
        cls_arr[cls_arr == int(cls_nd)] = -1
    conf_arr[conf_arr == LANDSCAPE_CONF_NODATA] = np.nan
    if conf_nd is not None and conf_nd != LANDSCAPE_CONF_NODATA:
        conf_arr[conf_arr == conf_nd] = np.nan

    return cls_arr, cls_gt, cls_rows, cls_cols, conf_arr, conf_gt


def sample_landscape_at_crown(poly_geom, cls_arr, cls_gt,
                               cls_rows, cls_cols,
                               conf_arr, conf_gt):
    """
    Sample landscape classification and confidence within a crown polygon.

    Strategy:
        1. Find all Sentinel pixel centres that fall inside the crown polygon
        2. If at least one found: return majority class + mean confidence
        3. If none found (small crown with no pixel centre inside):
           fall back to the single nearest pixel to the crown centroid

    Returns:
        landscape_class : int, majority class ID (-1 if invalid)
        landscape_conf  : float, mean confidence (0.0 if invalid)
    """
    # Get crown bounding box for fast pixel candidate selection
    env  = poly_geom.GetEnvelope()  # (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = env[0], env[1], env[2], env[3]

    # Convert bbox to pixel index range in classification raster
    px_w = cls_gt[1]   # positive pixel width
    px_h = cls_gt[5]   # negative pixel height
    x0   = cls_gt[0]
    y0   = cls_gt[3]

    col_min = max(0, int((xmin - x0) / px_w))
    col_max = min(cls_cols - 1, int((xmax - x0) / px_w))
    row_min = max(0, int((ymax - y0) / px_h))  # note: y0 is top, px_h negative
    row_max = min(cls_rows - 1, int((ymin - y0) / px_h))

    if col_min > col_max or row_min > row_max:
        # Crown is entirely outside raster extent
        return -1, 0.0

    # Build pixel centre coordinates for candidate pixels
    cols_range = np.arange(col_min, col_max + 1)
    rows_range = np.arange(row_min, row_max + 1)
    col_grid, row_grid = np.meshgrid(cols_range, rows_range)
    col_flat = col_grid.flatten()
    row_flat = row_grid.flatten()

    # Pixel centres in geographic coordinates
    px_cx = x0 + (col_flat + 0.5) * px_w
    px_cy = y0 + (row_flat + 0.5) * px_h

    # Point-in-polygon test for each pixel centre
    # Build OGR points and test containment
    inside_mask = np.zeros(len(px_cx), dtype=bool)
    for i, (cx, cy) in enumerate(zip(px_cx, px_cy)):
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint(float(cx), float(cy))
        if poly_geom.Contains(pt):
            inside_mask[i] = True

    if np.any(inside_mask):
        # Use pixels whose centres fall inside the crown
        inside_cols = col_flat[inside_mask]
        inside_rows = row_flat[inside_mask]
    else:
        # Fallback: use single nearest pixel to centroid
        centroid = poly_geom.Centroid()
        cx = centroid.GetX()
        cy = centroid.GetY()
        nearest_col = int(np.clip((cx - x0) / px_w, 0, cls_cols - 1))
        nearest_row = int(np.clip((cy - y0) / px_h, 0, cls_rows - 1))
        inside_cols = np.array([nearest_col])
        inside_rows = np.array([nearest_row])

    # Extract class values
    class_vals = cls_arr[inside_rows, inside_cols]

    # Filter out nodata class values (-1)
    valid_class = class_vals[class_vals > 0]
    if len(valid_class) == 0:
        return -1, 0.0

    # Majority class
    unique, counts = np.unique(valid_class, return_counts=True)
    majority_class = int(unique[np.argmax(counts)])

    # Mean confidence for pixels with valid class
    valid_mask_conf = class_vals > 0
    conf_vals = conf_arr[inside_rows[valid_mask_conf],
                         inside_cols[valid_mask_conf]]
    # Exclude nodata confidence values
    valid_conf = conf_vals[~np.isnan(conf_vals)]
    mean_conf  = float(np.mean(valid_conf)) if len(valid_conf) > 0 else 0.0

    return majority_class, mean_conf


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
    xs         = xs_all[veg_filter]
    ys         = ys_all[veg_filter]
    zs         = zs_all[veg_filter]
    intensity  = intensity_all[veg_filter]
    ndvi       = ndvi_all[veg_filter]
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

    col_idx   = ((xs - x_min) / cell_size).astype(int)
    row_idx   = ((ys - y_min) / cell_size).astype(int)
    cell_keys = col_idx * 100000 + row_idx

    index = {}
    for i, key in enumerate(cell_keys):
        if key not in index:
            index[key] = []
        index[key].append(i)
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
    add("is_edge",            ogr.OFTInteger)
    # Landscape classification context from 04b Sentinel-2 RF
    add("landscape_class",    ogr.OFTInteger)
    add("landscape_conf",     ogr.OFTReal)
    # One-hot encoded landscape class columns
    for lc_id, lc_name in [(1,"olive"), (2,"montado"), (3,"eucalyptus"),
                            (4,"broadleaf"), (5,"pine"), (6,"shrubland")]:
        add(f"lc_{lc_name}", ogr.OFTInteger)
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
                   intensity, ndvi,
                   raster_extent, edge_buffer,
                   cls_arr, cls_gt, cls_rows, cls_cols,
                   conf_arr, conf_gt,
                   logger):
    """
    Iterate over crown polygons, compute LiDAR metrics and sample
    landscape classification context, write all results to GeoPackage.
    """
    n_features = layer.GetFeatureCount()
    logger.info(f"Processing {n_features:,} crown polygons...")

    # Apply good mask to class 5 arrays upfront
    xs_v   = xs[good]
    ys_v   = ys[good]
    zs_v   = zs_norm[good]
    int_v  = intensity[good]
    ndvi_v = ndvi[good]

    # Build spatial indices for point cloud lookups
    logger.info("Building spatial indices...")
    sidx_veg, xmin_v, ymin_v, cs_v = build_spatial_index(
        xs_v, ys_v, cell_size=50.0, logger=logger
    )
    sidx_all, xmin_a, ymin_a, cs_a = build_spatial_index(
        xs_all, ys_all, cell_size=50.0, logger=logger
    )

    layer.ResetReading()
    processed      = 0
    skipped        = 0
    lc_fallback    = 0  # crowns that used nearest-pixel fallback
    lc_no_data     = 0  # crowns with no valid landscape pixels at all

    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            skipped += 1
            continue

        xmin, xmax, ymin, ymax = geom.GetEnvelope()

        # ---- Edge detection ----
        rx_min, rx_max, ry_min, ry_max = raster_extent
        is_edge_crown = int(
            xmin <= rx_min + edge_buffer or
            xmax >= rx_max - edge_buffer or
            ymin <= ry_min + edge_buffer or
            ymax >= ry_max - edge_buffer
        )
        feature.SetField("is_edge", is_edge_crown)

        # ---- Landscape classification sampling ----
        lc, lc_conf = sample_landscape_at_crown(
            geom, cls_arr, cls_gt, cls_rows, cls_cols,
            conf_arr, conf_gt
        )
        feature.SetField("landscape_class", lc)
        feature.SetField("landscape_conf",  round(lc_conf, 4))
        # One-hot encode landscape class
        lc_map = {1:"olive", 2:"montado", 3:"eucalyptus",
                4:"broadleaf", 5:"pine", 6:"shrubland"}
        for lc_id, lc_name in lc_map.items():
            feature.SetField(f"lc_{lc_name}",
                            1 if lc == lc_id else 0)
        if lc == -1:
            lc_no_data += 1

        # ---- LiDAR point cloud metrics ----
        cands     = query_spatial_index(sidx_veg, xmin_v, ymin_v, cs_v,
                                        xmin, xmax, ymin, ymax)
        cands_all = query_spatial_index(sidx_all, xmin_a, ymin_a, cs_a,
                                        xmin, xmax, ymin, ymax)

        if len(cands) == 0 and len(cands_all) == 0:
            layer.SetFeature(feature)
            skipped += 1
            continue

        poly = wkb.loads(bytes(geom.ExportToWkb()))

        # Class 5 points inside polygon
        if len(cands) > 0:
            inside = vectorized.contains(poly, xs_v[cands], ys_v[cands])
            idx    = cands[inside]
        else:
            idx = np.array([], dtype=int)

        # All points inside polygon (for pct_high_veg)
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
        max_h            = float(np.max(z_in))
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

    logger.info(f"  Processed           : {processed:,} crowns")
    logger.info(f"  Skipped             : {skipped:,} crowns "
                f"(no points or < 3)")
    logger.info(f"  Landscape no data   : {lc_no_data:,} crowns "
                f"(outside landscape raster or all nodata)")
    return processed, skipped


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("05a_feature_extraction")

    try:
        logger.info(f"  crowns-layer       = {args.crowns_layer}")
        logger.info(f"  edge-buffer        = {args.edge_buffer} m")
        logger.info(f"  landscape-classif  = "
                    f"{os.path.basename(args.landscape_classif)}")
        logger.info(f"  landscape-conf     = "
                    f"{os.path.basename(args.landscape_conf)}")

        # 1 Open crowns layer
        ds_crowns, layer, layer_name = load_crowns_layer(
            INPUT_CROWNS, args.crowns_layer, logger
        )
        logger.info(f"  crowns layer = {layer_name}")

        # 2 Load landscape rasters
        (cls_arr, cls_gt, cls_rows, cls_cols,
         conf_arr, conf_gt) = load_landscape_rasters(
            args.landscape_classif, args.landscape_conf, logger
        )

        # 3 Find and load LAZ files
        laz_files = find_laz_files(RAW_DATA_DIR, logger)
        (xs_all, ys_all, classification,
         xs, ys, zs, intensity, ndvi,
         ret_number, n_returns) = load_point_cloud(laz_files, logger)

        # 4 Normalize heights using merged DTM
        zs_norm, good = normalize_heights(xs, ys, zs, INPUT_DTM, logger)

        # 5 Get raster extent for edge detection
        logger.info("Getting raster extent for edge detection...")
        raster_extent = get_raster_extent(INPUT_DTM, logger)

        # 6 Add fields to layer
        add_fields(layer, logger)

        # 7 Process each crown
        processed, skipped = process_crowns(
            layer,
            xs_all, ys_all,
            xs, ys, zs_norm, good,
            intensity, ndvi,
            raster_extent, args.edge_buffer,
            cls_arr, cls_gt, cls_rows, cls_cols,
            conf_arr, conf_gt,
            logger
        )

        ds_crowns.Destroy()

        logger.info(f"  Crowns processed : {processed:,}")
        logger.info(f"  Crowns skipped   : {skipped:,}")
        logger.info(f"  Results in       : {INPUT_CROWNS}")
        logger.info(f"  Layer            : {layer_name}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise