"""
03b_crown_diagnostics.py
========================
Diagnostic script to inspect crown segmentation quality after
03_crown_segmentation.py.

Checks:
    1. Crown size distribution (area, point count)
    2. Edge crowns: polygons touching or near the raster boundary
    3. Segmentation artefacts: very small polygons
    4. Coverage gaps: crowns with no LAZ points at all
    5. Summary recommendations

Usage:
    python scripts/03b_crown_diagnostics.py --crowns-layer crowns_treetops_w9_mh2.0_t0.001_c0.001_mh2.0
    python scripts/03b_crown_diagnostics.py --edge-buffer 2.0 --min-area 4.0

Arguments:
    --crowns-layer    layer name inside crowns.gpkg (default: first layer)
    --edge-buffer     distance in metres from raster edge to flag as edge crown (default: 2.0)
    --min-area        minimum crown area in m2 below which flagged as artefact (default: 4.0)
"""

import os
import sys
import argparse
import numpy as np
from osgeo import gdal, ogr, osr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CROWNS   = os.path.join(BASE_DIR, "data", "processed", "crowns.gpkg")
INPUT_NDSM     = os.path.join(BASE_DIR, "data", "processed", "nDSM.tif")
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnostic checks on crown segmentation output."
    )
    parser.add_argument(
        "--crowns-layer",
        type=str,
        default=None,
        help="Layer name inside crowns.gpkg (default: first layer found)"
    )
    parser.add_argument(
        "--edge-buffer",
        type=float,
        default=2.0,
        help="Distance in metres from raster edge to flag as edge crown (default: 2.0)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=4.0,
        help="Minimum crown area in m2 below which flagged as artefact (default: 4.0)"
    )
    return parser.parse_args()


def get_raster_extent(raster_path, logger):
    """Get the geographic extent and pixel size of the reference raster."""
    logger.info(f"Reading raster extent: {os.path.basename(raster_path)}")
    ds = gdal.Open(raster_path)
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
    pixel = gt[1]

    logger.info(f"  Extent     : ({x_min:.1f}, {y_min:.1f}) - "
                f"({x_max:.1f}, {y_max:.1f})")
    logger.info(f"  Pixel size : {pixel:.2f} m")
    logger.info(f"  Dimensions : {cols} x {rows} pixels")
    return x_min, x_max, y_min, y_max, pixel


def load_crowns(gpkg_path, layer_name, logger):
    """Load crown polygons, return ds and layer."""
    logger.info(f"Loading crowns: {os.path.basename(gpkg_path)}")
    ds = ogr.Open(gpkg_path, 0)  # read only
    if ds is None:
        msg = f"Cannot open GeoPackage: {gpkg_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    available = [ds.GetLayerByIndex(i).GetName()
                 for i in range(ds.GetLayerCount())]
    logger.info(f"  Available layers: {available}")

    if layer_name is None:
        layer_name = available[0]
        logger.info(f"  No layer specified, using: '{layer_name}'")
    elif layer_name not in available:
        msg = f"Layer '{layer_name}' not found. Available: {available}"
        logger.error(msg)
        raise ValueError(msg)

    layer = ds.GetLayerByName(layer_name)
    n     = layer.GetFeatureCount()
    logger.info(f"  Total crown polygons: {n:,}")
    return ds, layer, layer_name, n


def collect_crown_stats(layer, x_min, x_max, y_min, y_max,
                        edge_buffer, min_area, logger):
    """
    Iterate over all crowns and collect statistics.
    Returns arrays of metrics and flag counts.
    """
    logger.info("Collecting crown statistics...")

    areas        = []
    n_points_all = []
    is_edge      = []
    has_n_points = []  # whether n_points field exists and is filled

    # Check if n_points field exists (only present after script 04)
    layer_defn    = layer.GetLayerDefn()
    field_names   = [layer_defn.GetFieldDefn(i).GetName()
                     for i in range(layer_defn.GetFieldCount())]
    has_np_field  = "n_points" in field_names
    if not has_np_field:
        logger.info("  Note: 'n_points' field not found — "
                    "run 04_feature_extraction.py first for point count stats")

    # Edge boundary as geometry for intersection test
    edge_x_min = x_min + edge_buffer
    edge_x_max = x_max - edge_buffer
    edge_y_min = y_min + edge_buffer
    edge_y_max = y_max - edge_buffer

    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        area = geom.Area()
        areas.append(area)

        # Check if n_points field is filled
        if has_np_field:
            np_val = feature.GetField("n_points")
            n_points_all.append(np_val if np_val is not None else 0)
            has_n_points.append(np_val is not None)

        # Edge check — does crown bbox touch the edge buffer zone
        env = geom.GetEnvelope()  # xmin, xmax, ymin, ymax
        crown_xmin, crown_xmax = env[0], env[1]
        crown_ymin, crown_ymax = env[2], env[3]

        on_edge = (
            crown_xmin <= edge_x_min or
            crown_xmax >= edge_x_max or
            crown_ymin <= edge_y_min or
            crown_ymax >= edge_y_max
        )
        is_edge.append(on_edge)

    areas        = np.array(areas)
    is_edge      = np.array(is_edge)
    n_points_arr = np.array(n_points_all) if has_np_field else None

    return areas, is_edge, n_points_arr, has_np_field


def print_section(title, logger):
    logger.info("")
    logger.info("-" * 50)
    logger.info(title)
    logger.info("-" * 50)


def report_area_distribution(areas, min_area, logger):
    """Report crown area distribution and flag artefacts."""
    print_section("CROWN AREA DISTRIBUTION", logger)

    total = len(areas)
    logger.info(f"  Total crowns       : {total:,}")
    logger.info(f"  Min area           : {areas.min():.2f} m2")
    logger.info(f"  Max area           : {areas.max():.2f} m2")
    logger.info(f"  Mean area          : {areas.mean():.2f} m2")
    logger.info(f"  Median area        : {np.median(areas):.2f} m2")
    logger.info("")
    logger.info(f"  Size breakdown:")
    logger.info(f"    < {min_area} m2 (likely artefacts) : "
                f"{np.sum(areas < min_area):,} "
                f"({100*np.sum(areas < min_area)/total:.1f}%)")
    logger.info(f"    {min_area}-25 m2 (small trees)     : "
                f"{np.sum((areas >= min_area) & (areas < 25)):,} "
                f"({100*np.sum((areas >= min_area) & (areas < 25))/total:.1f}%)")
    logger.info(f"    25-100 m2 (medium trees)           : "
                f"{np.sum((areas >= 25) & (areas < 100)):,} "
                f"({100*np.sum((areas >= 25) & (areas < 100))/total:.1f}%)")
    logger.info(f"    100-400 m2 (large trees)           : "
                f"{np.sum((areas >= 100) & (areas < 400)):,} "
                f"({100*np.sum((areas >= 100) & (areas < 400))/total:.1f}%)")
    logger.info(f"    > 400 m2 (very large / merged)     : "
                f"{np.sum(areas >= 400):,} "
                f"({100*np.sum(areas >= 400)/total:.1f}%)")

    return np.sum(areas < min_area)


def report_edge_crowns(areas, is_edge, edge_buffer, logger):
    """Report crowns touching the raster boundary."""
    print_section("EDGE CROWNS", logger)

    total      = len(areas)
    n_edge     = np.sum(is_edge)
    n_edge_small = np.sum(is_edge & (areas < 25))

    logger.info(f"  Edge buffer used   : {edge_buffer} m")
    logger.info(f"  Total edge crowns  : {n_edge:,} "
                f"({100*n_edge/total:.1f}%)")
    logger.info(f"  Edge + small (<25m2): {n_edge_small:,} "
                f"({100*n_edge_small/total:.1f}%) "
                f"-> most likely to be half-crowns")
    logger.info("")
    logger.info(f"  Note: edge crowns are real trees but their metrics")
    logger.info(f"  may be incomplete — consider excluding from")
    logger.info(f"  classification or flagging with an 'is_edge' field.")

    return n_edge


def report_point_counts(n_points_arr, logger):
    """Report point count distribution if available."""
    print_section("POINT COUNT DISTRIBUTION (from script 04)", logger)

    total = len(n_points_arr)
    logger.info(f"  0 points (no data)  : "
                f"{np.sum(n_points_arr == 0):,} "
                f"({100*np.sum(n_points_arr == 0)/total:.1f}%)")
    logger.info(f"  1-2 points          : "
                f"{np.sum((n_points_arr > 0) & (n_points_arr < 3)):,} "
                f"({100*np.sum((n_points_arr > 0) & (n_points_arr < 3))/total:.1f}%)")
    logger.info(f"  3-9 points          : "
                f"{np.sum((n_points_arr >= 3) & (n_points_arr < 10)):,} "
                f"({100*np.sum((n_points_arr >= 3) & (n_points_arr < 10))/total:.1f}%)")
    logger.info(f"  10-49 points        : "
                f"{np.sum((n_points_arr >= 10) & (n_points_arr < 50)):,} "
                f"({100*np.sum((n_points_arr >= 10) & (n_points_arr < 50))/total:.1f}%)")
    logger.info(f"  50+ points          : "
                f"{np.sum(n_points_arr >= 50):,} "
                f"({100*np.sum(n_points_arr >= 50)/total:.1f}%)")
    logger.info("")
    logger.info(f"  Mean points per crown  : {n_points_arr.mean():.1f}")
    logger.info(f"  Median points per crown: {np.median(n_points_arr):.1f}")
    logger.info(f"  Max points per crown   : {n_points_arr.max():,}")


def report_recommendations(n_total, n_artefacts, n_edge,
                            n_points_arr, min_area, has_np_field, logger):
    """Print actionable recommendations based on findings."""
    print_section("RECOMMENDATIONS", logger)

    if n_artefacts > 0:
        pct = 100 * n_artefacts / n_total
        logger.info(f"  1. FILTER ARTEFACTS: {n_artefacts:,} crowns ({pct:.1f}%) "
                    f"are below {min_area} m2.")
        logger.info(f"     Add --min-area {min_area} to script 03 to remove "
                    f"these before extraction.")

    if n_edge > 0:
        pct = 100 * n_edge / n_total
        logger.info(f"  2. EDGE CROWNS: {n_edge:,} crowns ({pct:.1f}%) touch "
                    f"the raster boundary.")
        logger.info(f"     Consider adding an 'is_edge' flag field in script 04")
        logger.info(f"     and excluding these from classification in script 05.")

    if has_np_field and n_points_arr is not None:
        n_no_points = np.sum(n_points_arr == 0)
        if n_no_points > 0:
            logger.info(f"  3. NO-POINT CROWNS: {n_no_points:,} crowns have "
                        f"zero class 5 points.")
            logger.info(f"     These may be buildings, shadows, or data gaps.")
            logger.info(f"     Filter using: pct_high_veg = 0 in QGIS or script 05.")

    if not has_np_field:
        logger.info(f"  3. Run 04_feature_extraction.py to get point count")
        logger.info(f"     stats for fuller diagnostics.")

    logger.info("")
    logger.info(f"  Overall: {n_total - n_artefacts:,} crowns remain after "
                f"artefact filtering.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("03b_crown_diagnostics")

    try:
        logger.info("=" * 60)
        logger.info("STEP 03b: Crown segmentation diagnostics")
        logger.info("=" * 60)
        logger.info(f"Parameters:")
        logger.info(f"  edge-buffer = {args.edge_buffer} m")
        logger.info(f"  min-area    = {args.min_area} m2")

        # 1. Get raster extent for edge detection
        x_min, x_max, y_min, y_max, pixel = get_raster_extent(
            INPUT_NDSM, logger
        )

        # 2. Load crowns
        ds, layer, layer_name, n_total = load_crowns(
            INPUT_CROWNS, args.crowns_layer, logger
        )
        logger.info(f"  Layer: {layer_name}")

        # 3. Collect stats
        areas, is_edge, n_points_arr, has_np_field = collect_crown_stats(
            layer, x_min, x_max, y_min, y_max,
            args.edge_buffer, args.min_area, logger
        )
        ds = None

        # 4. Report
        n_artefacts = report_area_distribution(areas, args.min_area, logger)
        n_edge      = report_edge_crowns(areas, is_edge, args.edge_buffer,
                                         logger)
        if has_np_field and n_points_arr is not None:
            report_point_counts(n_points_arr, logger)

        report_recommendations(
            n_total, n_artefacts, n_edge,
            n_points_arr, args.min_area, has_np_field, logger
        )

        logger.info("=" * 60)
        logger.info("Diagnostics complete.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise