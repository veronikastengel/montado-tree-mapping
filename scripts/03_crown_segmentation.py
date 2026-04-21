"""
03_crown_segmentation.py
========================
Segments individual tree crowns from nDSM using marker-controlled
watershed algorithm, seeded by treetop points.

Usage:
    python scripts/03_crown_segmentation.py --treetops-layer treetops_w15_mh3.0_t0.001 --compactness 0.01

Arguments:
    --treetops-layer    name of layer inside treetops.gpkg to use as seeds
                        (default: uses the first/only layer found)
    --compactness       float, watershed compactness parameter (default: 0.001)
                        lower = follow height contours, higher = rounder segments
    --minheight         float, minimum height for vegetation mask (default: 2.0)
    --min-area          float, polygons at or below this area in m2 are either
                        merged into a single touching neighbour or removed (default: 1.0)

"""

import os
import sys
import argparse
import numpy as np
from osgeo import gdal, ogr, osr
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_NDSM       = os.path.join(BASE_DIR, "data", "processed", "nDSM.tif")
INPUT_MASK       = os.path.join(BASE_DIR, "data", "processed", "vegetation_mask.tif")
INPUT_TREETOPS   = os.path.join(BASE_DIR, "data", "processed", "treetops.gpkg") #treetops.gpkg or treetops_lidar.gpkg
OUTPUT_DIR       = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_CROWNS_GPKG = os.path.join(OUTPUT_DIR, "crowns.gpkg")
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment tree crowns from nDSM using watershed."
    )
    parser.add_argument(
        "--treetops-layer",
        type=str,
        default=None,
        help="Layer name inside treetops.gpkg to use as seeds. "
             "If not specified, uses the first layer found."
    )
    parser.add_argument(
        "--compactness",
        type=float,
        default=0.001,
        help="Watershed compactness (default: 0.001). "
             "Lower = follow contours, higher = rounder segments."
    )
    parser.add_argument(
        "--minheight",
        type=float,
        default=2.0,
        help="Minimum height in metres for vegetation mask (default: 2.0)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1.0,
        help="Polygons at or below this area in m2 are merged into a single "
             "touching neighbour or removed if isolated (default: 1.0)"
    )
    return parser.parse_args()


def load_raster(path, logger, nodata_fill=0.0, dtype=np.float32):
    """Load a raster band into a numpy array."""
    logger.info(f"Loading raster: {os.path.basename(path)}")
    ds = gdal.Open(path)
    if ds is None:
        msg = f"Cannot open raster: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    band = ds.GetRasterBand(1)
    arr  = band.ReadAsArray().astype(dtype)
    nd   = band.GetNoDataValue()
    if nd is not None:
        arr[arr == nd] = nodata_fill
    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    logger.info(f"  Dimensions : {ds.RasterXSize} x {ds.RasterYSize} pixels")
    logger.info(f"  Pixel size : {gt[1]:.2f} m")
    ds = None
    return arr, gt, proj


def load_treetops_layer(gpkg_path, layer_name, logger):
    """
    Open treetops GeoPackage and return the requested layer.
    If layer_name is None, uses first layer and reports its name.
    """
    logger.info(f"Loading treetops: {os.path.basename(gpkg_path)}")
    ds = ogr.Open(gpkg_path, 0)  # read only
    if ds is None:
        msg = f"Cannot open treetops GeoPackage: {gpkg_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # List available layers
    available = [ds.GetLayerByIndex(i).GetName()
                 for i in range(ds.GetLayerCount())]
    logger.info(f"  Available layers: {available}")

    if layer_name is None:
        layer_name = available[0]
        logger.info(f"  No layer specified, using: '{layer_name}'")
    elif layer_name not in available:
        msg = (f"Layer '{layer_name}' not found in {gpkg_path}. "
               f"Available: {available}")
        logger.error(msg)
        raise ValueError(msg)
    else:
        logger.info(f"  Using layer: '{layer_name}'")

    layer = ds.GetLayerByName(layer_name)
    n_points = layer.GetFeatureCount()
    logger.info(f"  Tree top points: {n_points:,}")
    return ds, layer, layer_name


def rasterize_treetops(layer, gt, proj, rows, cols, logger):
    """
    Rasterize treetop points as unique integer seeds.
    Each point gets a unique ID burned into the raster.
    """
    logger.info("Rasterizing treetop points as seeds...")
    seeds = np.zeros((rows, cols), dtype=np.int32)

    x_origin     = gt[0]
    y_origin     = gt[3]
    pixel_width  = gt[1]
    pixel_height = gt[5]  # negative

    layer.ResetReading()
    seed_id  = 1
    skipped  = 0
    in_range = 0

    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            skipped += 1
            continue
        cx = geom.GetX()
        cy = geom.GetY()

        col = int((cx - x_origin) / pixel_width)
        row = int((cy - y_origin) / pixel_height)

        if 0 <= col < cols and 0 <= row < rows:
            seeds[row, col] = seed_id
            in_range += 1
        else:
            skipped += 1
        seed_id += 1

    logger.info(f"  Seeds placed    : {in_range:,}")
    logger.info(f"  Seeds skipped   : {skipped:,} (outside raster extent)")
    logger.info(f"  Unique seed IDs : {len(np.unique(seeds[seeds > 0])):,}")
    return seeds


def run_watershed(ndsm, seeds, veg_mask, compactness, logger):
    """Run marker-controlled watershed on inverted nDSM."""
    logger.info(f"Running watershed (compactness={compactness})...")

    rows, cols = ndsm.shape
    veg_pixels = np.sum(veg_mask)
    logger.info(f"  Vegetation pixels : {veg_pixels:,} of {rows*cols:,} "
                f"({100*veg_pixels/(rows*cols):.1f}%)")

    # Invert nDSM -> tree tops become valleys, watershed fills upward
    ndsm_inv = np.where(veg_mask, np.max(ndsm) - ndsm, 0).astype(np.float32)

    segments = watershed(
        image=ndsm_inv,
        markers=seeds,
        mask=veg_mask,
        connectivity=2,     # 8-connectivity
        compactness=compactness
    )

    n_segs = len(np.unique(segments[segments > 0]))
    logger.info(f"  Crown segments found: {n_segs:,}")
    return segments


def fill_crown_holes(segments, logger):
    """
    Fill enclosed holes within crowns
    For each connected blob of background pixels: If bordered by exactly 
    one crown ID on all sides -> fill with that ID

    Uses 4-connectivity for background blobs (cardinal directions only)
    so diagonal-only connections are treated as separate blobs.
    """
    logger.info("Filling enclosed holes within crowns...")

    from scipy.ndimage import label, find_objects

    rows, cols = segments.shape

    # Background mask -> pixels not belonging to any crown
    bg_mask = segments == 0

    # Label connected background regions using 4-connectivity
    struct_4 = np.array([[0,1,0],
                         [1,1,1],
                         [0,1,0]], dtype=int)
    bg_labels, n_blobs = label(bg_mask, structure=struct_4)
    logger.info(f"  Background blobs found: {n_blobs:,}")

    filled       = segments.copy()
    total_filled = 0
    blobs_filled = 0
    blobs_left   = 0
    blobs_edge   = 0

    # Process each background blob
    blob_slices = find_objects(bg_labels)

    for blob_id, slices in enumerate(blob_slices, start=1):
        if slices is None:
            continue

        # Extract blob region with 1 pixel padding for border check
        row_sl, col_sl = slices
        r0 = max(0, row_sl.start - 1)
        r1 = min(rows, row_sl.stop + 1)
        c0 = max(0, col_sl.start - 1)
        c1 = min(cols, col_sl.stop + 1)

        blob_region  = bg_labels[r0:r1, c0:c1] == blob_id
        crown_region = segments[r0:r1, c0:c1]

        # Check if blob touches raster edge
        # A blob touching the edge is open space, not an enclosed hole
        full_row_sl = slice(row_sl.start, row_sl.stop)
        full_col_sl = slice(col_sl.start, col_sl.stop)
        blob_pixels = bg_labels[full_row_sl, full_col_sl] == blob_id

        touches_edge = (
            row_sl.start == 0 or
            row_sl.stop  == rows or
            col_sl.start == 0 or
            col_sl.stop  == cols
        )
        if touches_edge:
            blobs_edge += 1
            continue

        # Find crown IDs bordering this blob using dilation
        # Dilate blob by 1 pixel and check what crown IDs appear
        from scipy.ndimage import binary_dilation
        struct_4_local = np.array([[0,1,0],
                                   [1,1,1],
                                   [0,1,0]], dtype=int)
        dilated      = binary_dilation(blob_region, structure=struct_4_local)
        border_mask  = dilated & ~blob_region
        border_crowns = crown_region[border_mask]

        # Get unique crown IDs on border, excluding background
        unique_border = np.unique(border_crowns)
        unique_border = unique_border[unique_border > 0]

        if len(unique_border) == 1:
            # Enclosed by exactly one crown -> fill it
            crown_id = unique_border[0]
            # Apply fill to global segments array
            blob_global = bg_labels[
                row_sl.start:row_sl.stop,
                col_sl.start:col_sl.stop
            ] == blob_id
            n_pixels = np.sum(blob_global)
            filled[
                row_sl.start:row_sl.stop,
                col_sl.start:col_sl.stop
            ][blob_global] = crown_id
            total_filled += n_pixels
            blobs_filled += 1
        else:
            blobs_left += 1

    logger.info(f"  Blobs touching edge    : {blobs_edge:,} (skipped)")
    logger.info(f"  Blobs filled           : {blobs_filled:,}")
    logger.info(f"  Blobs left (open space): {blobs_left:,}")
    logger.info(f"  Total pixels filled    : {total_filled:,}")
    return filled


def save_segment_raster(segments, gt, proj, output_path, logger):
    """Save watershed segments as integer GeoTIFF."""
    logger.info(f"Saving segment raster...")
    rows, cols = segments.shape
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        output_path, cols, rows, 1, gdal.GDT_Int32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj)
    band_out = ds_out.GetRasterBand(1)
    band_out.WriteArray(segments)
    band_out.SetNoDataValue(0)
    ds_out.FlushCache()
    ds_out = None
    logger.info(f"  Saved: {output_path}")


def vectorize_to_gpkg(segment_raster_path, output_gpkg, layer_name, proj,
                      logger):
    """Vectorize segment raster to crown polygons in GeoPackage."""
    logger.info("Vectorizing segments to crown polygons...")

    ds_seg   = gdal.Open(segment_raster_path)
    band_seg = ds_seg.GetRasterBand(1)

    gpkg_driver = ogr.GetDriverByName("GPKG")

    if os.path.exists(output_gpkg):
        ds_gpkg = gpkg_driver.Open(output_gpkg, 1)
        if ds_gpkg is None:
            msg = f"Could not open existing GeoPackage: {output_gpkg}"
            logger.error(msg)
            raise RuntimeError(msg)
        for i in range(ds_gpkg.GetLayerCount()):
            if ds_gpkg.GetLayerByIndex(i).GetName() == layer_name:
                ds_gpkg.DeleteLayer(i)
                logger.info(f"  Deleted existing layer: '{layer_name}'")
                break
        logger.info(f"  Opened existing GeoPackage: {output_gpkg}")
    else:
        ds_gpkg = gpkg_driver.CreateDataSource(output_gpkg)
        logger.info(f"  Created new GeoPackage: {output_gpkg}")

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    layer = ds_gpkg.CreateLayer(
        layer_name, srs=srs, geom_type=ogr.wkbPolygon
    )
    layer.CreateField(ogr.FieldDefn("crown_id", ogr.OFTInteger))

    gdal.Polygonize(band_seg, band_seg, layer, 0, [], callback=None)

    n_polys = layer.GetFeatureCount()
    ds_gpkg.Destroy()
    ds_seg = None

    logger.info(f"  Polygons created : {n_polys:,}")
    logger.info(f"  Layer '{layer_name}' saved to: {output_gpkg}")
    return n_polys


def clean_small_polygons(output_gpkg, layer_name, min_area, pixel_size,
                         logger):
    """
    Post-process small polygons in vector layer.
    merge: Small polygon (<=min_area) touching exactly
    one neighbour on a shared edge -> merge into that neighbour.
    remove: Small polygon touching no neighbours or
    touching multiple neighbours -> delete.
    """
    logger.info(f"Cleaning small polygons (min_area={min_area} m2)...")

    ds    = ogr.Open(output_gpkg, 1)
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        msg = f"Layer '{layer_name}' not found"
        logger.error(msg)
        raise ValueError(msg)

    # Collect all small polygon FIDs and their geometries
    small_fids  = []
    small_geoms = {}

    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        area = geom.Area()
        if area <= min_area:
            fid = feature.GetFID()
            small_fids.append(fid)
            small_geoms[fid] = geom.Clone()

    logger.info(f"  Small polygons found: {len(small_fids):,}")

    merged  = 0
    removed = 0

    for fid in small_fids:
        small_geom = small_geoms[fid]

        # Find candidate neighbours via spatial filter
        # Expand bbox by one pixel to catch touching polygons
        env = small_geom.GetEnvelope()
        layer.SetSpatialFilterRect(
            env[0] - pixel_size, env[2] - pixel_size,
            env[1] + pixel_size, env[3] + pixel_size
        )

        neighbours = []
        layer.ResetReading()
        for candidate in layer:
            c_fid = candidate.GetFID()
            if c_fid == fid:
                continue
            c_geom = candidate.GetGeometryRef()
            if c_geom is None:
                continue
            # Check for shared edge (not just corner touch)
            # Intersection length > 0 means shared edge not just point
            intersection = small_geom.Intersection(c_geom)
            if intersection is None:
                continue
            geom_type = intersection.GetGeometryType()
            # Shared edge = LineString or MultiLineString (not Point)
            if geom_type in (ogr.wkbLineString,
                             ogr.wkbMultiLineString,
                             ogr.wkbLinearRing):
                neighbours.append(c_fid)

        layer.SetSpatialFilter(None)

        if len(neighbours) == 1:
            # merge into the single neighbour
            neighbour_feat = layer.GetFeature(neighbours[0])
            if neighbour_feat is None:
                removed += 1
                layer.DeleteFeature(fid)
                continue
            n_geom   = neighbour_feat.GetGeometryRef()
            merged_g = n_geom.Union(small_geom)
            if merged_g is not None:
                neighbour_feat.SetGeometry(merged_g)
                layer.SetFeature(neighbour_feat)
            layer.DeleteFeature(fid)
            merged += 1

        else:
            # isolated or ambiguous -> remove
            layer.DeleteFeature(fid)
            removed += 1

    ds.Destroy()

    logger.info(f"  Merged into neighbour : {merged:,}")
    logger.info(f"  Removed (isolated)    : {removed:,}")
    logger.info(f"  Total removed from layer: {merged + removed:,}")
    return merged, removed

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("03_crown_segmentation")

    try:
        # Resolve treetops layer name first so we can build param_str
        ds_tops, tops_layer, treetops_layer_name = load_treetops_layer(
            INPUT_TREETOPS, args.treetops_layer, logger
        )

        # Build output names from parameters
        # Use treetops layer name as base so outputs are traceable
        param_str        = (f"{treetops_layer_name}"
                            f"_c{args.compactness}"
                            f"_mh{args.minheight}")
        layer_name       = f"crowns_{param_str}"
        segment_raster   = os.path.join(OUTPUT_DIR,
                                        f"crown_segments_{param_str}.tif")

        logger.info(f"Parameters:")
        logger.info(f"  treetops layer = {treetops_layer_name}")
        logger.info(f"  compactness    = {args.compactness}")
        logger.info(f"  minheight      = {args.minheight} m")
        logger.info(f"  output layer   = {layer_name}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1 Load nDSM
        ndsm, gt, proj = load_raster(INPUT_NDSM, logger)
        rows, cols = ndsm.shape
        pixel_size     = gt[1]

        # 2 Load and apply vegetation mask
        mask_arr, _, _ = load_raster(
            INPUT_MASK, logger, nodata_fill=0.0, dtype=np.uint8
        )
        veg_mask = (mask_arr == 1) & (ndsm > args.minheight)
        logger.info(f"Combined vegetation mask + height threshold applied")

        # 3 Rasterize treetop seeds
        seeds = rasterize_treetops(tops_layer, gt, proj, rows, cols, logger)
        ds_tops = None  # close gpkg

        # Check to seeds match nDSM
        n_seeds = len(np.unique(seeds[seeds > 0]))
        if n_seeds == 0:
            msg = "No seeds found in raster extent -> check CRS matches nDSM"
            logger.error(msg)
            raise ValueError(msg)
        if ndsm.shape != seeds.shape:
            msg = (f"Shape mismatch: nDSM {ndsm.shape} "
                   f"vs seeds {seeds.shape}")
            logger.error(msg)
            raise ValueError(msg)

        # 4 Run watershed
        segments = run_watershed(ndsm, seeds, veg_mask, args.compactness,
                                 logger)
        
        # 5 Fill enclosed holes within crowns (raster post-processing)
        segments = fill_crown_holes(segments, logger)

        # 6 Save segment raster
        save_segment_raster(segments, gt, proj, segment_raster, logger)

        # 7 Vectorize to GeoPackage
        n_polys = vectorize_to_gpkg(
            segment_raster, OUTPUT_CROWNS_GPKG, layer_name, proj, logger
        )
        logger.info(f"  Polygons after vectorization: {n_polys:,}")

        # 8 Clean small polygons (vector post-processing)
        merged, removed = clean_small_polygons(
            OUTPUT_CROWNS_GPKG, layer_name, args.min_area, pixel_size, logger
        )
        n_final = n_polys - merged - removed
        logger.info(f"  Polygons after cleanup: {n_final:,}")

        logger.info(f"  Segment raster : {segment_raster}")
        logger.info(f"  Crown polygons : {OUTPUT_CROWNS_GPKG}")
        logger.info(f"  Layer          : {layer_name}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise