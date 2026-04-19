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
"""

import os
import sys
import argparse
import numpy as np
from osgeo import gdal, ogr, osr
from skimage.segmentation import watershed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_NDSM       = os.path.join(BASE_DIR, "data", "processed", "nDSM.tif")
INPUT_MASK       = os.path.join(BASE_DIR, "data", "processed", "vegetation_mask.tif")
INPUT_TREETOPS   = os.path.join(BASE_DIR, "data", "processed", "treetops.gpkg")
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
    """
    Run marker-controlled watershed on inverted nDSM.
    Tree tops are seeds (markers), watershed grows outward until
    it meets neighbouring crowns or vegetation boundary.
    """
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
    """
    Vectorize segment raster to crown polygons in GeoPackage.
    Opens existing GeoPackage if present, replaces layer if it exists.
    """
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

        # 5 Save segment raster
        save_segment_raster(segments, gt, proj, segment_raster, logger)

        # 6 Vectorize to GeoPackage
        n_polys = vectorize_to_gpkg(
            segment_raster, OUTPUT_CROWNS_GPKG, layer_name, proj, logger
        )

        logger.info(f"Done! {n_polys:,} crown polygons created.")
        logger.info(f"  Segment raster : {segment_raster}")
        logger.info(f"  Crown polygons : {OUTPUT_CROWNS_GPKG}")
        logger.info(f"  Layer          : {layer_name}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise