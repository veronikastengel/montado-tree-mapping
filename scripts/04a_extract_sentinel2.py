"""
04a_extract_sentinel2.py
========================
Extracts and preprocesses Sentinel-2 bands from a SAFE archive for
use in landscape classification.

Pipeline:
    1. Locate R10m and R20m bands in SAFE folder structure
    2. Clip each band to the LiDAR study area extent
    3. Reproject from EPSG:32629 to EPSG:3763
    4. Resample R20m bands to 10m to match R10m
    5. Stack all bands into a single multiband GeoTIFF
    6. Apply SCL mask to remove clouds, shadows, water

Bands used:
    R10m: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR broad)
    R20m: B05, B06, B07 (Red Edge 1/2/3), B8A (NIR narrow),
          B11, B12 (SWIR 1/2)
    R20m: SCL (Scene Classification -> used for masking only)

Output:
    data/processed/sentinel2_stack.tif  -> 10 band multiband GeoTIFF
                                          in EPSG:3763, clipped to
                                          study area, clouds masked

Usage:
    python scripts/04a_extract_sentinel2.py
"""

import os
import sys
import numpy as np
from osgeo import gdal, osr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE_ROOT    = os.path.join(
    BASE_DIR, "data", "raw", "sentinel2",
    "S2B_.SAFE",
    "GRANULE", "L2A_T29SPD_A039531_20240930T112114", "IMG_DATA"
)
R10M_DIR     = os.path.join(SAFE_ROOT, "R10m")
R20M_DIR     = os.path.join(SAFE_ROOT, "R20m")
INPUT_NDSM   = os.path.join(BASE_DIR, "data", "processed", "nDSM.tif")
OUTPUT_DIR   = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_STACK = os.path.join(OUTPUT_DIR, "sentinel2_stack.tif")

# Bands to use — (directory, filename pattern, band name)
BANDS = [
    (R10M_DIR, "B02_10m.jp2", "B02_blue"),
    (R10M_DIR, "B03_10m.jp2", "B03_green"),
    (R10M_DIR, "B04_10m.jp2", "B04_red"),
    (R10M_DIR, "B08_10m.jp2", "B08_nir"),
    (R20M_DIR, "B05_20m.jp2", "B05_re1"),
    (R20M_DIR, "B06_20m.jp2", "B06_re2"),
    (R20M_DIR, "B07_20m.jp2", "B07_re3"),
    (R20M_DIR, "B8A_20m.jp2", "B8A_nir2"),
    (R20M_DIR, "B11_20m.jp2", "B11_swir1"),
    (R20M_DIR, "B12_20m.jp2", "B12_swir2"),
]
SCL_PATTERN  = "SCL_20m.jp2"

# SCL classes to mask out (set to NoData)
# 0=no data, 1=saturated, 2=dark, 3=cloud shadow,
# 8=cloud medium, 9=cloud high, 10=thin cirrus, 11=snow
SCL_MASK_CLASSES = [0, 1, 2, 3, 8, 9, 10, 11]

TARGET_CRS   = "EPSG:3763"
TARGET_RES   = 10.0  # metres
# ============================================================


def get_study_extent(ndsm_path, target_crs, logger):
    """
    Get study area extent from nDSM in target CRS.
    Returns (x_min, y_min, x_max, y_max) in target_crs.
    """
    logger.info(f"Getting study area extent from nDSM...")
    ds   = gdal.Open(ndsm_path)
    if ds is None:
        msg = f"Cannot open nDSM: {ndsm_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    gt   = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    proj = ds.GetProjection()
    ds   = None

    x_min = gt[0]
    x_max = gt[0] + cols * gt[1]
    y_max = gt[3]
    y_min = gt[3] + rows * gt[5]

    logger.info(f"  nDSM extent ({proj[:20]}...):")
    logger.info(f"  ({x_min:.1f}, {y_min:.1f}) - ({x_max:.1f}, {y_max:.1f})")

    # Add buffer so reprojection doesn't clip edges
    buffer = 500  # metres
    x_min -= buffer
    x_max += buffer
    y_min -= buffer
    y_max += buffer

    logger.info(f"  Buffered extent: "
                f"({x_min:.1f}, {y_min:.1f}) - ({x_max:.1f}, {y_max:.1f})")
    return x_min, y_min, x_max, y_max, proj


def find_band_file(directory, pattern, logger):
    """Find band file matching pattern in directory."""
    import glob
    matches = glob.glob(os.path.join(directory, f"*{pattern}"))
    if not matches:
        msg = f"No file matching '*{pattern}' in {directory}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if len(matches) > 1:
        logger.info(f"  Multiple matches for {pattern}, using first: "
                    f"{os.path.basename(matches[0])}")
    return matches[0]


def warp_band(src_path, target_crs, x_min, y_min, x_max, y_max,
              src_crs, target_res, logger):
    """
    Reproject, clip and resample a single band to target CRS and
    resolution. Returns numpy array and geotransform.
    Uses GDAL Warp in memory.
    """
    # Reproject extent from nDSM CRS to source S2 CRS for clipping
    # We warp directly — GDAL handles the CRS transformation
    warp_options = gdal.WarpOptions(
        dstSRS=target_crs,
        outputBounds=(x_min, y_min, x_max, y_max),
        outputBoundsSRS=src_crs,
        xRes=target_res,
        yRes=target_res,
        resampleAlg="bilinear",
        format="MEM"
    )
    ds_warped = gdal.Warp("", src_path, options=warp_options)
    if ds_warped is None:
        msg = f"Warp failed for {src_path}"
        logger.error(msg)
        raise RuntimeError(msg)

    arr = ds_warped.GetRasterBand(1).ReadAsArray().astype(np.float32)
    gt  = ds_warped.GetGeoTransform()
    proj = ds_warped.GetProjection()
    ds_warped = None
    return arr, gt, proj


def load_scl_mask(scl_dir, target_crs, x_min, y_min, x_max, y_max,
                  src_crs, target_res, mask_classes, logger):
    """
    Load SCL band and create cloud/shadow mask.
    Returns boolean array: True = valid pixel, False = masked.
    """
    logger.info("Loading SCL cloud mask...")
    scl_path = find_band_file(scl_dir, SCL_PATTERN, logger)
    logger.info(f"  SCL file: {os.path.basename(scl_path)}")

    scl_arr, gt, proj = warp_band(
        scl_path, target_crs, x_min, y_min, x_max, y_max,
        src_crs, target_res, logger
    )

    # Valid pixels = not in mask classes
    valid = np.ones(scl_arr.shape, dtype=bool)
    for cls in mask_classes:
        valid &= (scl_arr != cls)

    n_total   = valid.size
    n_valid   = np.sum(valid)
    n_masked  = n_total - n_valid
    logger.info(f"  Valid pixels  : {n_valid:,} ({100*n_valid/n_total:.1f}%)")
    logger.info(f"  Masked pixels : {n_masked:,} "
                f"({100*n_masked/n_total:.1f}%) clouds/shadows")
    return valid, gt, proj


def stack_bands(bands_config, scl_valid, target_crs,
                x_min, y_min, x_max, y_max,
                src_crs, target_res, logger):
    """
    Load all bands, warp to target CRS/extent/resolution,
    apply SCL mask, stack into 3D array.
    """
    logger.info(f"Stacking {len(bands_config)} bands...")
    arrays     = []
    band_names = []

    for band_dir, pattern, name in bands_config:
        logger.info(f"  Processing {name} ({pattern})...")
        path = find_band_file(band_dir, pattern, logger)
        arr, gt, proj = warp_band(
            path, target_crs, x_min, y_min, x_max, y_max,
            src_crs, target_res, logger
        )

        # Resize to match SCL if slightly different due to resampling
        if arr.shape != scl_valid.shape:
            logger.info(f"    Resizing {arr.shape} -> {scl_valid.shape}")
            from skimage.transform import resize
            arr = resize(
                arr, scl_valid.shape,
                order=1, preserve_range=True
            ).astype(np.float32)

        # Apply SCL mask
        arr[~scl_valid] = np.nan

        # Normalize from DN to reflectance (S2 L2A DN / 10000)
        arr = arr / 10000.0

        valid_pixels = arr[~np.isnan(arr)]
        logger.info(f"    Range: {valid_pixels.min():.4f} - "
                    f"{valid_pixels.max():.4f} reflectance")

        arrays.append(arr)
        band_names.append(name)

    stack = np.stack(arrays, axis=0)  # shape: (n_bands, rows, cols)
    logger.info(f"  Stack shape: {stack.shape}")
    return stack, gt, proj, band_names


def save_stack(stack, gt, proj, band_names, output_path, logger):
    """Save multiband stack as compressed GeoTIFF."""
    logger.info(f"Saving band stack...")
    n_bands, rows, cols = stack.shape

    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        output_path, cols, rows, n_bands, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj)

    for i, (arr, name) in enumerate(zip(stack, band_names)):
        band = ds_out.GetRasterBand(i + 1)
        band.WriteArray(arr)
        band.SetNoDataValue(np.nan)
        band.SetDescription(name)
        logger.info(f"  Band {i+1}: {name}")

    ds_out.FlushCache()
    ds_out = None
    logger.info(f"  Saved: {output_path}")


def compute_indices(stack, band_names, gt, proj, output_dir, logger):
    """
    Compute common vegetation indices from band stack and save
    as separate single-band GeoTIFFs.

    Indices:
        NDVI  = (NIR - Red) / (NIR + Red)
        NDRE  = (RedEdge1 - Red) / (RedEdge1 + Red)  -- sensitive to oak species
        NDWI  = (Green - NIR) / (Green + NIR)          -- water/moisture
        NBR   = (NIR - SWIR2) / (NIR + SWIR2)          -- burn ratio, eucalyptus
        EVI   = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    """
    logger.info("Computing vegetation indices...")

    def get_band(name):
        idx = band_names.index(name)
        return stack[idx].astype(np.float32)

    blue  = get_band("B02_blue")
    green = get_band("B03_green")
    red   = get_band("B04_red")
    nir   = get_band("B08_nir")
    re1   = get_band("B05_re1")
    nir2  = get_band("B8A_nir2")
    swir2 = get_band("B12_swir2")

    rows, cols = red.shape

    def safe_divide(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(b != 0, a / b, np.nan)
        return result.astype(np.float32)

    def save_index(arr, name):
        path = os.path.join(output_dir, f"s2_{name}.tif")
        driver = gdal.GetDriverByName("GTiff")
        ds_out = driver.Create(
            path, cols, rows, 1, gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "TILED=YES"]
        )
        ds_out.SetGeoTransform(gt)
        ds_out.SetProjection(proj)
        band = ds_out.GetRasterBand(1)
        band.WriteArray(arr)
        band.SetNoDataValue(np.nan)
        ds_out.FlushCache()
        ds_out = None
        valid = arr[~np.isnan(arr)]
        logger.info(f"  {name:6s}: range {valid.min():.3f} - "
                    f"{valid.max():.3f}, saved: {path}")
        return path

    indices = {}

    # NDVI
    ndvi = safe_divide(nir - red, nir + red)
    indices["NDVI"] = save_index(ndvi, "NDVI")

    # NDRE -> red edge index, sensitive to chlorophyll differences
    # between cork and holm oak
    ndre = safe_divide(re1 - red, re1 + red)
    indices["NDRE"] = save_index(ndre, "NDRE")

    # NDWI -> moisture index
    ndwi = safe_divide(green - nir, green + nir)
    indices["NDWI"] = save_index(ndwi, "NDWI")

    # NBR -> normalised burn ratio
    # eucalyptus has distinctive NBR signature vs oak
    nbr = safe_divide(nir2 - swir2, nir2 + swir2)
    indices["NBR"] = save_index(nbr, "NBR")

    # EVI -> enhanced vegetation index
    evi = np.where(
        ~np.isnan(nir) & ~np.isnan(red) & ~np.isnan(blue),
        2.5 * safe_divide(
            nir - red,
            nir + 6 * red - 7.5 * blue + 1
        ),
        np.nan
    ).astype(np.float32)
    indices["EVI"] = save_index(evi, "EVI")

    return indices


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logger = get_logger("04a_extract_sentinel2")

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Check SAFE folder exists
        if not os.path.exists(R10M_DIR):
            msg = f"R10m directory not found: {R10M_DIR}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if not os.path.exists(R20M_DIR):
            msg = f"R20m directory not found: {R20M_DIR}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.info(f"  SAFE R10m: {R10M_DIR}")
        logger.info(f"  SAFE R20m: {R20M_DIR}")

        # S2 source CRS
        src_crs = "EPSG:32629"
        logger.info(f"  Source CRS : {src_crs}")
        logger.info(f"  Target CRS : {TARGET_CRS}")
        logger.info(f"  Target res : {TARGET_RES} m")

        # 1 Get study area extent from nDSM
        x_min, y_min, x_max, y_max, ndsm_proj = get_study_extent(
            INPUT_NDSM, TARGET_CRS, logger
        )

        # 2 Load SCL mask
        scl_valid, gt, proj = load_scl_mask(
            R20M_DIR, TARGET_CRS,
            x_min, y_min, x_max, y_max,
            ndsm_proj, TARGET_RES,
            SCL_MASK_CLASSES, logger
        )

        # 3 Stack all bands
        stack, gt, proj, band_names = stack_bands(
            BANDS, scl_valid,
            TARGET_CRS, x_min, y_min, x_max, y_max,
            ndsm_proj, TARGET_RES, logger
        )

        # 4 Save multiband stack
        save_stack(stack, gt, proj, band_names, OUTPUT_STACK, logger)

        # 5 Compute vegetation indices
        indices = compute_indices(
            stack, band_names, gt, proj, OUTPUT_DIR, logger
        )

        logger.info(f"  Band stack     : {OUTPUT_STACK}")
        logger.info(f"  Bands          : {band_names}")
        logger.info(f"  Indices saved  : {list(indices.keys())}")
        logger.info(f"  All outputs in : {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise