"""
04b_landscape_classification.py
================================
Supervised Random Forest classification of the Sentinel-2 scene
into landscape/vegetation structural units using digitised training
polygons from COSc land inventory.

Pipeline:
    1. Load Sentinel-2 band stack + vegetation indices
    2. Apply vegetation mask (excludes village, buildings, bare ground)
    3. Load training polygons from GeoPackage
    4. Extract pixel values within each training polygon (fast rasterization)
    5. Train Random Forest classifier
    6. Classify full scene (masked pixels stay NoData)
    7. Save classification raster + confidence raster
    8. Save model diagnostics (feature importance, confusion matrix)

Classes (from digitised COSc polygons):
    213  sub_olive          -> olive groves
    311  sobreiro_azinheira -> cork/holm oak montado
    312  eucalipto          -> eucalyptus plantation
    313  outras_folhosas    -> other broadleaf
    321  pinheiro_bravo     -> maritime pine
    410  matos              -> shrubland/maquis

Usage:
    python scripts/04b_landscape_classification.py
    python scripts/04b_landscape_classification.py --n-estimators 200 --max-depth 20

Arguments:
    --n-estimators      number of RF trees (default: 200)
    --max-depth         max tree depth, None = unlimited (default: None)
    --min-samples-leaf  min samples per leaf (default: 5)
    --test-size         fraction of samples held out for validation (default: 0.25)
    --mask-threshold    Minimum fraction of vegetation pixels within a Sentinel-2
                        pixel to include it (default: 0.3 = 30%)
"""

import os
import sys
import argparse
import numpy as np
import csv
from osgeo import gdal, ogr, osr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_STACK       = os.path.join(BASE_DIR, "data", "processed",
                                 "sentinel2_stack.tif")
INPUT_INDICES     = {
    "NDVI": os.path.join(BASE_DIR, "data", "processed", "s2_NDVI.tif"),
    "NDRE": os.path.join(BASE_DIR, "data", "processed", "s2_NDRE.tif"),
    "NDWI": os.path.join(BASE_DIR, "data", "processed", "s2_NDWI.tif"),
    "NBR":  os.path.join(BASE_DIR, "data", "processed", "s2_NBR.tif"),
    "EVI":  os.path.join(BASE_DIR, "data", "processed", "s2_EVI.tif"),
}
INPUT_VEG_MASK    = os.path.join(BASE_DIR, "data", "processed",
                                 "vegetation_mask.tif")
INPUT_TRAINING    = os.path.join(BASE_DIR, "data", "raw",
                                 "area_information.gpkg")
TRAINING_LAYER    = "structural_area_locations_3763"
TRAINING_FIELD    = "type"
OUTPUT_DIR        = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIAG_DIR   = os.path.join(BASE_DIR, "outputs", "classification")

CONF_NODATA       = -9999.0  # nodata value for confidence raster

# Class label mapping -> text to integer
# Add any additional classes you digitised here
CLASS_MAP = {
    "213": 1,
    "311": 2,
    "312": 3,
    "313": 4,
    "321": 5,
    "410": 6,
}

CLASS_NAMES = {
    1: "Olive grove",
    2: "Cork/Holm oak montado",
    3: "Eucalyptus",
    4: "Other broadleaf",
    5: "Maritime pine",
    6: "Shrubland",
}
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random Forest landscape classification from Sentinel-2."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of Random Forest trees (default: 200)"
    )
    parser.add_argument(
        "--max-depth",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=None,
        help="Max tree depth, None=unlimited (default: None)"
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=5,
        help="Min samples per leaf (default: 5)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of samples for validation (default: 0.25)"
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.3,
        help="Minimum fraction of vegetation pixels within a Sentinel-2 "
            "pixel to include it (default: 0.3 = 30%%)"
    )
    return parser.parse_args()


def load_raster_stack(stack_path, indices_paths, logger):
    """
    Load multiband Sentinel-2 stack and vegetation indices
    into a single array. Returns (n_features, rows, cols) array.
    """
    logger.info("Loading Sentinel-2 stack and indices...")

    ds = gdal.Open(stack_path)
    if ds is None:
        msg = f"Cannot open stack: {stack_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    n_bands = ds.RasterCount
    rows    = ds.RasterYSize
    cols    = ds.RasterXSize
    gt      = ds.GetGeoTransform()
    proj    = ds.GetProjection()

    arrays     = []
    band_names = []
    for i in range(1, n_bands + 1):
        band = ds.GetRasterBand(i)
        arr  = band.ReadAsArray().astype(np.float32)
        name = band.GetDescription() or f"band_{i}"
        arrays.append(arr)
        band_names.append(name)
        logger.info(f"  Loaded band {i}: {name}")
    ds = None

    # Load vegetation indices
    for idx_name, idx_path in indices_paths.items():
        if not os.path.exists(idx_path):
            logger.info(f"  Skipping {idx_name} -> file not found")
            continue
        ds_idx = gdal.Open(idx_path)
        if ds_idx is None:
            logger.info(f"  Skipping {idx_name} -> cannot open")
            continue
        arr = ds_idx.GetRasterBand(1).ReadAsArray().astype(np.float32)
        if arr.shape != (rows, cols):
            logger.info(f"  Resizing {idx_name} {arr.shape} -> {(rows, cols)}")
            from skimage.transform import resize
            arr = resize(
                arr, (rows, cols),
                order=1, preserve_range=True
            ).astype(np.float32)
        arrays.append(arr)
        band_names.append(idx_name)
        ds_idx = None
        logger.info(f"  Loaded index: {idx_name}")

    stack = np.stack(arrays, axis=0)
    logger.info(f"  Full stack shape  : {stack.shape} "
                f"({stack.shape[0]} features, {rows}x{cols} pixels)")
    logger.info(f"  Total features for RF: {len(band_names)}")
    return stack, gt, proj, band_names, rows, cols


def apply_vegetation_mask(stack, gt, rows, cols, mask_path,
                          mask_threshold, logger):
    """
    Apply vegetation mask to Sentinel-2 stack using fractional
    coverage thresholding.

    Because the vegetation mask (50cm) is much finer than Sentinel-2
    (10m), each S2 pixel covers ~400 mask pixels. Rather than masking
    an entire S2 pixel because part of it is non-vegetation, we
    include the pixel if at least mask_threshold fraction of the
    corresponding mask area is valid vegetation.

    This is standard practice when downscaling fine masks to coarser
    imagery — it preserves S2 pixels that contain real vegetation
    signal even if they also contain some bare ground or road.
    """
    logger.info(f"Applying vegetation mask "
                f"(threshold={mask_threshold*100:.0f}% vegetation)...")

    if not os.path.exists(mask_path):
        msg = f"Vegetation mask not found: {mask_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Warp mask to stack grid using AVERAGE resampling
    # This gives fraction of valid pixels per S2 pixel (0.0 to 1.0)
    # because the mask is binary (0/1) and average = mean = fraction
    x_min = gt[0]
    y_max = gt[3]
    x_max = gt[0] + cols * gt[1]
    y_min = gt[3] + rows * gt[5]

    # warp mask to float MEM raster at stack resolution
    # using average resampling, with nodata explicitly excluded
    warp_options = gdal.WarpOptions(
        width=cols,
        height=rows,
        outputBounds=(x_min, y_min, x_max, y_max),
        resampleAlg="average",
        srcNodata=255,           # exclude nodata from average
        dstNodata=-1.0,          # output nodata for pixels fully outside
        outputType=gdal.GDT_Float32,
        format="MEM"
    )
    ds_warped = gdal.Warp("", mask_path, options=warp_options)
    if ds_warped is None:
        msg = "Failed to warp vegetation mask to stack grid"
        logger.error(msg)
        raise RuntimeError(msg)

    frac_arr  = ds_warped.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds_warped = None

    # Pixels with dstNodata (-1) are fully outside study area -> mask them
    frac_arr[frac_arr < 0] = 0.0

    # Pixel is valid if vegetation fraction >= threshold
    valid   = frac_arr >= mask_threshold
    invalid = ~valid

    n_total   = rows * cols
    n_valid   = int(np.sum(valid))
    n_invalid = int(np.sum(invalid))
    logger.info(f"  S2 pixels included : {n_valid:,} "
                f"({100*n_valid/n_total:.1f}%) "
                f"[>= {mask_threshold*100:.0f}% vegetation]")
    logger.info(f"  S2 pixels excluded : {n_invalid:,} "
                f"({100*n_invalid/n_total:.1f}%)")

    # Report fraction distribution for tuning
    logger.info(f"  Vegetation fraction stats:")
    logger.info(f"    Mean   : {frac_arr.mean():.2f}")
    logger.info(f"    Median : {np.median(frac_arr):.2f}")
    logger.info(f"    P25    : {np.percentile(frac_arr, 25):.2f}")
    logger.info(f"    P75    : {np.percentile(frac_arr, 75):.2f}")

    for i in range(stack.shape[0]):
        stack[i][invalid] = np.nan

    return stack


def rasterize_polygon_to_mask(geom, gt, rows, cols):
    """
    Rasterize a single OGR polygon geometry into a boolean numpy mask
    at the same grid as the stack. Returns True where inside polygon.

    Much faster than the pixel-by-pixel point-in-polygon approach
    because GDAL's RasterizeLayer runs in C.
    """
    mem_driver = gdal.GetDriverByName("MEM")
    ds_mask    = mem_driver.Create("", cols, rows, 1, gdal.GDT_Byte)
    ds_mask.SetGeoTransform(gt)

    # Create in-memory vector layer with just this polygon
    srs       = osr.SpatialReference()
    mem_vec   = ogr.GetDriverByName("MEM").CreateDataSource("")
    mem_layer = mem_vec.CreateLayer("poly", srs=srs,
                                    geom_type=ogr.wkbPolygon)
    feat      = ogr.Feature(mem_layer.GetLayerDefn())
    feat.SetGeometry(geom.Clone())
    mem_layer.CreateFeature(feat)

    # Burn polygon into raster
    gdal.RasterizeLayer(ds_mask, [1], mem_layer, burn_values=[1])
    mask = ds_mask.GetRasterBand(1).ReadAsArray().astype(bool)

    ds_mask = None
    mem_vec = None
    return mask


def extract_training_samples(stack, gt, proj, training_path,
                              layer_name, field_name, class_map,
                              class_names, logger):
    """
    Extract pixel values from stack at training polygon locations.
    Uses fast GDAL rasterization instead of pixel-by-pixel geometry tests.
    Returns X (n_samples, n_features) and y (n_samples,) arrays.
    """
    logger.info("Extracting training samples from polygons...")

    ds_train = ogr.Open(training_path, 0)
    if ds_train is None:
        msg = f"Cannot open training data: {training_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    layer = ds_train.GetLayerByName(layer_name)
    if layer is None:
        available = [ds_train.GetLayerByIndex(i).GetName()
                     for i in range(ds_train.GetLayerCount())]
        msg = (f"Layer '{layer_name}' not found. "
               f"Available: {available}")
        logger.error(msg)
        raise ValueError(msg)

    rows, cols = stack.shape[1], stack.shape[2]

    X_samples    = []
    y_samples    = []
    class_counts = {}
    unmatched    = []

    layer.ResetReading()
    for feature in layer:
        raw_label = feature.GetField(field_name)
        if raw_label is None:
            continue

        # Normalise label — strip whitespace for matching
        raw_clean = str(raw_label).strip()
        label_int = None

        # Exact match first
        if raw_clean in class_map:
            label_int = class_map[raw_clean]
        else:
            # Case-insensitive fallback
            for key, val in class_map.items():
                if key.lower() == raw_clean.lower():
                    label_int = val
                    break

        if label_int is None:
            if raw_clean not in unmatched:
                unmatched.append(raw_clean)
                logger.info(f"  Unmatched label: '{raw_clean}' — skipping")
            continue

        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        # Fast rasterization — replaces slow pixel-by-pixel loop
        poly_mask = rasterize_polygon_to_mask(geom, gt, rows, cols)

        # Get pixel indices inside polygon
        pixel_rows, pixel_cols = np.where(poly_mask)
        if len(pixel_rows) == 0:
            logger.info(f"  Warning: polygon for '{raw_clean}' has no "
                        f"pixels in raster extent — check CRS alignment")
            continue

        for r, c in zip(pixel_rows, pixel_cols):
            pixel_vals = stack[:, r, c]
            # Skip NaN pixels (cloud masked or vegetation masked)
            if np.any(np.isnan(pixel_vals)):
                continue
            X_samples.append(pixel_vals)
            y_samples.append(label_int)
            class_counts[label_int] = class_counts.get(label_int, 0) + 1

    ds_train = None

    if len(X_samples) == 0:
        msg = ("No training samples extracted — "
               "check CRS alignment and polygon locations. "
               "Also check that training polygons overlap valid "
               "(non-masked) pixels.")
        logger.error(msg)
        raise ValueError(msg)

    X = np.array(X_samples, dtype=np.float32)
    y = np.array(y_samples,  dtype=np.int32)

    logger.info(f"  Total training pixels: {len(y):,}")
    logger.info(f"  Samples per class:")
    for class_id, count in sorted(class_counts.items()):
        name = class_names.get(class_id, f"class_{class_id}")
        logger.info(f"    {class_id} {name:30s}: {count:,}")

    if unmatched:
        logger.info(f"  Unmatched labels (skipped): {unmatched}")

    return X, y


def train_random_forest(X, y, n_estimators, max_depth,
                        min_samples_leaf, test_size,
                        class_names, band_names,
                        output_dir, param_str, logger):
    """
    Train Random Forest classifier with stratified train/test split.
    Reports accuracy, per-class metrics, and feature importance.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report,
                                 confusion_matrix,
                                 accuracy_score)

    logger.info(f"Training Random Forest "
                f"(n_estimators={n_estimators}, "
                f"max_depth={max_depth}, "
                f"min_samples_leaf={min_samples_leaf})...")

    # Stratified split maintains class proportions in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    logger.info(f"  Train samples: {len(X_train):,}")
    logger.info(f"  Test samples : {len(X_test):,}")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,        # use all CPU cores
        random_state=42,
        class_weight="balanced"  # handle class imbalance
    )
    rf.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred   = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"  Overall accuracy: {accuracy*100:.1f}%")

    target_names = [
        class_names.get(c, f"class_{c}")
        for c in sorted(set(y))
    ]
    report = classification_report(
        y_test, y_pred,
        target_names=target_names
    )
    logger.info(f"  Classification report:\n{report}")

    # Feature importance
    importances      = rf.feature_importances_
    importance_pairs = sorted(
        zip(band_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    logger.info(f"  Feature importances (top 10):")
    for name, imp in importance_pairs[:10]:
        logger.info(f"    {name:20s}: {imp:.4f}")

    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix
    cm_path = os.path.join(
        output_dir, f"confusion_matrix_{param_str}.csv"
    )
    cm = confusion_matrix(y_test, y_pred)
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + target_names)
        for i, row_cm in enumerate(cm):
            writer.writerow([target_names[i]] + list(row_cm))
    logger.info(f"  Confusion matrix saved: {cm_path}")

    # Save feature importances
    imp_path = os.path.join(
        output_dir, f"feature_importance_{param_str}.csv"
    )
    with open(imp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for name, imp in importance_pairs:
            writer.writerow([name, round(float(imp), 6)])
    logger.info(f"  Feature importance saved: {imp_path}")

    # Save classification report
    report_path = os.path.join(
        output_dir, f"classification_report_{param_str}.txt"
    )
    with open(report_path, "w") as f:
        f.write(f"Overall accuracy: {accuracy*100:.1f}%\n\n")
        f.write(report)
    logger.info(f"  Classification report saved: {report_path}")

    return rf, accuracy


def classify_scene(rf, stack, gt, proj, rows, cols,
                   class_names, output_classif,
                   output_confidence, logger):
    """
    Apply trained RF classifier to full scene pixel by pixel.
    Pixels that are NaN (masked by vegetation mask or clouds) are
    set to NoData in both output rasters — they are never classified.
    Saves classification raster (Int16) and confidence raster (Float32).
    """
    logger.info("Classifying full scene...")

    n_features = stack.shape[0]

    # Reshape to (n_pixels, n_features)
    pixels = stack.reshape(n_features, -1).T  # (rows*cols, n_features)

    # Valid pixels = no NaN in any band
    # NaN pixels include both cloud-masked and vegetation-masked pixels
    valid_mask = ~np.any(np.isnan(pixels), axis=1)
    n_valid    = int(np.sum(valid_mask))
    n_total    = len(valid_mask)
    logger.info(f"  Valid pixels to classify : {n_valid:,} of {n_total:,} "
                f"({100*n_valid/n_total:.1f}%)")
    logger.info(f"  Masked/excluded pixels   : {n_total - n_valid:,} "
                f"({100*(n_total - n_valid)/n_total:.1f}%)")

    # Initialise output arrays with nodata values
    classif_nodata = -1
    labels_flat    = np.full(n_total, classif_nodata, dtype=np.int16)
    conf_flat      = np.full(n_total, CONF_NODATA,    dtype=np.float32)

    valid_indices = np.where(valid_mask)[0]
    batch_size    = 500000
    n_batches     = int(np.ceil(len(valid_indices) / batch_size))
    logger.info(f"  Processing {n_batches} batches of up to "
                f"{batch_size:,} pixels...")

    for i in range(n_batches):
        start      = i * batch_size
        end        = min(start + batch_size, len(valid_indices))
        idx        = valid_indices[start:end]
        batch_X    = pixels[idx]
        batch_pred = rf.predict(batch_X)
        batch_prob = rf.predict_proba(batch_X)
        batch_conf = batch_prob.max(axis=1)

        labels_flat[idx] = batch_pred.astype(np.int16)
        conf_flat[idx]   = batch_conf.astype(np.float32)

        if (i + 1) % 5 == 0 or i == n_batches - 1:
            logger.info(f"  Batch {i+1}/{n_batches} done "
                        f"({100*(i+1)/n_batches:.0f}%)")

    # Reshape to 2D
    labels_2d = labels_flat.reshape(rows, cols)
    conf_2d   = conf_flat.reshape(rows, cols)

    # Report class distribution (exclude nodata)
    logger.info(f"  Classification results (valid pixels only):")
    unique, counts = np.unique(
        labels_2d[labels_2d != classif_nodata],
        return_counts=True
    )
    total_classified = int(np.sum(counts))
    for class_id, count in zip(unique, counts):
        name = class_names.get(int(class_id), f"class_{class_id}")
        pct  = 100 * count / total_classified
        logger.info(f"    {class_id} {name:30s}: "
                    f"{count:,} px ({pct:.1f}%)")

    # Save classification raster
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(
        output_classif, cols, rows, 1, gdal.GDT_Int16,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj)
    band_out = ds_out.GetRasterBand(1)
    band_out.WriteArray(labels_2d)
    band_out.SetNoDataValue(classif_nodata)
    ds_out.FlushCache()
    ds_out = None
    logger.info(f"  Classification saved : {output_classif}")

    # Save confidence raster
    # Use CONF_NODATA (-9999.0) as nodata — avoids NaN compatibility issues
    ds_conf = driver.Create(
        output_confidence, cols, rows, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    ds_conf.SetGeoTransform(gt)
    ds_conf.SetProjection(proj)
    band_conf = ds_conf.GetRasterBand(1)
    band_conf.WriteArray(conf_2d)
    band_conf.SetNoDataValue(CONF_NODATA)
    ds_conf.FlushCache()
    ds_conf = None
    logger.info(f"  Confidence saved     : {output_confidence}")
    logger.info(f"  Confidence NoData    : {CONF_NODATA} "
                f"(masked/unclassified pixels)")

    return labels_2d, conf_2d


def save_colormap(class_names, output_classif, output_dir,
                  param_str, logger):
    """
    Save a QGIS .qml file for the classification raster.
    """
    colors = {
        1: ("255,200,0",   "Olive grove"),
        2: ("34,139,34",   "Cork/Holm oak montado"),
        3: ("0,100,200",   "Eucalyptus"),
        4: ("100,180,100", "Other broadleaf"),
        5: ("180,120,60",  "Maritime pine"),
        6: ("200,180,130", "Shrubland"),
    }

    # Build palette entries
    palette_items = ""
    for class_id, (rgb, name) in colors.items():
        r, g, b = rgb.split(",")
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
        palette_items += (
            f'          <paletteEntry value="{class_id}" '
            f'color="{hex_color}" '
            f'label="{name}" '
            f'alpha="255"/>\n'
        )
    palette_items += (
        '          <paletteEntry value="-1" '
        'color="#000000" '
        'label="NoData" '
        'alpha="0"/>\n'
    )

    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
                <qgis version="3.28" styleCategories="Symbology">
                <pipe>
                    <provider>
                    <resampling zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour" enabled="false" maxOversampling="2"/>
                    </provider>
                    <rasterrenderer type="paletted" opacity="1" alphaBand="-1" band="1" nodataColor="">
                    <rasterTransparency/>
                    <minMaxOrigin>
                        <limits>None</limits>
                        <extent>WholeRaster</extent>
                        <statAccuracy>Estimated</statAccuracy>
                        <cumulativeCutLower>0.02</cumulativeCutLower>
                        <cumulativeCutUpper>0.98</cumulativeCutUpper>
                        <stdDevFactor>2</stdDevFactor>
                    </minMaxOrigin>
                    <colorPalette>
                {palette_items.rstrip()}
                    </colorPalette>
                    <colorramp type="randomcolors" name="[source]">
                        <Option/>
                    </colorramp>
                    </rasterrenderer>
                    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
                    <huesaturation colorizeOn="0" colorizeRed="255" colorizeBlue="128" colorizeGreen="128" grayscaleMode="0" colorizeStrength="100" invertColors="0" saturation="0"/>
                    <rasterresampler maxOversampling="2"/>
                    <resamplingStage>resamplingFilter</resamplingStage>
                </pipe>
                </qgis>
                """

    qml_path = output_classif.replace(".tif", ".qml")
    with open(qml_path, "w") as f:
        f.write(qml)

    logger.info(f"  QGIS style saved : {qml_path}")
    logger.info(f"  Auto-loads when you open the .tif in QGIS")

    # Also save a plain CSV legend for reference
    legend_path = os.path.join(
        output_dir, f"landscape_legend_{param_str}.csv"
    )
    with open(legend_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "color_rgb"])
        for class_id, (rgb, name) in colors.items():
            writer.writerow([class_id, name, rgb])
        writer.writerow([-1, "NoData", "0,0,0"])
    logger.info(f"  Legend CSV saved : {legend_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("04b_landscape_classification")

    try:
        logger.info(f"Parameters:")
        logger.info(f"  n-estimators     = {args.n_estimators}")
        logger.info(f"  max-depth        = {args.max_depth}")
        logger.info(f"  min-samples-leaf = {args.min_samples_leaf}")
        logger.info(f"  test-size        = {args.test_size}")

        param_str = (f"ne{args.n_estimators}"
                     f"_md{args.max_depth}"
                     f"_msl{args.min_samples_leaf}"
                     f"_mt{args.mask_threshold}")
        
        OUTPUT_CLASSIF    = os.path.join(OUTPUT_DIR,
                                 f"landscape_classification_{param_str}.tif")
        OUTPUT_CONFIDENCE = os.path.join(OUTPUT_DIR,
                                 f"landscape_confidence_{param_str}.tif")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIAG_DIR, exist_ok=True)

        # Check sklearn available
        try:
            import sklearn
            logger.info(f"  scikit-learn: {sklearn.__version__}")
        except ImportError:
            msg = "scikit-learn not found. pip install scikit-learn"
            logger.error(msg)
            raise ImportError(msg)

        # 1. Load raster stack
        stack, gt, proj, band_names, rows, cols = load_raster_stack(
            INPUT_STACK, INPUT_INDICES, logger
        )

        # 2. Apply vegetation mask -> sets non-vegetation pixels to NaN
        #    This single step handles both the vegetation extent and
        #    the urban/civilisation exclusion zones
        stack = apply_vegetation_mask(
            stack, gt, rows, cols, INPUT_VEG_MASK,
            args.mask_threshold, logger
        )

        # 3. Extract training samples from digitised polygons
        #    NaN pixels (masked) are automatically skipped during extraction
        X, y = extract_training_samples(
            stack, gt, proj,
            INPUT_TRAINING, TRAINING_LAYER, TRAINING_FIELD,
            CLASS_MAP, CLASS_NAMES, logger
        )

        # 4. Train Random Forest
        rf, accuracy = train_random_forest(
            X, y,
            args.n_estimators,
            args.max_depth,
            args.min_samples_leaf,
            args.test_size,
            CLASS_NAMES,
            band_names,
            OUTPUT_DIAG_DIR,
            param_str,
            logger
        )

        # 5. Classify full scene
        #    Masked pixels remain NoData in output rasters
        labels_2d, conf_2d = classify_scene(
            rf, stack, gt, proj, rows, cols,
            CLASS_NAMES,
            OUTPUT_CLASSIF,
            OUTPUT_CONFIDENCE,
            logger
        )

        # 6. Save QGIS colour map
        save_colormap(CLASS_NAMES, OUTPUT_CLASSIF, OUTPUT_DIAG_DIR,
              param_str, logger)

        logger.info(f"  Classification : {OUTPUT_CLASSIF}")
        logger.info(f"  Confidence     : {OUTPUT_CONFIDENCE}")
        logger.info(f"  Diagnostics    : {OUTPUT_DIAG_DIR}")
        logger.info(f"  Accuracy       : {accuracy*100:.1f}%")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise