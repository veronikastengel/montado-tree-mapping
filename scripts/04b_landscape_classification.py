"""
05d_landscape_classification.py
================================
Supervised Random Forest classification of the Sentinel-2 scene
into landscape/vegetation structural units using digitised training
polygons from COSc land inventory.

Pipeline:
    1. Load Sentinel-2 band stack + vegetation indices
    2. Load training polygons from GeoPackage
    3. Extract pixel values within each training polygon
    4. Train Random Forest classifier
    5. Classify full scene
    6. Save classification raster + confidence raster
    7. Save model diagnostics (feature importance, confusion matrix)

Classes (from digitised COSc polygons):
    213  sub_olive       — olive groves
    311  sobreiro_azinheira — cork/holm oak montado
    312  eucalipto       — eucalyptus plantation
    313  outras_folhosas — other broadleaf
    321  pinheiro_bravo  — maritime pine
    410  matos           — shrubland/maquis

Usage:
    python scripts/05d_landscape_classification.py
    python scripts/05d_landscape_classification.py --n-estimators 200 --max-depth 20

Arguments:
    --n-estimators      number of RF trees (default: 200)
    --max-depth         max tree depth, None = unlimited (default: None)
    --min-samples-leaf  min samples per leaf (default: 5)
    --test-size         fraction of samples held out for validation (default: 0.25)
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
INPUT_TRAINING    = os.path.join(BASE_DIR, "data", "raw",
                                 "area_information.gpkg")
TRAINING_LAYER    = "structural_area_locations_3763"
TRAINING_FIELD    = "type"
OUTPUT_DIR        = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_CLASSIF    = os.path.join(OUTPUT_DIR, "landscape_classification.tif")
OUTPUT_CONFIDENCE = os.path.join(OUTPUT_DIR, "landscape_confidence.tif")
OUTPUT_DIAG_DIR   = os.path.join(BASE_DIR, "outputs", "classification")

# Class label mapping — text to integer
# Add any additional classes you digitised here
CLASS_MAP = {
    "213":                  1,
    "sub_olive":            1,
    "311":                  2,
    "sobreiro_azinheira":   2,
    "sobreiro e azinheira": 2,
    "312":                  3,
    "eucalipto":            3,
    "313":                  4,
    "outras_folhosas":      4,
    "outras folhosas":      4,
    "321":                  5,
    "pinheiro_bravo":       5,
    "pinheiro bravo":       5,
    "410":                  6,
    "matos":                6,
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
        type=int,
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
    return parser.parse_args()


def load_raster_stack(stack_path, indices_paths, logger):
    """
    Load multiband Sentinel-2 stack and vegetation indices
    into a single array. Returns (n_features, rows, cols) array.
    """
    logger.info("Loading Sentinel-2 stack and indices...")

    ds    = gdal.Open(stack_path)
    if ds is None:
        msg = f"Cannot open stack: {stack_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    n_bands = ds.RasterCount
    rows    = ds.RasterYSize
    cols    = ds.RasterXSize
    gt      = ds.GetGeoTransform()
    proj    = ds.GetProjection()

    # Load all bands
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
            logger.info(f"  Skipping {idx_name} — file not found")
            continue
        ds_idx = gdal.Open(idx_path)
        if ds_idx is None:
            logger.info(f"  Skipping {idx_name} — cannot open")
            continue
        arr = ds_idx.GetRasterBand(1).ReadAsArray().astype(np.float32)
        # Resize if needed
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
    logger.info(f"  Full stack shape: {stack.shape} "
                f"({stack.shape[0]} features, {rows}x{cols} pixels)")
    return stack, gt, proj, band_names, rows, cols


def extract_training_samples(stack, gt, proj, training_path,
                              layer_name, field_name, class_map,
                              class_names, logger):
    """
    Extract pixel values from stack at training polygon locations.
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

    n_features = stack.shape[0]
    rows, cols = stack.shape[1], stack.shape[2]
    x_origin   = gt[0]
    y_origin   = gt[3]
    px_w       = gt[1]
    px_h       = gt[5]  # negative

    # Rasterize each training polygon to get pixel indices
    X_samples  = []
    y_samples  = []
    class_counts = {}
    unmatched  = []

    layer.ResetReading()
    for feature in layer:
        raw_label = feature.GetField(field_name)
        if raw_label is None:
            continue

        # Normalise label — strip whitespace, lowercase for matching
        raw_clean = str(raw_label).strip()
        label_int = None

        # Try exact match first
        if raw_clean in class_map:
            label_int = class_map[raw_clean]
        else:
            # Try case-insensitive match
            for key, val in class_map.items():
                if key.lower() == raw_clean.lower():
                    label_int = val
                    break

        if label_int is None:
            if raw_clean not in unmatched:
                unmatched.append(raw_clean)
                logger.info(f"  Unmatched label: '{raw_clean}' — skipping")
            continue

        # Rasterize polygon to pixel coordinates
        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        env    = geom.GetEnvelope()
        col_min = max(0, int((env[0] - x_origin) / px_w))
        col_max = min(cols - 1, int((env[1] - x_origin) / px_w) + 1)
        row_min = max(0, int((env[3] - y_origin) / px_h))
        row_max = min(rows - 1, int((env[2] - y_origin) / px_h) + 1)

        # Check each pixel in bounding box
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                # Pixel centre coordinates
                px_x = x_origin + (col + 0.5) * px_w
                px_y = y_origin + (row + 0.5) * px_h

                # Point in polygon test
                pt = ogr.Geometry(ogr.wkbPoint)
                pt.AddPoint(px_x, px_y)
                if not geom.Contains(pt):
                    continue

                # Extract feature vector
                pixel_vals = stack[:, row, col]

                # Skip if any NaN (cloud masked)
                if np.any(np.isnan(pixel_vals)):
                    continue

                X_samples.append(pixel_vals)
                y_samples.append(label_int)
                class_counts[label_int] = (
                    class_counts.get(label_int, 0) + 1
                )

    ds_train = None

    if len(X_samples) == 0:
        msg = "No training samples extracted — check CRS and polygon locations"
        logger.error(msg)
        raise ValueError(msg)

    X = np.array(X_samples, dtype=np.float32)
    y = np.array(y_samples, dtype=np.int32)

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
    Train Random Forest classifier with train/test split.
    Reports accuracy metrics and feature importance.
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

    # Train/test split — stratified to maintain class balance
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
        n_jobs=-1,  # use all CPU cores
        random_state=42,
        class_weight="balanced"  # handle class imbalance
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred    = rf.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    logger.info(f"  Overall accuracy: {accuracy*100:.1f}%")

    # Per-class report
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
    importances = rf.feature_importances_
    importance_pairs = sorted(
        zip(band_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    logger.info(f"  Feature importances (top 10):")
    for name, imp in importance_pairs[:10]:
        logger.info(f"    {name:20s}: {imp:.4f}")

    # Save confusion matrix
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(
        output_dir, f"confusion_matrix_{param_str}.csv"
    )
    cm = confusion_matrix(y_test, y_pred)
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + target_names)
        for i, row in enumerate(cm):
            writer.writerow([target_names[i]] + list(row))
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
    Apply trained RF to full scene.
    Saves classification raster and confidence raster.
    """
    logger.info("Classifying full scene...")

    n_features = stack.shape[0]

    # Reshape stack to (n_pixels, n_features)
    pixels = stack.reshape(n_features, -1).T  # (rows*cols, n_features)

    # Identify valid pixels (no NaN)
    valid_mask = ~np.any(np.isnan(pixels), axis=1)
    n_valid    = np.sum(valid_mask)
    n_total    = len(valid_mask)
    logger.info(f"  Valid pixels: {n_valid:,} of {n_total:,} "
                f"({100*n_valid/n_total:.1f}%)")

    # Classify in batches to avoid memory issues
    batch_size  = 500000
    labels_flat = np.zeros(n_total, dtype=np.int16)
    conf_flat   = np.zeros(n_total, dtype=np.float32)
    nodata_val  = -1

    valid_indices = np.where(valid_mask)[0]
    n_batches     = int(np.ceil(len(valid_indices) / batch_size))

    logger.info(f"  Processing {n_batches} batches of {batch_size:,}...")

    for i in range(n_batches):
        start  = i * batch_size
        end    = min(start + batch_size, len(valid_indices))
        idx    = valid_indices[start:end]

        batch_X    = pixels[idx]
        batch_pred = rf.predict(batch_X)
        batch_prob = rf.predict_proba(batch_X)
        batch_conf = batch_prob.max(axis=1)

        labels_flat[idx] = batch_pred.astype(np.int16)
        conf_flat[idx]   = batch_conf.astype(np.float32)

        if (i + 1) % 5 == 0 or i == n_batches - 1:
            logger.info(f"  Batch {i+1}/{n_batches} done "
                        f"({100*(i+1)/n_batches:.0f}%)")

    # Set invalid pixels to nodata
    labels_flat[~valid_mask] = nodata_val
    conf_flat[~valid_mask]   = np.nan

    # Reshape to 2D
    labels_2d = labels_flat.reshape(rows, cols)
    conf_2d   = conf_flat.reshape(rows, cols)

    # Report class distribution
    logger.info(f"  Classification results:")
    unique, counts = np.unique(
        labels_2d[labels_2d != nodata_val],
        return_counts=True
    )
    for class_id, count in zip(unique, counts):
        name = class_names.get(int(class_id), f"class_{class_id}")
        pct  = 100 * count / np.sum(counts)
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
    band = ds_out.GetRasterBand(1)
    band.WriteArray(labels_2d)
    band.SetNoDataValue(nodata_val)
    ds_out.FlushCache()
    ds_out = None
    logger.info(f"  Classification saved: {output_classif}")

    # Save confidence raster
    ds_conf = driver.Create(
        output_confidence, cols, rows, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    ds_conf.SetGeoTransform(gt)
    ds_conf.SetProjection(proj)
    band_conf = ds_conf.GetRasterBand(1)
    band_conf.WriteArray(conf_2d)
    band_conf.SetNoDataValue(np.nan)
    ds_conf.FlushCache()
    ds_conf = None
    logger.info(f"  Confidence saved: {output_confidence}")

    return labels_2d, conf_2d


def save_colormap(class_names, output_dir, param_str, logger):
    """
    Save a QGIS-compatible colour map file for the classification raster.
    Load in QGIS via Layer Properties -> Symbology -> Load Color Map.
    """
    colors = {
        1: (255, 200, 0,   "Olive grove"),
        2: (34,  139, 34,  "Cork/Holm oak montado"),
        3: (0,   100, 200, "Eucalyptus"),
        4: (100, 180, 100, "Other broadleaf"),
        5: (180, 120, 60,  "Maritime pine"),
        6: (200, 180, 130, "Shrubland"),
    }
    cmap_path = os.path.join(
        output_dir, f"landscape_colormap_{param_str}.txt"
    )
    with open(cmap_path, "w") as f:
        f.write("# QGIS colour map\n")
        f.write("INTERPOLATION:EXACT\n")
        for class_id, (r, g, b, name) in colors.items():
            f.write(f"{class_id},{r},{g},{b},255,{name}\n")
        f.write("-1,0,0,0,0,NoData\n")

    logger.info(f"  QGIS colour map saved: {cmap_path}")
    logger.info(f"  Load in QGIS: Layer Properties -> "
                f"Symbology -> Load Color Map")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("05d_landscape_classification")

    try:
        logger.info("=" * 60)
        logger.info("STEP 05d: Landscape classification (Random Forest)")
        logger.info("=" * 60)
        logger.info(f"Parameters:")
        logger.info(f"  n-estimators     = {args.n_estimators}")
        logger.info(f"  max-depth        = {args.max_depth}")
        logger.info(f"  min-samples-leaf = {args.min_samples_leaf}")
        logger.info(f"  test-size        = {args.test_size}")

        param_str = (f"ne{args.n_estimators}"
                     f"_md{args.max_depth}"
                     f"_msl{args.min_samples_leaf}")

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

        # 2. Extract training samples
        X, y = extract_training_samples(
            stack, gt, proj,
            INPUT_TRAINING, TRAINING_LAYER, TRAINING_FIELD,
            CLASS_MAP, CLASS_NAMES, logger
        )

        # 3. Train Random Forest
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

        # 4. Classify full scene
        labels_2d, conf_2d = classify_scene(
            rf, stack, gt, proj, rows, cols,
            CLASS_NAMES,
            OUTPUT_CLASSIF,
            OUTPUT_CONFIDENCE,
            logger
        )

        # 5. Save QGIS colour map
        save_colormap(CLASS_NAMES, OUTPUT_DIAG_DIR, param_str, logger)

        logger.info("=" * 60)
        logger.info("Done!")
        logger.info(f"  Classification : {OUTPUT_CLASSIF}")
        logger.info(f"  Confidence     : {OUTPUT_CONFIDENCE}")
        logger.info(f"  Diagnostics    : {OUTPUT_DIAG_DIR}")
        logger.info(f"  Accuracy       : {accuracy*100:.1f}%")
        logger.info("=" * 60)
        logger.info("Next step: run 06_crown_classification.py")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise