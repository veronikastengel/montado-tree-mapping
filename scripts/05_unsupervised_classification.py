"""
05_classification.py
====================
Unsupervised classification of tree crowns using PCA + K-Means clustering.

Pipeline:
    1. Load crown polygons with extracted features
    2. Filter crowns (edge, minimum points)
    3. Standardize features
    4. PCA to understand feature importance and reduce dimensionality
    5. Elbow method to suggest optimal k
    6. K-Means clustering
    7. Write cluster labels back to GeoPackage
    8. Save PCA and cluster diagnostics as CSV and plots

Usage:
    python scripts/05_unsupervised_classification.py
    python scripts/05_unsupervised_classification.py --crowns-layer crowns_treetops_w5_mh2.0_t0.001_c0.01_mh2.0
    python scripts/05_unsupervised_classification.py --crowns-layer crowns_treetops_w5_mh2.0_t0.001_c0.01_mh2.0 --k 8
    python scripts/05_unsupervised_classification.py --crowns-layer crowns_treetops_w5_mh2.0_t0.001_c0.01_mh2.0 --k 8 --pca-components 5

Arguments:
    --crowns-layer      layer name inside crowns.gpkg (default: first layer)
    --k                 number of clusters for k-means (default: 8)
    --pca-components    number of PCA components to use for clustering (default: 5)
    --min-points        minimum n_points to include crown (default: 30)
    --elbow-max-k       maximum k to test in elbow method (default: 15)
"""

import os
import sys
import argparse
import numpy as np
import csv
from osgeo import ogr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_logger

# ============================================================
# CONFIG
# ============================================================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CROWNS = os.path.join(BASE_DIR, "data", "processed", "crowns.gpkg")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")

# Features to use for PCA and clustering
FEATURES = [
    "mean_height",
    "max_height",
    "height_width_ratio",
    "mean_intensity",
    "mean_ndvi",
    "rugosity",
    "vert_dist_top25",
    "vert_dist_mid50",
    "vert_dist_bot25",
    "point_density",
]
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unsupervised PCA + K-Means classification of tree crowns."
    )
    parser.add_argument(
        "--crowns-layer",
        type=str,
        default="crowns_treetops_w5_mh2.0_t0.001_c0.01_mh2.0",
        help="Layer name inside crowns.gpkg (default: crowns_treetops_w5_mh2.0_t0.001_c0.01_mh2.0)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of clusters for K-Means (default: 8)"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=5,
        help="Number of PCA components to use for clustering (default: 5)"
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=30,
        help="Minimum n_points to include crown in classification (default: 30)"
    )
    parser.add_argument(
        "--elbow-max-k",
        type=int,
        default=15,
        help="Maximum k to test in elbow method (default: 15)"
    )
    return parser.parse_args()


def load_crown_features(gpkg_path, layer_name, features,
                        min_points, logger):
    """
    Load crown features from GeoPackage.
    Filters out edge crowns and crowns with too few points.
    Returns feature matrix, FID list, and filter statistics.
    """
    logger.info(f"Loading crown features from: {os.path.basename(gpkg_path)}")
    ds = ogr.Open(gpkg_path, 0)
    if ds is None:
        msg = f"Cannot open GeoPackage: {gpkg_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    available = [ds.GetLayerByIndex(i).GetName()
                 for i in range(ds.GetLayerCount())]
    logger.info(f"  Available layers: {available}")

    if layer_name not in available:
        msg = f"Layer '{layer_name}' not found. Available: {available}"
        logger.error(msg)
        raise ValueError(msg)

    layer   = ds.GetLayerByName(layer_name)
    n_total = layer.GetFeatureCount()
    logger.info(f"  Total crowns: {n_total:,}")

    # Check all required fields exist
    layer_defn  = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName()
                   for i in range(layer_defn.GetFieldCount())]
    logger.info(f"  Available fields: {field_names}")

    missing = [f for f in features + ["n_points", "is_edge"]
               if f not in field_names]
    if missing:
        msg = (f"Missing fields: {missing}. "
               f"Run 04_feature_extraction.py first.")
        logger.error(msg)
        raise ValueError(msg)

    # Load features with filters
    fids          = []
    feature_rows  = []
    n_edge        = 0
    n_low_points  = 0
    n_null        = 0

    layer.ResetReading()
    for feature in layer:
        fid = feature.GetFID()

        # Filter edge crowns
        is_edge = feature.GetField("is_edge")
        if is_edge == 1:
            n_edge += 1
            continue

        # Filter low point count
        n_points = feature.GetField("n_points")
        if n_points is None or n_points < min_points:
            n_low_points += 1
            continue

        # Extract feature values
        row  = []
        null = False
        for f in features:
            val = feature.GetField(f)
            if val is None:
                null = True
                break
            row.append(float(val))

        if null:
            n_null += 1
            continue

        fids.append(fid)
        feature_rows.append(row)

    ds = None

    X   = np.array(feature_rows)
    logger.info(f"  Crowns after filtering:")
    logger.info(f"    Total          : {n_total:,}")
    logger.info(f"    Removed edge   : {n_edge:,}")
    logger.info(f"    Removed < {min_points} pts: {n_low_points:,}")
    logger.info(f"    Removed null   : {n_null:,}")
    logger.info(f"    Remaining      : {len(fids):,}")

    if len(fids) == 0:
        msg = "No crowns remaining after filtering"
        logger.error(msg)
        raise ValueError(msg)

    return X, fids


def standardize(X, logger):
    """Standardize features to zero mean and unit variance."""
    logger.info("Standardizing features...")
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    # Avoid division by zero for constant features
    std[std == 0] = 1.0
    X_std = (X - mean) / std
    logger.info(f"  Feature means  : {np.round(mean, 3)}")
    logger.info(f"  Feature stdevs : {np.round(std, 3)}")
    return X_std, mean, std


def run_pca(X_std, n_components, features, output_dir, param_str, logger):
    """
    Run PCA, report explained variance, save loadings to CSV.
    Returns transformed data and PCA object.
    """
    from sklearn.decomposition import PCA

    logger.info(f"Running PCA (n_components={n_components})...")

    pca     = PCA(n_components=n_components)
    X_pca   = pca.fit_transform(X_std)

    # Report explained variance
    explained     = pca.explained_variance_ratio_
    cumulative    = np.cumsum(explained)
    logger.info(f"  Explained variance per component:")
    for i, (e, c) in enumerate(zip(explained, cumulative)):
        logger.info(f"    PC{i+1}: {100*e:.1f}% (cumulative: {100*c:.1f}%)")

    # Report feature loadings for PC1 and PC2
    logger.info(f"  Feature loadings PC1 (most important features):")
    loadings_pc1 = sorted(
        zip(features, pca.components_[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feat, loading in loadings_pc1:
        logger.info(f"    {feat:25s}: {loading:+.3f}")

    logger.info(f"  Feature loadings PC2:")
    loadings_pc2 = sorted(
        zip(features, pca.components_[1]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feat, loading in loadings_pc2:
        logger.info(f"    {feat:25s}: {loading:+.3f}")

    # Save full loadings to CSV
    os.makedirs(output_dir, exist_ok=True)
    loadings_path = os.path.join(
        output_dir, f"pca_loadings_{param_str}.csv"
    )
    with open(loadings_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["feature"] + [f"PC{i+1}" for i in range(n_components)]
        writer.writerow(header)
        for j, feat in enumerate(features):
            row = [feat] + [
                round(pca.components_[i][j], 4)
                for i in range(n_components)
            ]
            writer.writerow(row)

    # Save explained variance to CSV
    variance_path = os.path.join(
        output_dir, f"pca_variance_{param_str}.csv"
    )
    with open(variance_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "explained_variance_pct",
                         "cumulative_pct"])
        for i, (e, c) in enumerate(zip(explained, cumulative)):
            writer.writerow([f"PC{i+1}", round(100*e, 2), round(100*c, 2)])

    logger.info(f"  PCA loadings saved  : {loadings_path}")
    logger.info(f"  PCA variance saved  : {variance_path}")

    return X_pca, pca


def elbow_method(X_pca, max_k, output_dir, param_str, logger):
    """
    Run K-Means for k=2..max_k, record inertia.
    Suggest optimal k using the elbow point.
    Saves inertia values to CSV.
    """
    from sklearn.cluster import KMeans

    logger.info(f"Running elbow method (k=2 to {max_k})...")

    ks       = list(range(2, max_k + 1))
    inertias = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        inertias.append(km.inertia_)
        logger.info(f"  k={k:2d}: inertia={km.inertia_:.1f}")

    # Find elbow using maximum curvature (second derivative)
    inertias_arr = np.array(inertias)
    if len(inertias_arr) >= 3:
        second_deriv = np.diff(np.diff(inertias_arr))
        elbow_idx    = np.argmax(second_deriv) + 2  # +2 offset for diffs
        suggested_k  = ks[elbow_idx]
        logger.info(f"  Suggested k from elbow: {suggested_k}")
    else:
        suggested_k = ks[0]

    # Save to CSV
    elbow_path = os.path.join(
        output_dir, f"elbow_{param_str}.csv"
    )
    with open(elbow_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "inertia"])
        for k, inertia in zip(ks, inertias):
            writer.writerow([k, round(inertia, 2)])

    logger.info(f"  Elbow data saved: {elbow_path}")
    return suggested_k, ks, inertias


def run_kmeans(X_pca, k, logger):
    """Run K-Means clustering, return cluster labels."""
    from sklearn.cluster import KMeans

    logger.info(f"Running K-Means (k={k})...")
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)

    # Report cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"  Cluster sizes:")
    for cluster_id, count in zip(unique, counts):
        logger.info(f"    Cluster {cluster_id:2d}: {count:,} crowns "
                    f"({100*count/len(labels):.1f}%)")

    return labels, km


def save_cluster_stats(X, labels, fids, features, k,
                       output_dir, param_str, logger):
    """
    Save per-cluster feature means to CSV.
    Useful for interpreting what each cluster represents.
    """
    stats_path = os.path.join(
        output_dir, f"cluster_stats_{param_str}.csv"
    )
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "n_crowns"] + features)
        for cluster_id in range(k):
            mask       = labels == cluster_id
            cluster_X  = X[mask]
            means      = np.mean(cluster_X, axis=0)
            row        = ([cluster_id, np.sum(mask)] +
                          [round(float(m), 4) for m in means])
            writer.writerow(row)

    logger.info(f"  Cluster stats saved: {stats_path}")


def write_labels_to_gpkg(gpkg_path, layer_name, fids, labels,
                         cluster_param_str, logger):
    """
    Write cluster labels back to GeoPackage layer.
    Adds 'cluster_id' field.
    Crowns that were filtered out get cluster_id = -1.
    """
    logger.info(f"Writing cluster labels to GeoPackage...")

    ds    = ogr.Open(gpkg_path, 1)
    layer = ds.GetLayerByName(layer_name)
    clusterfield = f"cluster_id_{cluster_param_str}"

    # Add fields if missing
    layer_defn  = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName()
                   for i in range(layer_defn.GetFieldCount())]

    if clusterfield not in field_names:
        layer.CreateField(ogr.FieldDefn(clusterfield, ogr.OFTInteger))

    # Build FID -> label lookup
    label_lookup = {fid: int(label)
                    for fid, label in zip(fids, labels)}

    # Write labels -> -1 for filtered crowns
    layer.ResetReading()
    written  = 0
    excluded = 0

    for feature in layer:
        fid   = feature.GetFID()
        label = label_lookup.get(fid, -1)
        feature.SetField(clusterfield,  label)
        layer.SetFeature(feature)
        if label >= 0:
            written += 1
        else:
            excluded += 1

    ds.Destroy()
    logger.info(f"  Labels written   : {written:,}")
    logger.info(f"  Excluded (-1)    : {excluded:,}")


def save_pca_scores(X_pca, fids, output_dir, param_str, logger):
    """Save PCA scores per crown to CSV for external plotting."""
    scores_path = os.path.join(
        output_dir, f"pca_scores_{param_str}.csv"
    )
    n_components = X_pca.shape[1]
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fid"] + [f"PC{i+1}" for i in range(n_components)])
        for fid, scores in zip(fids, X_pca):
            writer.writerow([fid] + [round(float(s), 4) for s in scores])
    logger.info(f"  PCA scores saved: {scores_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("05_classification")

    try:
        logger.info(f"Parameters:")
        logger.info(f"  crowns-layer   = {args.crowns_layer}")
        logger.info(f"  k              = {args.k}")
        logger.info(f"  pca-components = {args.pca_components}")
        logger.info(f"  min-points     = {args.min_points}")
        logger.info(f"  elbow-max-k    = {args.elbow_max_k}")
        logger.info(f"  features       = {FEATURES}")

        param_str  = (f"{args.crowns_layer}"
                      f"_k{args.k}"
                      f"_pca{args.pca_components}"
                      f"_mp{args.min_points}")
        cluster_param_str = (f"k{args.k}"
                      f"_pca{args.pca_components}"
                      f"_mp{args.min_points}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_dir = os.path.join(OUTPUT_DIR, "classification")
        os.makedirs(output_dir, exist_ok=True)

        # Check sklearn is available
        try:
            import sklearn
            logger.info(f"  scikit-learn version: {sklearn.__version__}")
        except ImportError:
            msg = ("scikit-learn not found. Install with: "
                   "pip install scikit-learn")
            logger.error(msg)
            raise ImportError(msg)

        # 1. Load features
        X, fids = load_crown_features(
            INPUT_CROWNS, args.crowns_layer,
            FEATURES, args.min_points, logger
        )
        logger.info(f"  Feature matrix shape: {X.shape}")

        # 2. Standardize
        X_std, feat_mean, feat_std = standardize(X, logger)

        # 3. PCA
        X_pca, pca = run_pca(
            X_std, args.pca_components,
            FEATURES, output_dir, param_str, logger
        )

        # Save PCA scores
        save_pca_scores(X_pca, fids, output_dir, param_str, logger)

        # 4. Elbow method
        suggested_k, ks, inertias = elbow_method(
            X_pca, args.elbow_max_k, output_dir, param_str, logger
        )
        logger.info(f"  Elbow suggests k={suggested_k} "
                    f"(you specified k={args.k})")

        # 5. K-Means with specified k
        labels, km = run_kmeans(X_pca, args.k, logger)

        # 6. Save cluster stats
        save_cluster_stats(
            X, labels, fids, FEATURES, args.k,
            output_dir, param_str, logger
        )

        # 7. Write labels back to GeoPackage
        write_labels_to_gpkg(
            INPUT_CROWNS, args.crowns_layer,
            fids, labels, cluster_param_str, logger
        )

        logger.info(f"Done!")
        logger.info(f"  Crowns classified  : {len(fids):,}")
        logger.info(f"  Clusters           : {args.k}")
        logger.info(f"  Elbow suggested k  : {suggested_k}")
        logger.info(f"  Results in         : {INPUT_CROWNS}")
        logger.info(f"  Layer              : {args.crowns_layer}")
        logger.info(f"  Diagnostics in     : {output_dir}")
        logger.info("Next steps:")
        logger.info("  1. Check elbow CSV to decide if k should change")
        logger.info("  2. Check cluster_stats CSV to interpret clusters")
        logger.info("  3. Load crowns.gpkg in QGIS, style by cluster_id")
        logger.info("  4. Re-run with different k if needed")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise