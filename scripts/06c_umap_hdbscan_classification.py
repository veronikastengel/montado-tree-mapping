"""
06c_umap_hdbscan.py
===================
Unsupervised classification of tree crowns using
UMAP dimensionality reduction + HDBSCAN density clustering.

Pipeline:
    1. Load and filter crown features (same as script 05)
    2. Standardize features
    3. UMAP dimensionality reduction to 2D (for visualization) and nD (for clustering)
    4. HDBSCAN clustering on nD UMAP embedding
    5. Write cluster labels back to GeoPackage
    6. Save UMAP coordinates and cluster stats to CSV
    7. Save diagnostic plots (UMAP scatter, cluster size distribution)

Usage:
    python scripts/06c_umap_hdbscan_classification.py
    python scripts/06c_umap_hdbscan_classification.py --crowns-layer crowns_treetops_lidar_ra2.0_rb0.15_mh2.0_md_4.5_c0.05_mh2.0
    python scripts/06c_umap_hdbscan_classification.py --umap-neighbors 15 --umap-components 5 --min-cluster-size 50

Arguments:
    --crowns-layer          layer name inside crowns.gpkg (default: first layer)
    --min-points            minimum n_points to include crown (default: 30)
    --umap-neighbors        UMAP n_neighbors parameter, controls local vs global
                            structure (default: 15, higher = more global)
    --umap-components       UMAP output dimensions for clustering (default: 5)
                            2D embedding always also computed for visualization
    --umap-min-dist         UMAP min_dist parameter, controls cluster tightness
                            in 2D plot (default: 0.1)
    --min-cluster-size      HDBSCAN minimum cluster size (default: 50)
                            smaller = more clusters, larger = fewer
    --min-samples           HDBSCAN min_samples, controls noise sensitivity
                            (default: 5, higher = more noise points)
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
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs", "classification")

FEATURES = [
    "mean_height",
    #"max_height",
    "height_width_ratio",
    "mean_intensity",
    "mean_ndvi",
    "rugosity",
    "vert_dist_top25",
    #"vert_dist_mid50",
    "vert_dist_bot25",
    "point_density",
    "landscape_conf",
    "lc_olive",
    "lc_montado",
    "lc_eucalyptus",
    #"lc_broadleaf",
    "lc_pine",
    "lc_shrubland"
]
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="UMAP + HDBSCAN classification of tree crowns."
    )
    parser.add_argument(
        "--crowns-layer",
        type=str,
        default=None,
        help="Layer name inside crowns.gpkg (default: first layer)"
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=30,
        help="Minimum n_points to include crown (default: 30)"
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15). Higher = more global structure."
    )
    parser.add_argument(
        "--umap-components",
        type=int,
        default=5,
        help="UMAP output dimensions for clustering (default: 5). "
             "2D embedding always computed separately for visualization."
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist for 2D visualization (default: 0.1). "
             "Lower = tighter clusters in plot."
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=50,
        help="HDBSCAN minimum cluster size (default: 50). "
             "Smaller = more clusters."
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="HDBSCAN min_samples (default: 5). "
             "Higher = more conservative, more noise points."
    )
    return parser.parse_args()


def load_crown_features(gpkg_path, layer_name, features,
                        min_points, logger):
    """Load and filter crown features from GeoPackage."""
    logger.info(f"Loading crown features: {os.path.basename(gpkg_path)}")
    ds = ogr.Open(gpkg_path, 0)
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
    else:
        logger.info(f"  Using layer: '{layer_name}'")

    layer   = ds.GetLayerByName(layer_name)
    n_total = layer.GetFeatureCount()
    logger.info(f"  Total crowns: {n_total:,}")

    # Check required fields
    layer_defn  = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName()
                   for i in range(layer_defn.GetFieldCount())]

    required = features + ["n_points", "is_edge"]
    missing  = [f for f in required if f not in field_names]
    if missing:
        msg = f"Missing fields: {missing}. Run 04_feature_extraction.py first."
        logger.error(msg)
        raise ValueError(msg)

    fids         = []
    feature_rows = []
    n_edge       = 0
    n_low_pts    = 0
    n_null       = 0

    layer.ResetReading()
    for feature in layer:
        fid = feature.GetFID()

        if feature.GetField("is_edge") == 1:
            n_edge += 1
            continue

        n_pts = feature.GetField("n_points")
        if n_pts is None or n_pts < min_points:
            n_low_pts += 1
            continue

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
    X  = np.array(feature_rows)

    logger.info(f"  Removed edge       : {n_edge:,}")
    logger.info(f"    Removed < {min_points} pts : {n_low_pts:,}")
    logger.info(f"  Removed null       : {n_null:,}")
    logger.info(f"  Remaining          : {len(fids):,}")

    if len(fids) == 0:
        msg = "No crowns remaining after filtering"
        logger.error(msg)
        raise ValueError(msg)

    return X, fids, layer_name


def standardize(X, features, logger):
    """Standardize features to zero mean and unit variance."""
    logger.info("Standardizing features...")
    mean    = X.mean(axis=0)
    std     = X.std(axis=0)
    std[std == 0] = 1.0
    X_std   = (X - mean) / std

    for i, f in enumerate(features):
        logger.info(f"  {f:25s}: mean={mean[i]:.3f}, std={std[i]:.3f}")

    return X_std, mean, std


def run_umap_2d(X_std, n_neighbors, min_dist, logger):
    """
    UMAP to 2D for visualization only.
    Uses lower min_dist for tighter visual clusters.
    """
    import umap
    logger.info(f"Running UMAP 2D visualization "
                f"(n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=False
    )
    embedding_2d = reducer_2d.fit_transform(X_std)
    logger.info(f"  2D embedding shape: {embedding_2d.shape}")
    return embedding_2d


def run_umap_nd(X_std, n_neighbors, n_components, logger):
    """
    UMAP to nD for clustering.
    Uses min_dist=0 for maximally separated clusters.
    """
    import umap
    logger.info(f"Running UMAP {n_components}D for clustering "
                f"(n_neighbors={n_neighbors})...")
    reducer_nd = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        random_state=42,
        verbose=False
    )
    embedding_nd = reducer_nd.fit_transform(X_std)
    logger.info(f"  {n_components}D embedding shape: {embedding_nd.shape}")
    return embedding_nd


def run_hdbscan(embedding_nd, min_cluster_size, min_samples, logger):
    """
    HDBSCAN clustering on nD UMAP embedding.
    Returns labels (-1 = noise) and probabilities.
    """
    import hdbscan
    logger.info(f"Running HDBSCAN "
                f"(min_cluster_size={min_cluster_size}, "
                f"min_samples={min_samples})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",  # excess of mass — standard
        prediction_data=True
    )
    labels      = clusterer.fit_predict(embedding_nd)
    probs       = clusterer.probabilities_

    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise     = np.sum(labels == -1)

    logger.info(f"  Clusters found     : {n_clusters}")
    logger.info(f"  Noise points (-1)  : {n_noise:,} "
                f"({100*n_noise/len(labels):.1f}%)")

    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"  Cluster sizes:")
    for cluster_id, count in zip(unique, counts):
        label_str = "noise" if cluster_id == -1 else str(cluster_id)
        logger.info(f"    Cluster {label_str:5s}: {count:,} crowns "
                    f"({100*count/len(labels):.1f}%)")

    return labels, probs, clusterer


def save_cluster_stats(X, labels, features, output_dir,
                       param_str, logger):
    """Save per-cluster feature means to CSV."""
    stats_path = os.path.join(output_dir, f"hdbscan_stats_{param_str}.csv")
    unique_labels = sorted(set(labels))

    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "n_crowns", "is_noise"] + features)
        for cluster_id in unique_labels:
            mask      = labels == cluster_id
            cluster_X = X[mask]
            means     = np.mean(cluster_X, axis=0)
            is_noise  = 1 if cluster_id == -1 else 0
            row       = ([cluster_id, int(np.sum(mask)), is_noise] +
                         [round(float(m), 4) for m in means])
            writer.writerow(row)

    logger.info(f"  Cluster stats saved: {stats_path}")
    return stats_path


def save_umap_coords(embedding_2d, embedding_nd, fids, labels,
                     probs, output_dir, param_str, logger):
    """
    Save UMAP 2D coordinates + cluster labels + probabilities to CSV.
    Useful for plotting in Tableau or Python outside this script.
    """
    coords_path = os.path.join(
        output_dir, f"umap_coords_{param_str}.csv"
    )
    n_nd = embedding_nd.shape[1]

    with open(coords_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (["fid", "umap_x", "umap_y"] +
                  [f"umap_{i+1}" for i in range(n_nd)] +
                  ["cluster_id", "cluster_prob"])
        writer.writerow(header)
        for i, fid in enumerate(fids):
            row = ([fid,
                    round(float(embedding_2d[i, 0]), 4),
                    round(float(embedding_2d[i, 1]), 4)] +
                   [round(float(embedding_nd[i, j]), 4)
                    for j in range(n_nd)] +
                   [int(labels[i]),
                    round(float(probs[i]), 4)])
            writer.writerow(row)

    logger.info(f"  UMAP coords saved: {coords_path}")
    return coords_path


def save_scatter_plot(embedding_2d, labels, output_dir,
                      param_str, logger):
    """
    Save UMAP 2D scatter plot coloured by cluster as PNG.
    Uses matplotlib — bundled with QGIS Python.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        logger.info("Saving UMAP scatter plot...")

        unique_labels = sorted(set(labels))
        n_clusters    = len([l for l in unique_labels if l >= 0])

        # Color map — noise points grey
        cmap    = cm.get_cmap("tab20", max(n_clusters, 1))
        fig, ax = plt.subplots(figsize=(12, 10))

        for cluster_id in unique_labels:
            mask  = labels == cluster_id
            color = "lightgrey" if cluster_id == -1 else cmap(cluster_id)
            label = "noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            alpha = 0.3 if cluster_id == -1 else 0.7
            size  = 2 if cluster_id == -1 else 4

            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[color],
                label=label,
                alpha=alpha,
                s=size,
                rasterized=True
            )

        ax.set_title(f"UMAP + HDBSCAN — {n_clusters} clusters\n{param_str}",
                     fontsize=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=4, bbox_to_anchor=(1.05, 1),
                  loc="upper left", fontsize=8)
        plt.tight_layout()

        plot_path = os.path.join(
            output_dir, f"umap_scatter_{param_str}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Scatter plot saved: {plot_path}")
        return plot_path

    except Exception as e:
        logger.info(f"  Plot skipped (matplotlib error): {e}")
        return None


def write_labels_to_gpkg(gpkg_path, layer_name, fids, labels,
                         probs, param_str, logger):
    """
    Write HDBSCAN cluster labels and probabilities to GeoPackage.
    Adds hdbscan_cluster and hdbscan_prob fields.
    Filtered crowns get hdbscan_cluster = -2 (distinguished from
    HDBSCAN noise which is -1).
    """
    logger.info("Writing labels to GeoPackage...")

    ds    = ogr.Open(gpkg_path, 1)
    layer = ds.GetLayerByName(layer_name)

    layer_defn  = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName()
                   for i in range(layer_defn.GetFieldCount())]
    
    field_cluster = f"hdbscan_cluster_{param_str}"
    field_prob = f"hdbscan_prob_{param_str}"

    if field_cluster not in field_names:
        layer.CreateField(
            ogr.FieldDefn(field_cluster, ogr.OFTInteger)
        )
    if field_prob not in field_names:
        layer.CreateField(
            ogr.FieldDefn(field_prob, ogr.OFTReal)
        )

    # Build lookup: fid -> (label, prob)
    lookup = {
        fid: (int(label), float(prob))
        for fid, label, prob in zip(fids, labels, probs)
    }

    layer.ResetReading()
    written  = 0
    excluded = 0

    for feature in layer:
        fid = feature.GetFID()
        if fid in lookup:
            label, prob = lookup[fid]
            feature.SetField(field_cluster, label)
            feature.SetField(field_prob,    prob)
            written += 1
        else:
            # Filtered out crown -> mark as -2
            feature.SetField(field_cluster, -2)
            feature.SetField(field_prob,    0.0)
            excluded += 1
        layer.SetFeature(feature)

    ds.Destroy()
    logger.info(f"  Labels written   : {written:,}")
    logger.info(f"  Excluded (-2)    : {excluded:,}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("06c_umap_hdbscan_classification")

    try:
        logger.info(f"Parameters:")
        logger.info(f"  umap-neighbors   = {args.umap_neighbors}")
        logger.info(f"  umap-components  = {args.umap_components}")
        logger.info(f"  umap-min-dist    = {args.umap_min_dist}")
        logger.info(f"  min-cluster-size = {args.min_cluster_size}")
        logger.info(f"  min-samples      = {args.min_samples}")
        logger.info(f"  min-points       = {args.min_points}")
        logger.info(f"  features         = {FEATURES}")

        param_str = (f"nn{args.umap_neighbors}"
                     f"_uc{args.umap_components}"
                     f"_mcs{args.min_cluster_size}"
                     f"_ms{args.min_samples}"
                     f"_mp{args.min_points}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1. Load features
        X, fids, layer_name = load_crown_features(
            INPUT_CROWNS, args.crowns_layer,
            FEATURES, args.min_points, logger
        )
        logger.info(f"  Feature matrix: {X.shape}")

        # 2. Standardize
        X_std, feat_mean, feat_std = standardize(X, FEATURES, logger)

        # 3. UMAP 2D for visualization
        embedding_2d = run_umap_2d(
            X_std,
            args.umap_neighbors,
            args.umap_min_dist,
            logger
        )

        # 4. UMAP nD for clustering
        if args.umap_components == 2:
            # Reuse 2D if components=2
            embedding_nd = embedding_2d
            logger.info("  Using 2D embedding for clustering "
                        "(umap-components=2)")
        else:
            embedding_nd = run_umap_nd(
                X_std,
                args.umap_neighbors,
                args.umap_components,
                logger
            )

        # 5. HDBSCAN clustering
        labels, probs, clusterer = run_hdbscan(
            embedding_nd,
            args.min_cluster_size,
            args.min_samples,
            logger
        )

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int(np.sum(labels == -1))

        # 6. Save cluster stats
        save_cluster_stats(
            X, labels, FEATURES, OUTPUT_DIR, param_str, logger
        )

        # 7. Save UMAP coordinates
        save_umap_coords(
            embedding_2d, embedding_nd, fids, labels, probs,
            OUTPUT_DIR, param_str, logger
        )

        # 8. Save scatter plot
        save_scatter_plot(
            embedding_2d, labels, OUTPUT_DIR, param_str, logger
        )

        # 9. Write labels to GeoPackage
        write_labels_to_gpkg(
            INPUT_CROWNS, layer_name, fids, labels, probs,
            param_str, logger
        )

        logger.info(f"Done!")
        logger.info(f"  Crowns classified : {len(fids):,}")
        logger.info(f"  Clusters found    : {n_clusters}")
        logger.info(f"  Noise points      : {n_noise:,} "
                    f"({100*n_noise/len(labels):.1f}%)")
        logger.info(f"  Results in        : {INPUT_CROWNS}")
        logger.info(f"  Layer             : {layer_name}")
        logger.info(f"  Diagnostics in    : {OUTPUT_DIR}")

        logger.info("Interpretation guide:")
        logger.info("  hdbscan_cluster = -2 : filtered out (edge/low points)")
        logger.info("  hdbscan_cluster = -1 : HDBSCAN noise (ambiguous crown)")
        logger.info("  hdbscan_cluster >= 0 : cluster member")
        logger.info("  hdbscan_prob         : confidence 0-1 within cluster")
        logger.info("Next steps:")
        logger.info("  Open UMAP scatter PNG -> look for clear separation")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise