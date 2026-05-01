"""
feature_correlation.py
======================
Computes and visualises feature correlations for crown classification features.
Saves a heatmap image and a correlation matrix CSV.

Usage:
    python scripts/feature_correlation.py
    python scripts/feature_correlation.py --crowns-layer crowns_treetops_w5_mh2.0_t0.001_c0.01_mh2.0
    python scripts/feature_correlation.py --min-points 30 --method spearman
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
        description="Compute feature correlation heatmap for crown features."
    )
    parser.add_argument(
        "--crowns-layer", type=str, default=None,
        help="Layer name inside crowns.gpkg (default: first layer)"
    )
    parser.add_argument(
        "--min-points", type=int, default=30,
        help="Minimum n_points to include crown (default: 30)"
    )
    parser.add_argument(
        "--method", type=str, default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method: pearson or spearman (default: pearson). "
             "Use spearman for non-normal distributions."
    )
    return parser.parse_args()


def load_features(gpkg_path, layer_name, features, min_points, logger):
    """Load feature matrix from GeoPackage with same filters as script 05."""
    logger.info(f"Loading features from: {os.path.basename(gpkg_path)}")
    ds = ogr.Open(gpkg_path, 0)
    if ds is None:
        msg = f"Cannot open: {gpkg_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    available = [ds.GetLayerByIndex(i).GetName()
                 for i in range(ds.GetLayerCount())]
    if layer_name is None:
        layer_name = available[0]
        logger.info(f"  Using layer: '{layer_name}'")
    elif layer_name not in available:
        msg = f"Layer '{layer_name}' not found. Available: {available}"
        logger.error(msg)
        raise ValueError(msg)

    layer      = ds.GetLayerByName(layer_name)
    n_total    = layer.GetFeatureCount()
    rows       = []
    n_edge     = 0
    n_low      = 0
    n_null     = 0

    layer.ResetReading()
    for feature in layer:
        if feature.GetField("is_edge") == 1:
            n_edge += 1
            continue
        n_pts = feature.GetField("n_points")
        if n_pts is None or n_pts < min_points:
            n_low += 1
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
        rows.append(row)

    ds = None
    X  = np.array(rows)

    logger.info(f"  Total crowns    : {n_total:,}")
    logger.info(f"  Edge removed    : {n_edge:,}")
    logger.info(f"  Low pts removed : {n_low:,}")
    logger.info(f"  Null removed    : {n_null:,}")
    logger.info(f"  Used for corr   : {len(rows):,}")
    return X, layer_name


def compute_correlation(X, method, logger):
    """Compute correlation matrix using pearson or spearman."""
    logger.info(f"Computing {method} correlation matrix...")

    if method == "spearman":
        from scipy.stats import spearmanr
        corr, _ = spearmanr(X)
        corr    = np.array(corr)
    else:
        corr = np.corrcoef(X.T)

    logger.info(f"  Matrix shape: {corr.shape}")
    return corr


def save_correlation_csv(corr, features, output_path, logger):
    """Save correlation matrix to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature"] + features)
        for i, feat in enumerate(features):
            row = [feat] + [round(float(corr[i, j]), 4)
                            for j in range(len(features))]
            writer.writerow(row)
    logger.info(f"  Correlation CSV saved: {output_path}")


def plot_heatmap(corr, features, method, output_path, layer_name, logger):
    """
    Plot a styled correlation heatmap and save as PNG.
    Uses matplotlib and seaborn.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        msg = "matplotlib not found. Install: pip install matplotlib"
        logger.error(msg)
        raise ImportError(msg)

    try:
        import seaborn as sns
    except ImportError:
        msg = "seaborn not found. Install: pip install seaborn"
        logger.error(msg)
        raise ImportError(msg)

    logger.info("Generating correlation heatmap...")

    n       = len(features)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Custom diverging colormap — blue=negative, white=zero, red=positive
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw heatmap
    sns.heatmap(
        corr,
        ax=ax,
        cmap=cmap,
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 9},
        linewidths=0.5,
        linecolor="white",
        square=True,
        xticklabels=features,
        yticklabels=features,
        cbar_kws={"label": f"{method.capitalize()} correlation",
                  "shrink": 0.8}
    )

    # Highlight highly correlated pairs (|r| > 0.8) with a border
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr[i, j]) > 0.8:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False, edgecolor="black", lw=2
                ))

    # Style
    ax.set_title(
        f"Feature Correlation Matrix ({method.capitalize()})\n"
        f"Layer: {layer_name}\n"
        f"Black borders = |r| > 0.8 (consider removing one feature)",
        fontsize=12, pad=15
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=9
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), rotation=0, fontsize=9
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Heatmap saved: {output_path}")


def report_high_correlations(corr, features, threshold, logger):
    """Log pairs of features with correlation above threshold."""
    logger.info(f"Highly correlated pairs (|r| > {threshold}):")
    found = False
    n     = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if abs(r) > threshold:
                direction = "positive" if r > 0 else "negative"
                logger.info(f"  {features[i]:25s} <-> {features[j]:25s}"
                            f"  r={r:+.3f}  ({direction})")
                found = True
    if not found:
        logger.info(f"  No pairs above threshold {threshold}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("feature_correlation")

    try:
        logger.info(f"Parameters:")
        logger.info(f"  method     = {args.method}")
        logger.info(f"  min-points = {args.min_points}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1. Load features
        X, layer_name = load_features(
            INPUT_CROWNS, args.crowns_layer,
            FEATURES, args.min_points, logger
        )

        param_str = f"{layer_name}_{args.method}_mp{args.min_points}"

        # 2. Compute correlation
        corr = compute_correlation(X, args.method, logger)

        # 3. Report high correlations
        report_high_correlations(corr, FEATURES, threshold=0.7, logger=logger)

        # 4. Save CSV
        csv_path = os.path.join(
            OUTPUT_DIR, f"correlation_{param_str}.csv"
        )
        save_correlation_csv(corr, FEATURES, csv_path, logger)

        # 5. Plot heatmap
        img_path = os.path.join(
            OUTPUT_DIR, f"correlation_{param_str}.png"
        )
        plot_heatmap(corr, FEATURES, args.method, img_path,
                     layer_name, logger)

        logger.info("Done!")
        logger.info(f"  CSV     : {csv_path}")
        logger.info(f"  Heatmap : {img_path}")
        logger.info("Tip: features with |r| > 0.8 are redundant.")
        logger.info("Consider dropping one from each highly correlated pair")
        logger.info("before running 05_classification.py")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise