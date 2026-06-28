"""
06b_gmm_classification.py
=========================
Unsupervised classification of tree crowns using PCA + Gaussian Mixture
Model (GMM) with Bayesian Information Criterion (BIC) for automatic
selection of optimal number of components.

Usage:
    python scripts/06b_gmm_classification.py
    python scripts/06b_gmm_classification.py --crowns-layer crowns_treetops_lidar_ra2.0_rb0.15_mh2.0_md_4.5_c0.05_mh2.0
    python scripts/06b_gmm_classification.py --crowns-layer crowns_treetops_lidar_ra2.0_rb0.15_mh2.0_md_4.5_c0.05_mh2.0 --max-components 18 --covariance-type full

Arguments:
    --crowns-layer      layer name inside crowns.gpkg (default: first layer)
    --max-components    maximum number of GMM components to test (default: 12)
    --pca-components    number of PCA components as input to GMM (default: 5)
    --min-points        minimum n_points to include crown (default: 30)
    --covariance-type   GMM covariance type: full, tied, diag, spherical
                        (default: full -> most flexible, best for species data)
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
        description="GMM + BIC unsupervised classification of tree crowns."
    )
    parser.add_argument(
        "--crowns-layer", type=str, default=None,
        help="Layer name inside crowns.gpkg (default: first layer found)"
    )
    parser.add_argument(
        "--max-components", type=int, default=12,
        help="Maximum GMM components to test with BIC (default: 12)"
    )
    parser.add_argument(
        "--pca-components", type=int, default=5,
        help="PCA components to use as GMM input (default: 5)"
    )
    parser.add_argument(
        "--min-points", type=int, default=30,
        help="Minimum n_points to include crown (default: 30)"
    )
    parser.add_argument(
        "--covariance-type", type=str, default="full",
        choices=["full", "tied", "diag", "spherical"],
        help="GMM covariance type (default: full)"
    )
    return parser.parse_args()


def load_crown_features(gpkg_path, layer_name, features, min_points, logger):
    """Load crown features with same filters as script 05."""
    logger.info(f"Loading crown features from: {os.path.basename(gpkg_path)}")
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
        logger.info(f"  Using: '{layer_name}'")
    elif layer_name not in available:
        msg = f"Layer '{layer_name}' not found. Available: {available}"
        logger.error(msg)
        raise ValueError(msg)

    layer      = ds.GetLayerByName(layer_name)
    n_total    = layer.GetFeatureCount()
    fids       = []
    rows       = []
    n_edge = n_low = n_null = 0

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
        fids.append(feature.GetFID())
        rows.append(row)

    ds = None
    X  = np.array(rows)
    logger.info(f"  Total: {n_total:,} | Edge: {n_edge:,} | "
                f"Low pts: {n_low:,} | Null: {n_null:,} | "
                f"Used: {len(fids):,}")
    return X, fids, layer_name


def standardize(X, logger):
    """Standardize to zero mean unit variance."""
    logger.info("Standardizing features...")
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def run_pca(X_std, n_components, logger):
    """Run PCA, return transformed data and explained variance."""
    from sklearn.decomposition import PCA
    logger.info(f"Running PCA (n_components={n_components})...")
    pca   = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    expl  = pca.explained_variance_ratio_
    cum   = np.cumsum(expl)
    for i, (e, c) in enumerate(zip(expl, cum)):
        logger.info(f"  PC{i+1}: {100*e:.1f}% (cumulative: {100*c:.1f}%)")
    return X_pca, pca


def fit_gmm_bic(X_pca, max_components, cov_type, output_dir,
                param_str, logger):
    """
    Fit GMM for n_components = 1..max_components.
    Select best model by lowest BIC score.
    BIC penalises complexity — lower is better.
    Saves BIC scores to CSV.
    """
    from sklearn.mixture import GaussianMixture

    logger.info(f"Fitting GMM with BIC selection "
                f"(covariance={cov_type}, max_k={max_components})...")

    bic_scores = []
    aic_scores = []
    models     = []
    n_range    = range(1, max_components + 1)

    for n in n_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type=cov_type,
            random_state=42,
            n_init=15,       # multiple initialisations for stability
            max_iter=200
        )
        gmm.fit(X_pca)
        bic = gmm.bic(X_pca)
        aic = gmm.aic(X_pca)
        bic_scores.append(bic)
        aic_scores.append(aic)
        models.append(gmm)
        logger.info(f"  n={n:2d}: BIC={bic:,.1f}  AIC={aic:,.1f}"
                    f"  converged={gmm.converged_}")

    # Best by BIC
    best_idx = int(np.argmin(bic_scores))
    best_n   = list(n_range)[best_idx]
    best_gmm = models[best_idx]
    logger.info(f"  Best BIC at n={best_n} "
                f"(BIC={bic_scores[best_idx]:,.1f})")

    # Save BIC/AIC to CSV
    bic_path = os.path.join(output_dir, f"gmm_bic_{param_str}.csv")
    with open(bic_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "bic", "aic"])
        for n, bic, aic in zip(n_range, bic_scores, aic_scores):
            writer.writerow([n, round(bic, 2), round(aic, 2)])
    logger.info(f"  BIC/AIC scores saved: {bic_path}")

    return best_gmm, best_n, bic_scores, aic_scores, list(n_range)


def plot_bic(n_range, bic_scores, aic_scores, best_n,
             output_dir, param_str, logger):
    """Plot BIC and AIC curves to visualise model selection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("  matplotlib not available, skipping BIC plot")
        return

    logger.info("Plotting BIC/AIC curves...")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(n_range, bic_scores, "o-", color="#2196F3",
            linewidth=2, markersize=6, label="BIC")
    ax.plot(n_range, aic_scores, "s--", color="#FF9800",
            linewidth=2, markersize=6, label="AIC")
    ax.axvline(x=best_n, color="#F44336", linestyle=":",
               linewidth=2, label=f"Best BIC (n={best_n})")

    ax.set_xlabel("Number of components", fontsize=12)
    ax.set_ylabel("Score (lower = better)", fontsize=12)
    ax.set_title("GMM Model Selection: BIC and AIC\n"
                 "Lower BIC = better fit with less complexity",
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(n_range)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"gmm_bic_{param_str}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  BIC plot saved: {plot_path}")


def get_labels_and_probs(gmm, X_pca, logger):
    """Get hard labels and soft probabilities from fitted GMM."""
    labels = gmm.predict(X_pca)
    probs  = gmm.predict_proba(X_pca)
    max_prob = probs.max(axis=1)

    # Report cluster sizes and mean confidence
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Cluster sizes and mean assignment probability:")
    for cid, cnt in zip(unique, counts):
        mask      = labels == cid
        mean_conf = max_prob[mask].mean()
        logger.info(f"  Cluster {cid:2d}: {cnt:,} crowns  "
                    f"mean confidence={mean_conf:.3f}")

    low_conf = np.sum(max_prob < 0.6)
    logger.info(f"  Low confidence (<0.6): {low_conf:,} crowns "
                f"({100*low_conf/len(labels):.1f}%) — ambiguous crowns")
    return labels, probs, max_prob


def save_cluster_stats(X, labels, features, n_components,
                       output_dir, param_str, logger):
    """Save per-cluster mean feature values to CSV."""
    stats_path = os.path.join(
        output_dir, f"gmm_cluster_stats_{param_str}.csv"
    )
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "n_crowns"] + features)
        for cid in range(n_components):
            mask  = labels == cid
            means = np.mean(X[mask], axis=0)
            writer.writerow(
                [cid, int(np.sum(mask))] +
                [round(float(m), 4) for m in means]
            )
    logger.info(f"  Cluster stats saved: {stats_path}")


def write_labels_to_gpkg(gpkg_path, layer_name, fids, labels,
                         max_probs, param_str, cluster_param_str, logger):
    """
    Write GMM cluster labels and assignment confidence back to GeoPackage.
    Adds gmm_cluster and gmm_confidence fields.
    Crowns filtered out get gmm_cluster = -1.
    """
    logger.info("Writing GMM labels to GeoPackage...")
    ds    = ogr.Open(gpkg_path, 1)
    layer = ds.GetLayerByName(layer_name)

    defn        = layer.GetLayerDefn()
    field_names = [defn.GetFieldDefn(i).GetName()
                   for i in range(defn.GetFieldCount())]
    
    field_cluster = f"gmm_cluster_{cluster_param_str}"
    field_conf = f"gmm_confidence_{cluster_param_str}"

    if field_cluster not in field_names:
        layer.CreateField(ogr.FieldDefn(field_cluster, ogr.OFTInteger))
    if field_conf not in field_names:
        layer.CreateField(ogr.FieldDefn(field_conf, ogr.OFTReal))

    label_lookup = {fid: (int(lbl), float(prob))
                    for fid, lbl, prob in zip(fids, labels, max_probs)}

    layer.ResetReading()
    written = excluded = 0
    for feature in layer:
        fid = feature.GetFID()
        if fid in label_lookup:
            lbl, prob = label_lookup[fid]
            feature.SetField(field_cluster,    lbl)
            feature.SetField(field_conf, round(prob, 4))
            written += 1
        else:
            feature.SetField(field_cluster,    -1)
            feature.SetField(field_conf, 0.0)
            excluded += 1
        layer.SetFeature(feature)

    ds.Destroy()
    logger.info(f"  Written  : {written:,}")
    logger.info(f"  Excluded : {excluded:,} (gmm_cluster=-1)")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("06b_gmm_classification")

    try:
        logger.info(f"Parameters:")
        logger.info(f"  max-components  = {args.max_components}")
        logger.info(f"  pca-components  = {args.pca_components}")
        logger.info(f"  covariance-type = {args.covariance_type}")
        logger.info(f"  min-points      = {args.min_points}")

        cluster_param_str = (f"mc{args.max_components}"
                      f"_pca{args.pca_components}"
                      f"_cov{args.covariance_type}"
                      f"_mp{args.min_points}")
        
        try:
            import sklearn
            logger.info(f"  scikit-learn: {sklearn.__version__}")
        except ImportError:
            msg = "scikit-learn not found. Install: pip install scikit-learn"
            logger.error(msg)
            raise ImportError(msg)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1. Load features
        X, fids, layer_name = load_crown_features(
            INPUT_CROWNS, args.crowns_layer,
            FEATURES, args.min_points, logger
        )
        logger.info(f"  Feature matrix: {X.shape}")

        param_str = (f"{layer_name}"
                     f"_cov{args.covariance_type}"
                     f"_pca{args.pca_components}"
                     f"_mp{args.min_points}")

        # 2. Standardize
        X_std, _, _ = standardize(X, logger)

        # 3. PCA
        X_pca, pca = run_pca(X_std, args.pca_components, logger)

        # 4. Fit GMM with BIC selection
        best_gmm, best_n, bic_scores, aic_scores, n_range = fit_gmm_bic(
            X_pca, args.max_components, args.covariance_type,
            OUTPUT_DIR, param_str, logger
        )

        # 5. Plot BIC curve
        plot_bic(n_range, bic_scores, aic_scores, best_n,
                 OUTPUT_DIR, param_str, logger)

        # 6. Get labels and probabilities
        labels, probs, max_probs = get_labels_and_probs(
            best_gmm, X_pca, logger
        )

        # 7. Save cluster stats
        save_cluster_stats(
            X, labels, FEATURES, best_n,
            OUTPUT_DIR, param_str, logger
        )

        # 8. Write labels to GeoPackage
        write_labels_to_gpkg(
            INPUT_CROWNS, layer_name,
            fids, labels, max_probs, param_str, cluster_param_str, logger
        )

        logger.info(f"Done!")
        logger.info(f"  Crowns classified : {len(fids):,}")
        logger.info(f"  Best n components : {best_n} (by BIC)")
        logger.info(f"  Results in        : {INPUT_CROWNS}")
        logger.info(f"  Layer             : {layer_name}")
        logger.info(f"  Diagnostics in    : {OUTPUT_DIR}")
        logger.info("Next steps:")
        logger.info("  Check gmm_bic PNG -> does BIC curve have a clear minimum?")
        logger.info("  Check gmm_cluster_stats CSV to interpret clusters")

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        raise