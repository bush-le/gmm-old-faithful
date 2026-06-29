"""
compare_models.py — Metrics comparison and report generation.

Computes evaluation metrics for all models, writes metrics.txt,
and generates the final comparison plot.
"""
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PLOTS_DIR, LOGS_DIR
from src.metrics import silhouette_score, cluster_separation
from src.em import compute_log_likelihood
from src.visualization import plot_comparison


def compute_all_metrics(X, gmm_labels, gmm_params,
                        kmeans_labels, hierarchical_labels,
                        knn_consistency):
    """
    Compute evaluation metrics for all clustering methods.

    Metrics:
    - Log-Likelihood (GMM only — other methods don't define a likelihood)
    - Silhouette Score (all methods)
    - Cluster Separation Distance (all methods)
    - KNN Consistency (KMeans pseudo-labels)

    Args:
        X: Data of shape (N, D).
        gmm_labels: GMM cluster labels.
        gmm_params: GMM parameters (for log-likelihood).
        kmeans_labels: K-Means labels.
        hierarchical_labels: Hierarchical labels.
        knn_consistency: KNN consistency score.

    Returns:
        dict: All computed metrics.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON — EVALUATION METRICS")
    print("=" * 60)

    metrics = {}

    # --- GMM Metrics ---
    print("\n[GMM]")
    gmm_ll = compute_log_likelihood(X, gmm_params)
    gmm_sil = silhouette_score(X, gmm_labels)
    gmm_sep = cluster_separation(X, gmm_labels)
    print(f"  Log-Likelihood:       {gmm_ll:.6f}")
    print(f"  Silhouette Score:     {gmm_sil:.6f}")
    print(f"  Cluster Separation:   {gmm_sep:.6f}")
    metrics['GMM'] = {
        'log_likelihood': gmm_ll,
        'silhouette': gmm_sil,
        'separation': gmm_sep,
    }

    # --- K-Means Metrics ---
    print("\n[K-Means]")
    km_sil = silhouette_score(X, kmeans_labels)
    km_sep = cluster_separation(X, kmeans_labels)
    print(f"  Silhouette Score:     {km_sil:.6f}")
    print(f"  Cluster Separation:   {km_sep:.6f}")
    metrics['K-Means'] = {
        'log_likelihood': 'N/A',
        'silhouette': km_sil,
        'separation': km_sep,
    }

    # --- Hierarchical Metrics ---
    print("\n[Hierarchical]")
    hc_sil = silhouette_score(X, hierarchical_labels)
    hc_sep = cluster_separation(X, hierarchical_labels)
    print(f"  Silhouette Score:     {hc_sil:.6f}")
    print(f"  Cluster Separation:   {hc_sep:.6f}")
    metrics['Hierarchical'] = {
        'log_likelihood': 'N/A',
        'silhouette': hc_sil,
        'separation': hc_sep,
    }

    # --- KNN ---
    print("\n[KNN]")
    print(f"  Consistency Score:    {knn_consistency:.6f}")
    metrics['KNN'] = {
        'consistency': knn_consistency,
    }

    return metrics


def write_metrics_report(metrics, filepath):
    """
    Write metrics to a structured text report.

    Args:
        metrics (dict): Computed metrics for each model.
        filepath (str): Output file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("EXPERIMENTAL RESULTS — MODEL COMPARISON\n")
        f.write("Dataset: Iris Dataset\n")
        f.write("=" * 65 + "\n\n")

        # Metrics table
        f.write(f"{'Metric':<25} {'GMM':>12} {'K-Means':>12} {'Hierarchical':>14}\n")
        f.write("-" * 65 + "\n")

        # Log-Likelihood
        gmm_ll = metrics['GMM']['log_likelihood']
        f.write(f"{'Log-Likelihood':<25} {gmm_ll:>12.4f} {'N/A':>12} {'N/A':>14}\n")

        # Silhouette
        f.write(f"{'Silhouette Score':<25} "
                f"{metrics['GMM']['silhouette']:>12.4f} "
                f"{metrics['K-Means']['silhouette']:>12.4f} "
                f"{metrics['Hierarchical']['silhouette']:>14.4f}\n")

        # Separation
        f.write(f"{'Cluster Separation':<25} "
                f"{metrics['GMM']['separation']:>12.4f} "
                f"{metrics['K-Means']['separation']:>12.4f} "
                f"{metrics['Hierarchical']['separation']:>14.4f}\n")

        f.write("-" * 65 + "\n")

        # KNN
        f.write(f"\nKNN Consistency Score: {metrics['KNN']['consistency']:.4f}\n")

        # Analysis
        f.write("\n" + "=" * 65 + "\n")
        f.write("ANALYSIS\n")
        f.write("=" * 65 + "\n\n")

        f.write("1. WHY GMM OUTPERFORMS K-MEANS ON PROBABILISTIC CLUSTERS:\n")
        f.write("   - GMM models full covariance (elliptical clusters), while\n")
        f.write("     K-Means assumes spherical clusters (equal variance).\n")
        f.write("   - Iris clusters are elongated and correlated,\n")
        f.write("     favoring GMM's covariance modeling.\n")
        f.write("   - GMM provides soft assignments (probabilities), giving\n")
        f.write("     nuanced membership for boundary points.\n\n")

        f.write("2. WHY KNN IS NOT SUITABLE FOR UNSUPERVISED STRUCTURE:\n")
        f.write("   - KNN is a supervised classifier — it needs labels.\n")
        f.write("   - We generated pseudo-labels from K-Means, so KNN merely\n")
        f.write("     validates K-Means consistency, not independent structure.\n")
        f.write("   - KNN cannot discover clusters, only classify points\n")
        f.write("     given an existing labeling scheme.\n\n")

        f.write("3. WHY HIERARCHICAL CLUSTERING BEHAVES DIFFERENTLY:\n")
        f.write("   - Agglomerative clustering uses distance-based merging\n")
        f.write("     without distributional assumptions.\n")
        f.write("   - Average linkage may create differently shaped clusters\n")
        f.write("     than the elliptical Gaussians that generated the data.\n")
        f.write("   - No probabilistic interpretation of assignments.\n\n")

        f.write("4. EFFECT OF COVARIANCE MODELING IN GMM:\n")
        f.write("   - Full covariance captures the correlation between\n")
        f.write("     features within each cluster.\n")
        f.write("   - This respects the natural variations in flower dimensions.\n")
        f.write("   - Without covariance modeling, boundary points would be\n")
        f.write("     misassigned, reducing clustering quality.\n")

    print(f"\n  Metrics report saved: {filepath}")


def run_comparison(X, gmm_labels, gmm_params,
                   kmeans_labels, hierarchical_labels,
                   knn_consistency):
    """
    Run full model comparison: metrics + report + plots.

    Args:
        X: Data array.
        gmm_labels: GMM cluster assignments.
        gmm_params: Trained GMM parameters.
        kmeans_labels: K-Means cluster assignments.
        hierarchical_labels: Hierarchical cluster assignments.
        knn_consistency: KNN consistency score.

    Returns:
        dict: All computed metrics.
    """
    # Compute metrics
    metrics = compute_all_metrics(
        X, gmm_labels, gmm_params,
        kmeans_labels, hierarchical_labels,
        knn_consistency
    )

    # Write report
    write_metrics_report(metrics, os.path.join(LOGS_DIR, "metrics.txt"))

    # Comparison plot
    print("\n[Comparison Plot]")
    results_dict = {
        'GMM': gmm_labels,
        'K-Means': kmeans_labels,
        'Hierarchical': hierarchical_labels,
    }
    plot_comparison(X, results_dict, os.path.join(PLOTS_DIR, "comparison.png"))

    return metrics


if __name__ == "__main__":
    print("Run via run_experiments.py for full pipeline.")
