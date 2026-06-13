"""
run_baselines.py — Driver script for baseline model experiments.

Runs K-Means, Hierarchical Clustering, and KNN consistency check
on the preprocessed Iris dataset.
"""
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import K, RANDOM_SEED, K_NEIGHBORS, KMEANS_MAX_ITERS, PROCESSED_DATA_PATH, PLOTS_DIR
from src.data_loader import load_csv
from src.kmeans import fit_kmeans
from src.hierarchical import fit_agglomerative
from src.knn import evaluate_consistency
from src.visualization import plot_clusters


def run_kmeans_experiment(X):
    """
    Run K-Means clustering experiment.

    Args:
        X (numpy.ndarray): Data of shape (N, 2).

    Returns:
        tuple: (labels, centroids)
    """
    print("\n" + "-" * 40)
    print("K-MEANS CLUSTERING")
    print("-" * 40)

    labels, centroids = fit_kmeans(X, K, KMEANS_MAX_ITERS, RANDOM_SEED)

    # Print cluster sizes
    for k in range(K):
        n_k = np.sum(labels == k)
        print(f"    Cluster {k+1}: {n_k} points, centroid = {centroids[k]}")

    # Save plot
    plot_clusters(
        X, labels, "K-Means Clustering (K=3)",
        os.path.join(PLOTS_DIR, "kmeans_result.png"),
        centroids=centroids
    )

    return labels, centroids


def run_hierarchical_experiment(X):
    """
    Run Agglomerative Hierarchical Clustering experiment.

    Args:
        X (numpy.ndarray): Data of shape (N, 2).

    Returns:
        numpy.ndarray: Cluster labels.
    """
    print("\n" + "-" * 40)
    print("HIERARCHICAL CLUSTERING (Average Linkage)")
    print("-" * 40)

    labels = fit_agglomerative(X, K, linkage="average")

    # Print cluster sizes
    for k in range(K):
        n_k = np.sum(labels == k)
        print(f"    Cluster {k+1}: {n_k} points")

    # Save plot
    plot_clusters(
        X, labels, "Agglomerative Hierarchical Clustering (K=3)",
        os.path.join(PLOTS_DIR, "hierarchical_result.png")
    )

    return labels


def run_knn_experiment(X, kmeans_labels):
    """
    Run KNN consistency check using K-Means pseudo-labels.

    Args:
        X (numpy.ndarray): Data of shape (N, 2).
        kmeans_labels (numpy.ndarray): Labels from K-Means.

    Returns:
        float: Consistency score.
    """
    print("\n" + "-" * 40)
    print("KNN CONSISTENCY CHECK")
    print("-" * 40)

    consistency = evaluate_consistency(X, kmeans_labels, K_NEIGHBORS)
    return consistency


def run_all_baselines():
    """
    Run all baseline experiments.

    Returns:
        dict: Results from each baseline.
    """
    print("\n" + "=" * 60)
    print("BASELINE EXPERIMENTS")
    print("=" * 60)

    # Load data
    print("\nLoading processed data...")
    raw_data = load_csv(PROCESSED_DATA_PATH)
    X = np.array(raw_data)
    print(f"  Data shape: {X.shape}")

    # Run baselines
    kmeans_labels, kmeans_centroids = run_kmeans_experiment(X)
    hierarchical_labels = run_hierarchical_experiment(X)
    knn_consistency = run_knn_experiment(X, kmeans_labels)

    results = {
        'kmeans_labels': kmeans_labels,
        'kmeans_centroids': kmeans_centroids,
        'hierarchical_labels': hierarchical_labels,
        'knn_consistency': knn_consistency,
    }

    print("\n[DONE] All baselines complete.")
    return results


if __name__ == "__main__":
    run_all_baselines()
