"""
baseline.py — Baseline models for comparison against GMM.

Implements ML_PIPELINE_REFERENCE.md §11 (Baseline Comparison):
  1. Single Gaussian (K=1): Simplest generative baseline. Fits a single
     multivariate Gaussian to all data. Expected to underfit Old Faithful's
     bimodal structure, establishing a lower bound on performance.
  2. K-Means: Hard-assignment clustering baseline. Uses the same K as the GMM
     for a fair comparison. K-Means assumes spherical clusters with equal
     variance — a restrictive assumption for Old Faithful's elliptical clusters.

Purpose:
  - Quantify how much the GMM improves over simpler models
  - Validate that K=2 GMM is justified (vs K=1 single Gaussian)
  - Compare soft (GMM) vs hard (K-Means) assignment quality

⚠️ Pure numpy only. No pandas, no scipy, no sklearn.
"""
import numpy as np
import os


def fit_single_gaussian(X):
    """
    Fit a single multivariate Gaussian (K=1) to the data.

    This is the simplest possible generative model: assume ALL data comes
    from one Gaussian distribution. The MLE solution is:
      mean = (1/N) * sum(x_i)
      cov  = (1/N) * sum((x_i - mean)(x_i - mean)^T)

    For Old Faithful, this baseline is expected to perform poorly because
    the data is clearly bimodal. The single Gaussian will be centered between
    the two clusters, assigning low probability to points in both modes.

    Reference: ML_PIPELINE_REFERENCE.md §11

    Args:
        X (numpy.ndarray): Data of shape (N, D).

    Returns:
        tuple: (mean, cov, log_likelihood) where:
            - mean (numpy.ndarray): MLE mean vector of shape (D,).
            - cov (numpy.ndarray): MLE covariance matrix of shape (D, D).
            - log_likelihood (float): Total log-likelihood of data under
              the fitted Gaussian: sum_i log N(x_i | mean, cov).
    """
    n_samples, n_features = X.shape

    # MLE mean: sample mean
    mean = np.zeros(n_features)
    for i in range(n_samples):
        mean += X[i]
    mean /= n_samples

    # MLE covariance: sample covariance (biased estimator, consistent with GMM)
    cov = np.zeros((n_features, n_features))
    for i in range(n_samples):
        diff = (X[i] - mean).reshape(-1, 1)  # Column vector (D, 1)
        cov += np.dot(diff, diff.T)           # Outer product (D, D)
    cov /= n_samples

    # Add small regularization to prevent singularity
    cov += 1e-6 * np.eye(n_features)

    # Compute log-likelihood: sum_i log N(x_i | mean, cov)
    log_likelihood = _compute_gaussian_log_likelihood(X, mean, cov)

    return mean, cov, log_likelihood


def _compute_gaussian_log_likelihood(X, mean, cov):
    """
    Compute total log-likelihood of data under a multivariate Gaussian.

    log p(X | mean, cov) = sum_i log N(x_i | mean, cov)

    where log N(x | mean, cov) = -D/2 * log(2π) - 1/2 * log|Σ| 
                                  - 1/2 * (x - μ)^T Σ^{-1} (x - μ)

    Uses vectorized computation for efficiency.

    Args:
        X (numpy.ndarray): Data of shape (N, D).
        mean (numpy.ndarray): Mean vector of shape (D,).
        cov (numpy.ndarray): Covariance matrix of shape (D, D).

    Returns:
        float: Total log-likelihood (sum over all N samples).
    """
    n_samples, n_features = X.shape

    # Log-determinant (more numerically stable than log(det))
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        log_det = -700.0  # Fallback for degenerate covariance

    # Precision matrix (inverse covariance)
    cov_inv = np.linalg.inv(cov)

    # Constant term: -D/2 * log(2π)
    const = -0.5 * n_features * np.log(2 * np.pi)

    # Vectorized Mahalanobis distance for all points
    diff = X - mean  # (N, D)
    mahalanobis_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)  # (N,)

    # Log-likelihood per sample: const - 0.5*log|Σ| - 0.5*mahalanobis²
    log_probs = const - 0.5 * log_det - 0.5 * mahalanobis_sq  # (N,)

    # Total log-likelihood
    total_ll = np.sum(log_probs)

    return total_ll


def run_baselines(X_train, X_test, K, seed, logs_dir):
    """
    Run all baseline models and compare their performance.

    Baselines implemented (ref ML_PIPELINE_REFERENCE.md §11):
      1. Single Gaussian (K=1): Generative baseline, no clustering
      2. K-Means (K=K): Hard-assignment clustering baseline

    Each baseline is evaluated on both train and test sets using:
      - Log-likelihood (generative quality, for single Gaussian)
      - Silhouette score (cluster cohesion vs separation)
      - Cluster separation (inter-centroid distance)

    Outputs:
      - Comparison table logged to {logs_dir}/11_baseline.txt

    Args:
        X_train (numpy.ndarray): Training data of shape (N_train, D).
        X_test (numpy.ndarray): Test data of shape (N_test, D).
        K (int): Number of clusters for K-Means (should match GMM's K).
        seed (int): Random seed for reproducibility.
        logs_dir (str): Directory for log files.

    Returns:
        dict: Results dictionary with keys:
            - 'single_gaussian': {
                'mean', 'cov', 'train_ll', 'test_ll'
              }
            - 'kmeans': {
                'train_labels', 'train_centroids',
                'test_labels', 'test_centroids',
                'train_silhouette', 'test_silhouette',
                'train_separation', 'test_separation'
              }
    """
    from src.kmeans import fit_kmeans
    from src.metrics import silhouette_score, cluster_separation
    from src.knn import fit_predict_knn
    from src.hierarchical import fit_predict_hierarchical

    os.makedirs(logs_dir, exist_ok=True)

    results = {}

    # ── Baseline 1: Single Gaussian (K=1) ──
    print("\n[Baseline 1] Single Gaussian (K=1)...")
    mean, cov, train_ll = fit_single_gaussian(X_train)
    test_ll = _compute_gaussian_log_likelihood(X_test, mean, cov)

    results['single_gaussian'] = {
        'mean': mean,
        'cov': cov,
        'train_ll': train_ll,
        'test_ll': test_ll,
        'avg_train_ll': train_ll / len(X_train),
        'avg_test_ll': test_ll / len(X_test),
    }

    print(f"  Train log-likelihood: {train_ll:.4f}")
    print(f"  Test  log-likelihood: {test_ll:.4f}")

    # ── Baseline 2: K-Means ──
    print(f"\n[Baseline 2] K-Means (K={K})...")

    # Fit on training data
    train_labels, train_centroids = fit_kmeans(X_train, K, max_iters=100, seed=seed)
    train_sil = silhouette_score(X_train, train_labels)
    train_sep = cluster_separation(X_train, train_labels)

    # Predict on test data: assign each test point to nearest centroid
    test_labels = _assign_to_centroids(X_test, train_centroids)
    test_sil = silhouette_score(X_test, test_labels)
    test_sep = cluster_separation(X_test, test_labels)

    results['kmeans'] = {
        'train_labels': train_labels,
        'train_centroids': train_centroids,
        'test_labels': test_labels,
        'train_silhouette': train_sil,
        'test_silhouette': test_sil,
        'train_separation': train_sep,
        'test_separation': test_sep,
    }

    print(f"  Train silhouette: {train_sil:.4f}, separation: {train_sep:.4f}")
    print(f"  Test  silhouette: {test_sil:.4f}, separation: {test_sep:.4f}")

    # ── Baseline 3: K-Nearest Neighbors (KNN) ──
    print(f"\n[Baseline 3] K-Nearest Neighbors (KNN k=3)...")
    # Pseudo-labels from KMeans
    test_labels_knn = fit_predict_knn(X_train, train_labels, X_test, k=3)
    test_sil_knn = silhouette_score(X_test, test_labels_knn)
    test_sep_knn = cluster_separation(X_test, test_labels_knn)
    
    results['knn'] = {
        'test_labels': test_labels_knn,
        'test_silhouette': test_sil_knn,
        'test_separation': test_sep_knn,
    }
    print(f"  Test  silhouette: {test_sil_knn:.4f}, separation: {test_sep_knn:.4f}")

    # ── Baseline 4: Hierarchical Clustering ──
    print(f"\n[Baseline 4] Hierarchical Clustering (K={K})...")
    # Fit on training data
    train_labels_hc = fit_predict_hierarchical(X_train, n_clusters=K, linkage='average')
    train_sil_hc = silhouette_score(X_train, train_labels_hc)
    train_sep_hc = cluster_separation(X_train, train_labels_hc)
    
    results['hierarchical'] = {
        'train_labels': train_labels_hc,
        'train_silhouette': train_sil_hc,
        'train_separation': train_sep_hc,
    }
    print(f"  Train silhouette: {train_sil_hc:.4f}, separation: {train_sep_hc:.4f}")

    # ── Cluster size summary for K-Means ──
    for split_name, labels in [('Train', train_labels), ('Test', test_labels)]:
        unique, counts = np.unique(labels, return_counts=True)
        sizes = ", ".join(f"C{u}={c}" for u, c in zip(unique, counts))
        print(f"  {split_name} cluster sizes: {sizes}")

    # ── Log comparison table ──
    _log_baseline_comparison(results, K, X_train, X_test, logs_dir)

    print("\n[DONE] Baseline comparison complete.")
    return results


def _assign_to_centroids(X, centroids):
    """
    Assign each data point to the nearest centroid (hard assignment).

    Used to apply K-Means centroids learned on the training set to
    the test set.

    Args:
        X (numpy.ndarray): Data of shape (N, D).
        centroids (numpy.ndarray): Centroids of shape (K, D).

    Returns:
        numpy.ndarray: Cluster labels of shape (N,).
    """
    n_samples = X.shape[0]
    K = centroids.shape[0]
    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        min_dist = np.inf
        for k in range(K):
            diff = X[i] - centroids[k]
            dist = np.sqrt(np.sum(diff * diff))
            if dist < min_dist:
                min_dist = dist
                labels[i] = k

    return labels


def _log_baseline_comparison(results, K, X_train, X_test, logs_dir):
    """
    Write baseline comparison results to a structured log file.

    Creates a formatted comparison table at {logs_dir}/11_baseline.txt
    showing performance of each baseline model on train and test sets.

    Reference: ML_PIPELINE_REFERENCE.md §11

    Args:
        results (dict): Results dictionary from run_baselines.
        K (int): Number of clusters.
        X_train (numpy.ndarray): Training data.
        X_test (numpy.ndarray): Test data.
        logs_dir (str): Directory for log files.
    """
    sg = results['single_gaussian']
    km = results['kmeans']

    log_lines = []
    log_lines.append("BASELINE MODEL COMPARISON")
    log_lines.append("=" * 70)
    log_lines.append(f"Reference: ML_PIPELINE_REFERENCE.md §11")
    log_lines.append(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    log_lines.append(f"Features: {X_train.shape[1]}")
    log_lines.append("")

    # ── Single Gaussian details ──
    log_lines.append("BASELINE 1: Single Gaussian (K=1)")
    log_lines.append("-" * 70)
    log_lines.append(f"  Description:  Fit one multivariate Gaussian to all data (MLE)")
    log_lines.append(f"  Assumption:   Data is unimodal → expected to underfit Old Faithful")
    log_lines.append(f"  Mean:         {sg['mean']}")
    log_lines.append(f"  Covariance:")
    for row in sg['cov']:
        log_lines.append(f"    {row}")
    log_lines.append(f"  Train log-likelihood: {sg['train_ll']:.4f}")
    log_lines.append(f"  Test  log-likelihood: {sg['test_ll']:.4f}")
    log_lines.append("")

    # ── K-Means details ──
    log_lines.append(f"BASELINE 2: K-Means (K={K})")
    log_lines.append("-" * 70)
    log_lines.append(f"  Description:  Hard-assignment clustering with Euclidean distance")
    log_lines.append(f"  Assumption:   Spherical clusters with equal variance")
    log_lines.append(f"  Centroids:")
    for k in range(K):
        log_lines.append(f"    Cluster {k}: {km['train_centroids'][k]}")
    log_lines.append(f"  Train silhouette score:  {km['train_silhouette']:.4f}")
    log_lines.append(f"  Test  silhouette score:  {km['test_silhouette']:.4f}")
    log_lines.append(f"  Train cluster separation: {km['train_separation']:.4f}")
    log_lines.append(f"  Test  cluster separation: {km['test_separation']:.4f}")

    # Cluster sizes
    for split_name, labels in [('Train', km['train_labels']),
                                ('Test', km['test_labels'])]:
        unique, counts = np.unique(labels, return_counts=True)
        sizes = ", ".join(f"C{u}={c}" for u, c in zip(unique, counts))
        log_lines.append(f"  {split_name} cluster sizes: {sizes}")
    log_lines.append("")

    # ── Comparison table ──
    log_lines.append("COMPARISON TABLE")
    log_lines.append("=" * 70)
    header = f"{'Metric':<30} {'Single Gaussian':>18} {'K-Means':>18}"
    log_lines.append(header)
    log_lines.append("-" * 70)

    log_lines.append(
        f"{'Train Log-Likelihood':<30} {sg['train_ll']:>18.4f} {'N/A':>18}"
    )
    log_lines.append(
        f"{'Test Log-Likelihood':<30} {sg['test_ll']:>18.4f} {'N/A':>18}"
    )
    log_lines.append(
        f"{'Train Silhouette':<30} {'N/A':>18} {km['train_silhouette']:>18.4f}"
    )
    log_lines.append(
        f"{'Test Silhouette':<30} {'N/A':>18} {km['test_silhouette']:>18.4f}"
    )
    log_lines.append(
        f"{'Train Cluster Separation':<30} {'N/A':>18} {km['train_separation']:>18.4f}"
    )
    log_lines.append(
        f"{'Test Cluster Separation':<30} {'N/A':>18} {km['test_separation']:>18.4f}"
    )
    log_lines.append("-" * 70)
    log_lines.append("")

    # ── Interpretation ──
    log_lines.append("INTERPRETATION")
    log_lines.append("=" * 70)
    log_lines.append("Single Gaussian (K=1):")
    log_lines.append("  - Expected to underfit: Old Faithful is bimodal, not unimodal.")
    log_lines.append("  - The single Gaussian sits between the two clusters, assigning")
    log_lines.append("    low probability to points in both modes.")
    log_lines.append("  - Low log-likelihood confirms the need for K≥2 components.")
    log_lines.append("")
    log_lines.append("K-Means:")
    log_lines.append("  - Hard assignment: each point belongs to exactly one cluster.")
    log_lines.append("  - Assumes spherical clusters with equal variance.")
    log_lines.append("  - For Old Faithful's elliptical clusters, K-Means may misassign")
    log_lines.append("    boundary points that a GMM would handle probabilistically.")
    log_lines.append("")
    log_lines.append("GMM Advantage (to be compared after GMM fitting):")
    log_lines.append("  - Soft assignment captures uncertainty for boundary points.")
    log_lines.append("  - Full covariance models elliptical cluster shapes.")
    log_lines.append("  - Higher log-likelihood expected due to better density estimation.")

    # Save log
    log_path = os.path.join(logs_dir, "11_baseline.txt")
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Baseline log saved: {log_path}")
