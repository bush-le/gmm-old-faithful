"""
metrics.py — GMM-specific evaluation metrics implemented from scratch.

All metrics computed manually without scikit-learn or scipy.

Metrics:
1. Log-Likelihood: How well the GMM fits the data (higher = better)
2. BIC (Bayesian Information Criterion): Model selection with complexity penalty (lower = better)
3. AIC (Akaike Information Criterion): Alternative model selection criterion (lower = better)
4. Silhouette Score: Cluster cohesion vs separation (-1 to 1, higher = better)
5. Cluster Separation: Mean distance between cluster centroids

⚠️ Pure numpy only.
"""
import numpy as np


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two points.
    
    Args:
        a (numpy.ndarray): Point of shape (D,).
        b (numpy.ndarray): Point of shape (D,).
        
    Returns:
        float: Euclidean distance.
    """
    diff = a - b
    return np.sqrt(np.sum(diff * diff))


def compute_bic(log_likelihood, K, n_samples, n_features):
    """
    Compute Bayesian Information Criterion for GMM.
    
    BIC = -2 * log_likelihood + n_params * log(N)
    
    Lower BIC = better model (balances fit vs complexity).
    
    GMM parameters:
    - K * D means
    - K * D*(D+1)/2 covariance entries (symmetric matrix)
    - K - 1 mixing weights (sum-to-1 constraint)
    
    Args:
        log_likelihood (float): Total log-likelihood.
        K (int): Number of components.
        n_samples (int): Number of data points.
        n_features (int): Number of features.
    
    Returns:
        float: BIC value.
    """
    n_params = (K * n_features +                        # means
                K * n_features * (n_features + 1) / 2 + # covariances
                (K - 1))                                 # weights
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic


def compute_aic(log_likelihood, K, n_features):
    """
    Compute Akaike Information Criterion for GMM.
    
    AIC = -2 * log_likelihood + 2 * n_params
    
    Lower AIC = better model. Less penalty than BIC for small datasets.
    
    Args:
        log_likelihood (float): Total log-likelihood.
        K (int): Number of components.
        n_features (int): Number of features.
    
    Returns:
        float: AIC value.
    """
    n_params = (K * n_features +
                K * n_features * (n_features + 1) / 2 +
                (K - 1))
    aic = -2 * log_likelihood + 2 * n_params
    return aic


def compute_silhouette_sample(X, labels, point_idx):
    """
    Compute silhouette score for a single data point.
    
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    Where:
    - a(i) = mean distance to other points in the SAME cluster (cohesion)
    - b(i) = min mean distance to points in ANOTHER cluster (separation)
    
    Interpretation:
    - s ≈ 1: point is well-matched to its cluster
    - s ≈ 0: point is on the boundary between clusters
    - s ≈ -1: point is likely misassigned
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        labels (numpy.ndarray): Cluster labels of shape (N,).
        point_idx (int): Index of the point to evaluate.
        
    Returns:
        float: Silhouette score for this point.
    """
    point_label = labels[point_idx]
    unique_labels = np.unique(labels)
    
    # a(i): mean distance to same-cluster points
    same_cluster = X[labels == point_label]
    if len(same_cluster) <= 1:
        return 0.0  # Single-point cluster
    
    a_i = 0.0
    count_same = 0
    for j in range(len(X)):
        if j != point_idx and labels[j] == point_label:
            a_i += euclidean_distance(X[point_idx], X[j])
            count_same += 1
    a_i /= count_same
    
    # b(i): minimum mean distance to any OTHER cluster
    b_i = np.inf
    for label in unique_labels:
        if label == point_label:
            continue
        
        other_points = X[labels == label]
        mean_dist = 0.0
        for j in range(len(X)):
            if labels[j] == label:
                mean_dist += euclidean_distance(X[point_idx], X[j])
        mean_dist /= len(other_points)
        
        if mean_dist < b_i:
            b_i = mean_dist
    
    # Silhouette score
    max_ab = max(a_i, b_i)
    if max_ab == 0:
        return 0.0
    
    return (b_i - a_i) / max_ab


def silhouette_score(X, labels):
    """
    Compute mean silhouette score across all data points.
    
    Range: [-1, 1]
    - Near 1: Dense, well-separated clusters
    - Near 0: Overlapping clusters  
    - Negative: Possible misassignment
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        labels (numpy.ndarray): Cluster labels of shape (N,).
        
    Returns:
        float: Mean silhouette score.
    """
    n_samples = X.shape[0]
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        scores[i] = compute_silhouette_sample(X, labels, i)
    
    mean_score = np.mean(scores)
    return mean_score


def cluster_separation(X, labels):
    """
    Compute mean inter-cluster centroid distance.
    
    Measures how far apart the cluster centers are.
    Higher = more separated clusters.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        labels (numpy.ndarray): Cluster labels of shape (N,).
        
    Returns:
        float: Mean distance between all pairs of centroids.
    """
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    
    # Compute centroids
    centroids = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    
    # Mean pairwise distance between centroids
    total_dist = 0.0
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            total_dist += euclidean_distance(centroids[i], centroids[j])
            n_pairs += 1
    
    if n_pairs == 0:
        return 0.0
    
    return total_dist / n_pairs


def compute_all_gmm_metrics(X, params, labels, log_likelihood):
    """
    Compute all GMM-specific evaluation metrics.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        params: GMMParams object.
        labels (numpy.ndarray): Hard cluster assignments.
        log_likelihood (float): Final log-likelihood.
    
    Returns:
        dict: All computed metrics.
    """
    n_samples, n_features = X.shape
    K = params.K
    
    bic = compute_bic(log_likelihood, K, n_samples, n_features)
    aic = compute_aic(log_likelihood, K, n_features)
    sil = silhouette_score(X, labels)
    sep = cluster_separation(X, labels)
    
    metrics = {
        'log_likelihood': log_likelihood,
        'bic': bic,
        'aic': aic,
        'silhouette': sil,
        'separation': sep,
        'K': K,
        'n_samples': n_samples,
        'n_features': n_features,
    }
    
    return metrics


def write_metrics_report(metrics, params, filepath):
    """
    Write GMM metrics to a structured text report.
    
    Args:
        metrics (dict): Computed metrics.
        params: GMMParams object.
        filepath (str): Output file path.
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GMM EVALUATION METRICS — OLD FAITHFUL GEYSER\n")
        f.write(f"Dataset: {metrics['n_samples']} samples, {metrics['n_features']} features\n")
        f.write(f"Number of components: K={metrics['K']}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"{'Metric':<25} {'Value':>15}\n")
        f.write("-" * 42 + "\n")
        f.write(f"{'Log-Likelihood':<25} {metrics['log_likelihood']:>15.4f}\n")
        f.write(f"{'BIC':<25} {metrics['bic']:>15.4f}\n")
        f.write(f"{'AIC':<25} {metrics['aic']:>15.4f}\n")
        f.write(f"{'Silhouette Score':<25} {metrics['silhouette']:>15.4f}\n")
        f.write(f"{'Cluster Separation':<25} {metrics['separation']:>15.4f}\n")
        f.write("-" * 42 + "\n\n")
        
        # Learned parameters
        f.write("LEARNED GMM PARAMETERS\n")
        f.write("=" * 60 + "\n")
        f.write(str(params) + "\n\n")
        
        # Cluster sizes
        f.write("CLUSTER SIZES\n")
        f.write("-" * 30 + "\n")
        for k in range(params.K):
            f.write(f"  Component {k+1}: weight π={params.weights[k]:.4f}\n")
        f.write("\n")
        
        # Interpretation
        f.write("METRIC INTERPRETATION\n")
        f.write("=" * 60 + "\n")
        f.write("Log-Likelihood: Higher = better fit. GMM maximizes this via EM.\n")
        f.write("BIC: Lower = better. Penalizes model complexity (n_params * log N).\n")
        f.write("AIC: Lower = better. Less penalty than BIC for small datasets.\n")
        f.write("Silhouette: Range [-1, 1]. Near 1 = well-separated clusters.\n")
        f.write("Cluster Separation: Higher = more distinct cluster centers.\n")
    
    print(f"  Metrics report saved: {filepath}")
