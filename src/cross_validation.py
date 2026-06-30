"""
cross_validation.py — K-Fold Cross-Validation for GMM model selection (§17).

Implements K-Fold CV from scratch to evaluate GMM stability and generalization
performance using ONLY the training set. The test set is NEVER touched during
cross-validation (§17.2).

Key design decisions (per ML_PIPELINE_REFERENCE.md §17):
    - Scaler (mean/std) is fit on fold-train ONLY, then applied to fold-val,
      preventing intra-CV data leakage (§17.3).
    - Each fold uses a different random seed (seed + fold_i) for GMM
      initialization to assess robustness across random starts.
    - Stability is classified per §17.4:
        σ < 0.02  → stable
        0.02 ≤ σ < 0.05 → moderate variability
        σ ≥ 0.05 → unstable

Metrics per fold:
    - Average log-likelihood on fold-validation set
    - Silhouette score on fold-validation set

⚠️ Pure numpy only — no scikit-learn, scipy, or any pre-built ML library.
"""
import os
import numpy as np
from datetime import datetime

from src.em import fit_gmm, compute_log_likelihood
from src.metrics import silhouette_score


# ═══════════════════════════════════════════════════════════════════════════
# §17.1 — K-Fold Index Generation
# ═══════════════════════════════════════════════════════════════════════════

def k_fold_indices(n_samples, n_folds=5, seed=42):
    """
    Generate K-Fold cross-validation indices from scratch.

    Splits n_samples indices into n_folds approximately equal-sized folds.
    Remainder samples (n_samples % n_folds) are distributed one per fold
    across the first folds, ensuring maximum balance.

    Algorithm (§17.1):
        1. Shuffle indices using a seeded RNG for reproducibility.
        2. Compute base fold size = n_samples // n_folds.
        3. Distribute remainder = n_samples % n_folds across first folds.
        4. Slice shuffled indices into fold arrays.

    Args:
        n_samples (int): Total number of samples to split.
        n_folds (int): Number of folds (default: 5). Must satisfy
            2 ≤ n_folds ≤ n_samples.
        seed (int): Random seed for reproducible shuffling (default: 42).

    Returns:
        list[numpy.ndarray]: List of n_folds arrays, each containing
            the original indices assigned to that fold.

    Raises:
        ValueError: If n_folds < 2 or n_folds > n_samples.

    Example:
        >>> folds = k_fold_indices(10, n_folds=3, seed=0)
        >>> [len(f) for f in folds]
        [4, 3, 3]
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be ≥ 2, got {n_folds}")
    if n_folds > n_samples:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed n_samples ({n_samples})"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    remainder = n_samples % n_folds
    # Distribute remainder: first `remainder` folds get +1 sample
    fold_sizes[:remainder] += 1

    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current: current + size])
        current += size

    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Helper — Numerically Safe Gaussian PDF for Cluster Assignments
# ═══════════════════════════════════════════════════════════════════════════

def _gaussian_pdf_batch_safe(X, mean, cov):
    """
    Compute multivariate Gaussian PDF for a batch of points with numerical
    safety guards for cross-validation fold subsets.

    Identical in formula to src.gaussian.gaussian_pdf_batch, but adds
    extra protection against degenerate covariance matrices that can arise
    when fitting GMM on smaller fold-train subsets:
        - Clamps determinant floor to 1e-300.
        - Clamps Mahalanobis exponent to prevent exp() overflow/underflow.
        - Falls back to regularized identity if inversion fails.

    Formula:
        N(x|μ, Σ) = (2π)^(-D/2) · |Σ|^(-1/2) · exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))

    Args:
        X (numpy.ndarray): Data matrix of shape (N, D).
        mean (numpy.ndarray): Mean vector of shape (D,).
        cov (numpy.ndarray): Covariance matrix of shape (D, D).

    Returns:
        numpy.ndarray: PDF values of shape (N,), all ≥ 0.
    """
    n_samples = X.shape[0]
    d = len(mean)

    det = np.linalg.det(cov)
    if det <= 0:
        det = 1e-300

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Fallback: regularized pseudo-inverse via diagonal loading
        cov_reg = cov + 1e-4 * np.eye(d)
        cov_inv = np.linalg.inv(cov_reg)
        det = np.linalg.det(cov_reg)
        if det <= 0:
            det = 1e-300

    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * det))

    diff = X - mean  # (N, D)
    mahalanobis_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)

    # Clamp exponent to [-700, 0] to prevent exp() underflow/overflow
    exponent = -0.5 * mahalanobis_sq
    exponent = np.clip(exponent, -700.0, 0.0)

    pdf_values = norm_const * np.exp(exponent)
    return pdf_values


# ═══════════════════════════════════════════════════════════════════════════
# §17.2–17.3 — Cross-Validate GMM
# ═══════════════════════════════════════════════════════════════════════════

def cross_validate_gmm(X_train_raw, K, n_folds, max_iters, tol, reg_covar,
                        init_method='kmeans', seed=42):
    """
    Perform K-Fold cross-validation for a GMM with K components.

    Operates ONLY within the training set (§17.2). For each fold:
        1. Split X_train_raw into fold-train and fold-val.
        2. Fit a z-score scaler (mean, std) on fold-train ONLY (§17.3).
        3. Transform both fold-train and fold-val using fold-train stats.
        4. Fit GMM on standardized fold-train via EM.
        5. Evaluate average log-likelihood on fold-val.
        6. Compute silhouette score on fold-val using hard cluster assignments.

    Each fold uses seed = base_seed + fold_index for GMM initialization,
    probing robustness across different random starts.

    Args:
        X_train_raw (numpy.ndarray): Raw (un-scaled) training data of shape
            (N_train, D). Must NOT be pre-standardized.
        K (int): Number of GMM components.
        n_folds (int): Number of CV folds (typically 5).
        max_iters (int): Maximum EM iterations per fold.
        tol (float): EM convergence tolerance (|Δ log-likelihood|).
        reg_covar (float): Covariance regularization term (ε·I).
        init_method (str): GMM initialization method ('kmeans' or 'random').
            Default: 'kmeans'.
        seed (int): Base random seed. Fold i uses seed + i. Default: 42.

    Returns:
        dict: Cross-validation results with keys:
            - 'll_mean' (float): Mean of per-fold average log-likelihoods.
            - 'll_std' (float): Std dev of per-fold average log-likelihoods.
            - 'll_scores' (list[float]): Per-fold average log-likelihoods.
            - 'sil_mean' (float): Mean of per-fold silhouette scores.
            - 'sil_std' (float): Std dev of per-fold silhouette scores.
            - 'sil_scores' (list[float]): Per-fold silhouette scores.

    Notes:
        - The scaler is fit per fold to prevent intra-CV data leakage (§17.3).
        - Silhouette is computed on the scaled fold-val data using hard
          assignments derived from the GMM's mixture densities.
        - If a fold produces a degenerate GMM (all points assigned to one
          cluster), silhouette defaults to 0.0 for that fold.
    """
    n_samples = X_train_raw.shape[0]
    folds = k_fold_indices(n_samples, n_folds=n_folds, seed=seed)

    ll_scores = []
    sil_scores = []

    print(f"\n{'='*60}")
    print(f"  K-FOLD CROSS-VALIDATION  (K_components={K}, n_folds={n_folds})")
    print(f"{'='*60}")

    for fold_i, val_indices in enumerate(folds):
        fold_seed = seed + fold_i

        # ── 1. Split into fold-train / fold-val ──
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_indices] = False
        train_indices = np.where(train_mask)[0]

        X_fold_train_raw = X_train_raw[train_indices]
        X_fold_val_raw = X_train_raw[val_indices]

        n_train = X_fold_train_raw.shape[0]
        n_val = X_fold_val_raw.shape[0]

        # ── 2. Fit scaler on fold-train ONLY (§17.3: prevent leakage) ──
        fold_mean = np.mean(X_fold_train_raw, axis=0)
        fold_std = np.std(X_fold_train_raw, axis=0)
        # Prevent division by zero for constant features
        fold_std = np.where(fold_std < 1e-10, 1.0, fold_std)

        # ── 3. Transform both splits using fold-train statistics ──
        X_fold_train = (X_fold_train_raw - fold_mean) / fold_std
        X_fold_val = (X_fold_val_raw - fold_mean) / fold_std

        # ── 4. Fit GMM on fold-train ──
        print(f"\n  Fold {fold_i + 1}/{n_folds}  "
              f"(train={n_train}, val={n_val}, seed={fold_seed})")
        params, responsibilities, ll_history, n_iters = fit_gmm(
            X_fold_train, K, max_iters, tol, reg_covar,
            init_method=init_method, seed=fold_seed
        )

        # ── 5. Evaluate log-likelihood on fold-val ──
        val_ll = compute_log_likelihood(X_fold_val, params)
        avg_val_ll = val_ll / n_val
        ll_scores.append(avg_val_ll)

        # ── 6. Compute silhouette on fold-val ──
        #    Hard assignments via argmax of mixture component densities
        val_labels = _assign_clusters(X_fold_val, params)
        n_unique = len(np.unique(val_labels))

        if n_unique < 2:
            # Degenerate case: all points assigned to one cluster
            sil = 0.0
            print(f"    ⚠ All val points assigned to 1 cluster → silhouette=0.0")
        else:
            sil = silhouette_score(X_fold_val, val_labels)
        sil_scores.append(sil)

        print(f"    Fold {fold_i + 1} results: "
              f"avg_LL={avg_val_ll:.4f}, silhouette={sil:.4f}")

    # ── Aggregate across folds ──
    ll_scores_arr = np.array(ll_scores)
    sil_scores_arr = np.array(sil_scores)

    cv_results = {
        'll_mean': float(np.mean(ll_scores_arr)),
        'll_std': float(np.std(ll_scores_arr)),
        'll_scores': [float(s) for s in ll_scores],
        'sil_mean': float(np.mean(sil_scores_arr)),
        'sil_std': float(np.std(sil_scores_arr)),
        'sil_scores': [float(s) for s in sil_scores],
    }

    print(f"\n  CV Summary: avg_LL = {cv_results['ll_mean']:.4f} "
          f"± {cv_results['ll_std']:.4f}")
    print(f"              silhouette = {cv_results['sil_mean']:.4f} "
          f"± {cv_results['sil_std']:.4f}")

    return cv_results


def _assign_clusters(X, params):
    """
    Assign hard cluster labels using GMM component densities.

    For each point x_n, compute π_k · N(x_n | μ_k, Σ_k) for all k,
    then assign to the component with the highest weighted density.
    Uses _gaussian_pdf_batch_safe for numerical robustness.

    Args:
        X (numpy.ndarray): Data of shape (N, D).
        params: GMMParams with weights, means, covariances, K.

    Returns:
        numpy.ndarray: Hard cluster labels of shape (N,), values in [0, K-1].
    """
    n_samples = X.shape[0]
    K = params.K

    weighted_pdfs = np.zeros((n_samples, K))
    for k in range(K):
        pdf_vals = _gaussian_pdf_batch_safe(X, params.means[k],
                                             params.covariances[k])
        weighted_pdfs[:, k] = params.weights[k] * pdf_vals

    labels = np.argmax(weighted_pdfs, axis=1)
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# §17.4 — Logging & Results Persistence
# ═══════════════════════════════════════════════════════════════════════════

def log_cv_results(cv_results, K, n_folds, logs_dir, metrics_dir):
    """
    Log cross-validation results to structured text files.

    Produces two output files:
        1. logs_dir/16_cross_validation.txt — Detailed per-fold table,
           summary statistics, and stability interpretation.
        2. metrics_dir/cv_results_mu_sigma.txt — Compact μ±σ summary
           for downstream consumption.

    Stability interpretation (§17.4):
        - σ < 0.02:          Stable — consistent generalization.
        - 0.02 ≤ σ < 0.05:  Moderate variability — acceptable but worth noting.
        - σ ≥ 0.05:          Unstable — high variance across folds, investigate.

    Args:
        cv_results (dict): Output from cross_validate_gmm() containing
            'll_mean', 'll_std', 'll_scores', 'sil_mean', 'sil_std',
            'sil_scores'.
        K (int): Number of GMM components evaluated.
        n_folds (int): Number of CV folds used.
        logs_dir (str): Directory for the detailed log file.
        metrics_dir (str): Directory for the compact metrics file.

    Side effects:
        Creates/overwrites two files. Directories are created if needed.
    """
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, "16_cross_validation.txt")
    metrics_path = os.path.join(metrics_dir, "cv_results_mu_sigma.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Stability interpretation (§17.4) ──
    ll_stability = _interpret_stability(cv_results['ll_std'])
    sil_stability = _interpret_stability(cv_results['sil_std'])

    # ═══════════════════════════════════════════════════════
    # 1. Detailed log file
    # ═══════════════════════════════════════════════════════
    with open(log_path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("STEP 16 — K-FOLD CROSS-VALIDATION RESULTS (§17)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"GMM Components: K={K}   |   Folds: {n_folds}\n")
        f.write("=" * 65 + "\n\n")

        # ── Per-fold table ──
        f.write("PER-FOLD RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Fold':<8} {'Avg Log-Lik':>14} {'Silhouette':>14}\n")
        f.write("-" * 50 + "\n")
        for i in range(n_folds):
            ll_val = cv_results['ll_scores'][i]
            sil_val = cv_results['sil_scores'][i]
            f.write(f"  {i + 1:<6} {ll_val:>14.6f} {sil_val:>14.6f}\n")
        f.write("-" * 50 + "\n\n")

        # ── μ ± σ summary ──
        f.write("SUMMARY (μ ± σ)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Avg Log-Likelihood:  {cv_results['ll_mean']:.6f} "
                f"± {cv_results['ll_std']:.6f}\n")
        f.write(f"  Silhouette Score:    {cv_results['sil_mean']:.6f} "
                f"± {cv_results['sil_std']:.6f}\n")
        f.write("-" * 50 + "\n\n")

        # ── Stability interpretation (§17.4) ──
        f.write("STABILITY INTERPRETATION (§17.4)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Log-Likelihood σ = {cv_results['ll_std']:.6f} "
                f"→ {ll_stability}\n")
        f.write(f"  Silhouette σ     = {cv_results['sil_std']:.6f} "
                f"→ {sil_stability}\n")
        f.write("-" * 50 + "\n\n")

        f.write("INTERPRETATION GUIDE (§17.4)\n")
        f.write("-" * 50 + "\n")
        f.write("  σ < 0.02          → Stable: consistent generalization\n")
        f.write("  0.02 ≤ σ < 0.05   → Moderate: acceptable variability\n")
        f.write("  σ ≥ 0.05          → Unstable: high fold-to-fold variance\n")
        f.write("-" * 50 + "\n\n")

        f.write("NOTE: Cross-validation operates ONLY within the training\n")
        f.write("set (§17.2). The held-out test set is never touched.\n")
        f.write("=" * 65 + "\n")

    print(f"\n  CV log saved: {log_path}")

    # ═══════════════════════════════════════════════════════
    # 2. Compact metrics file
    # ═══════════════════════════════════════════════════════
    with open(metrics_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("CROSS-VALIDATION RESULTS — μ ± σ\n")
        f.write(f"K={K}, n_folds={n_folds}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"avg_log_likelihood:  {cv_results['ll_mean']:.6f} "
                f"± {cv_results['ll_std']:.6f}\n")
        f.write(f"silhouette_score:    {cv_results['sil_mean']:.6f} "
                f"± {cv_results['sil_std']:.6f}\n\n")
        f.write(f"ll_stability:   {ll_stability}\n")
        f.write(f"sil_stability:  {sil_stability}\n")
        f.write("=" * 50 + "\n")

    print(f"  CV metrics saved: {metrics_path}")


def _interpret_stability(sigma):
    """
    Classify fold-to-fold standard deviation into a stability category.

    Thresholds per ML_PIPELINE_REFERENCE.md §17.4:
        σ < 0.02         → "Stable — consistent generalization across folds"
        0.02 ≤ σ < 0.05  → "Moderate variability — acceptable but noteworthy"
        σ ≥ 0.05         → "Unstable — high variance, investigate further"

    Args:
        sigma (float): Standard deviation of a metric across CV folds.

    Returns:
        str: Human-readable stability interpretation.
    """
    if sigma < 0.02:
        return "Stable — consistent generalization across folds"
    elif sigma < 0.05:
        return "Moderate variability — acceptable but noteworthy"
    else:
        return "Unstable — high variance, investigate further"
