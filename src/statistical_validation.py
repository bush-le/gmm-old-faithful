import numpy as np
import os
import json
from src.em import fit_gmm, compute_log_likelihood
from src.cross_validation import _assign_clusters
from src.metrics import silhouette_score, compute_bic, compute_aic, cluster_separation

def bootstrap_gmm_evaluation(X_train_raw, K, B=50, max_iters=100, tol=1e-3, reg_covar=1e-6, init_method='kmeans', seed=42):
    """
    Perform Bootstrap resampling to estimate the confidence interval of GMM performance.
    
    For each of the B iterations:
      1. Sample N points with replacement from X_train_raw (X_boot).
      2. The points not sampled become the Out-Of-Bag (OOB) set.
      3. Fit scaler on X_boot, scale both X_boot and X_oob.
      4. Fit GMM on scaled X_boot.
      5. Evaluate average log-likelihood and silhouette on scaled X_oob.
      
    Returns a dictionary of bootstrap results.
    """
    np.random.seed(seed)
    N = X_train_raw.shape[0]
    
    oob_ll = []
    oob_sil = []
    oob_bic = []
    oob_aic = []
    oob_sep = []
    
    print(f"\n{'='*60}")
    print(f"  BOOTSTRAP STATISTICAL VALIDATION  (B={B} iterations)")
    print(f"{'='*60}")
    
    for i in range(B):
        # 1. Resample with replacement
        boot_indices = np.random.choice(N, size=N, replace=True)
        oob_mask = np.ones(N, dtype=bool)
        oob_mask[boot_indices] = False
        oob_indices = np.where(oob_mask)[0]
        
        if len(oob_indices) == 0:
            continue
            
        X_boot_raw = X_train_raw[boot_indices]
        X_oob_raw = X_train_raw[oob_indices]
        
        # 2. Scale
        boot_mean = np.mean(X_boot_raw, axis=0)
        boot_std = np.std(X_boot_raw, axis=0)
        boot_std = np.where(boot_std < 1e-10, 1.0, boot_std)
        
        X_boot = (X_boot_raw - boot_mean) / boot_std
        X_oob = (X_oob_raw - boot_mean) / boot_std
        
        # 3. Fit GMM
        params, _, _, _ = fit_gmm(
            X_boot, K, max_iters, tol, reg_covar, 
            init_method=init_method, seed=seed+i
        )
        
        # 4. Evaluate OOB LL & BIC/AIC
        n_samples, n_features = X_oob.shape
        ll = compute_log_likelihood(X_oob, params)
        avg_ll = ll / n_samples
        oob_ll.append(avg_ll)
        
        oob_bic.append(compute_bic(ll, K, n_samples, n_features))
        oob_aic.append(compute_aic(ll, K, n_features))
        
        # 5. Evaluate OOB Silhouette & Separation
        val_labels = _assign_clusters(X_oob, params)
        n_unique = len(np.unique(val_labels))
        if n_unique < 2:
            oob_sil.append(0.0)
            oob_sep.append(0.0)
        else:
            sil = silhouette_score(X_oob, val_labels)
            oob_sil.append(sil)
            sep = cluster_separation(X_oob, val_labels)
            oob_sep.append(sep)
            
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1}/{B} bootstrap iterations...")
            
    # Compute Confidence Intervals (95%)
    ll_lower, ll_upper = np.percentile(oob_ll, [2.5, 97.5])
    sil_lower, sil_upper = np.percentile(oob_sil, [2.5, 97.5])
    bic_lower, bic_upper = np.percentile(oob_bic, [2.5, 97.5])
    aic_lower, aic_upper = np.percentile(oob_aic, [2.5, 97.5])
    sep_lower, sep_upper = np.percentile(oob_sep, [2.5, 97.5])
    
    results = {
        'B': len(oob_ll),
        'll_scores': oob_ll,
        'll_mean': np.mean(oob_ll),
        'll_ci_95': (ll_lower, ll_upper),
        'sil_scores': oob_sil,
        'sil_mean': np.mean(oob_sil),
        'sil_ci_95': (sil_lower, sil_upper),
        'bic_scores': oob_bic,
        'bic_mean': np.mean(oob_bic),
        'bic_ci_95': (bic_lower, bic_upper),
        'aic_scores': oob_aic,
        'aic_mean': np.mean(oob_aic),
        'aic_ci_95': (aic_lower, aic_upper),
        'sep_scores': oob_sep,
        'sep_mean': np.mean(oob_sep),
        'sep_ci_95': (sep_lower, sep_upper)
    }
    
    return results

def log_bootstrap_results(results, K, logs_dir, metrics_dir):
    log_path = os.path.join(logs_dir, "18_bootstrap_validation.txt")
    with open(log_path, "w") as f:
        f.write("STATISTICAL VALIDATION (BOOTSTRAP) RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Number of iterations (B): {results['B']}\n")
        f.write(f"Components (K): {K}\n\n")
        
        f.write("Out-Of-Bag (OOB) Log-Likelihood:\n")
        f.write(f"  Mean: {results['ll_mean']:.4f}\n")
        f.write(f"  95% Confidence Interval: [{results['ll_ci_95'][0]:.4f}, {results['ll_ci_95'][1]:.4f}]\n\n")
        
        f.write("Out-Of-Bag (OOB) Silhouette Score:\n")
        f.write(f"  Mean: {results['sil_mean']:.4f}\n")
        f.write(f"  95% Confidence Interval: [{results['sil_ci_95'][0]:.4f}, {results['sil_ci_95'][1]:.4f}]\n\n")
        
        f.write("Out-Of-Bag (OOB) BIC Score:\n")
        f.write(f"  Mean: {results['bic_mean']:.4f}\n")
        f.write(f"  95% Confidence Interval: [{results['bic_ci_95'][0]:.4f}, {results['bic_ci_95'][1]:.4f}]\n\n")
        
        f.write("Out-Of-Bag (OOB) AIC Score:\n")
        f.write(f"  Mean: {results['aic_mean']:.4f}\n")
        f.write(f"  95% Confidence Interval: [{results['aic_ci_95'][0]:.4f}, {results['aic_ci_95'][1]:.4f}]\n\n")
        
        f.write("Out-Of-Bag (OOB) Cluster Separation:\n")
        f.write(f"  Mean: {results['sep_mean']:.4f}\n")
        f.write(f"  95% Confidence Interval: [{results['sep_ci_95'][0]:.4f}, {results['sep_ci_95'][1]:.4f}]\n")
        
    metrics_path = os.path.join(metrics_dir, "bootstrap_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
