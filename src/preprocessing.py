"""
preprocessing.py — Data cleaning, outlier detection, and standardization pipeline.

Refactored to support the train/test split boundary (ML_PIPELINE_REFERENCE.md §10):
  - All "fit" operations compute statistics from TRAIN DATA ONLY
  - "transform" operations apply those statistics to both train and test
  - This prevents data leakage

Implements:
  - Data cleaning (remove invalid rows)
  - Missing value analysis (§4)
  - Outlier detection via IQR rule (§5)
  - Z-score standardization with separate fit/transform (§6)

⚠️ No pandas, no sklearn — pure Python + NumPy arrays.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ═══════════════════════════════════════════════════════════════
# DATA CLEANING
# ═══════════════════════════════════════════════════════════════

def clean_data(raw_data):
    """
    Remove invalid rows (NaN, negative, or zero values).
    
    Old Faithful data should have:
    - eruptions > 0 (duration in minutes)
    - waiting > 0 (wait time in minutes)
    
    Args:
        raw_data (list): List of [eruptions, waiting] pairs.
        
    Returns:
        numpy.ndarray: Cleaned data array of shape (N, 2).
    """
    cleaned = []
    removed_count = 0
    
    for row in raw_data:
        eruptions, waiting = row[0], row[1]
        
        # Check for valid numeric values
        if eruptions > 0 and waiting > 0:
            cleaned.append([eruptions, waiting])
        else:
            removed_count += 1
    
    if removed_count > 0:
        print(f"  Removed {removed_count} invalid rows")
    
    print(f"  Clean data: {len(cleaned)} rows")
    return np.array(cleaned)


# ═══════════════════════════════════════════════════════════════
# MISSING VALUES (§4)
# ═══════════════════════════════════════════════════════════════

def check_missing_values(X, feature_names, logs_dir, dataset_label="full"):
    """
    Check for missing values and apply imputation if needed (§4).
    
    Imputation strategy is chosen based on distribution:
    - Gaussian → mean imputation
    - Skewed → median imputation
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        feature_names (list): Feature names.
        logs_dir (str): Directory for log files.
        dataset_label (str): Label for logging ('train' or 'full').
        
    Returns:
        numpy.ndarray: Data with missing values handled (or unchanged if none).
    """
    log_lines = []
    log_lines.append("MISSING VALUE ANALYSIS")
    log_lines.append("=" * 50)
    log_lines.append(f"Dataset: {dataset_label} (N={X.shape[0]})")
    
    n_missing = np.sum(np.isnan(X), axis=0)
    total_missing = np.sum(n_missing)
    
    for i, name in enumerate(feature_names):
        log_lines.append(f"  {name}: {n_missing[i]} missing values")
    
    if total_missing == 0:
        log_lines.append("\nNo missing values found. No imputation needed.")
        log_lines.append("Decision: SKIP imputation (ref ML_PIPELINE_REFERENCE.md §4)")
        print("  No missing values found — skipping imputation")
    else:
        log_lines.append(f"\nTotal missing: {total_missing}")
        log_lines.append("Applying median imputation (robust to outliers)")
        # Median imputation for each feature
        for i in range(X.shape[1]):
            col = X[:, i]
            mask = np.isnan(col)
            if np.any(mask):
                median_val = np.nanmedian(col)
                X[mask, i] = median_val
                log_lines.append(f"  {feature_names[i]}: imputed {np.sum(mask)} values with median={median_val:.4f}")
        print(f"  Imputed {total_missing} missing values using median strategy")
    
    # Save log
    log_path = os.path.join(logs_dir, "05_imputation.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Log saved: {log_path}")
    
    return X


# ═══════════════════════════════════════════════════════════════
# OUTLIER DETECTION (§5) — fit on train, apply to both
# ═══════════════════════════════════════════════════════════════

def fit_outlier_bounds(X_train, feature_names, iqr_multiplier=1.5):
    """
    Compute IQR-based outlier bounds from TRAINING DATA ONLY (§5.2).
    
    IQR = Q3 - Q1
    Lower bound = Q1 - 1.5 × IQR
    Upper bound = Q3 + 1.5 × IQR
    
    These bounds are then applied to both train and test data.
    
    Args:
        X_train (numpy.ndarray): Training data of shape (N_train, D).
        feature_names (list): Feature names.
        iqr_multiplier (float): IQR multiplier (default 1.5).
    
    Returns:
        dict: Per-feature outlier bounds {name: {lower, upper, q1, q3, iqr}}.
    """
    bounds = {}
    
    for i, name in enumerate(feature_names):
        col = X_train[:, i]
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1
        
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        
        bounds[name] = {
            'lower': lower,
            'upper': upper,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'feature_idx': i,
        }
    
    return bounds


def detect_outliers(X, feature_names, bounds):
    """
    Detect outliers using pre-computed bounds (from training data).
    
    Args:
        X (numpy.ndarray): Data to check (train or test).
        feature_names (list): Feature names.
        bounds (dict): Outlier bounds from fit_outlier_bounds.
        
    Returns:
        dict: Per-feature outlier info {name: {indices, count, values}}.
    """
    outlier_info = {}
    
    for name in feature_names:
        i = bounds[name]['feature_idx']
        col = X[:, i]
        lower = bounds[name]['lower']
        upper = bounds[name]['upper']
        
        mask = (col < lower) | (col > upper)
        indices = np.where(mask)[0]
        
        outlier_info[name] = {
            'indices': indices,
            'count': len(indices),
            'values': col[mask] if len(indices) > 0 else np.array([]),
            'lower': lower,
            'upper': upper,
        }
    
    return outlier_info


def handle_outliers(X_train, X_test, feature_names, iqr_multiplier,
                    plots_dir, logs_dir):
    """
    Detect and document outliers — bounds computed from TRAIN ONLY (§5, §10).
    
    Strategy for Old Faithful: Document outliers but DO NOT remove them.
    Reason: Old Faithful data points are genuine measurements, not errors.
    The bimodal structure means some points naturally fall between clusters.
    GMM with full covariance handles this gracefully.
    
    Args:
        X_train (numpy.ndarray): Training data of shape (N_train, D).
        X_test (numpy.ndarray): Test data of shape (N_test, D).
        feature_names (list): Feature names.
        iqr_multiplier (float): IQR multiplier.
        plots_dir (str): Directory for plots.
        logs_dir (str): Directory for log files.
    
    Returns:
        tuple: (X_train, X_test) — unchanged, outliers documented but kept.
    """
    print("\n  Detecting outliers (IQR rule, bounds from TRAIN DATA)...")
    
    # Fit bounds on TRAIN ONLY
    bounds = fit_outlier_bounds(X_train, feature_names, iqr_multiplier)
    
    # Detect outliers in both sets using TRAIN bounds
    train_outliers = detect_outliers(X_train, feature_names, bounds)
    test_outliers = detect_outliers(X_test, feature_names, bounds)
    
    log_lines = []
    log_lines.append("OUTLIER DETECTION AND TREATMENT")
    log_lines.append("=" * 50)
    log_lines.append(f"Method: IQR Rule (multiplier = {iqr_multiplier})")
    log_lines.append(f"Bounds computed from: TRAINING DATA ONLY (§10)")
    log_lines.append(f"Reference: ML_PIPELINE_REFERENCE.md §5")
    log_lines.append("")
    
    total_train = 0
    total_test = 0
    for name in feature_names:
        b = bounds[name]
        t_info = train_outliers[name]
        te_info = test_outliers[name]
        
        log_lines.append(f"  {name}:")
        log_lines.append(f"    Q1 = {b['q1']:.4f}, Q3 = {b['q3']:.4f}, IQR = {b['iqr']:.4f}")
        log_lines.append(f"    Bounds: [{b['lower']:.4f}, {b['upper']:.4f}]")
        log_lines.append(f"    Train outliers: {t_info['count']}")
        log_lines.append(f"    Test outliers:  {te_info['count']}")
        if t_info['count'] > 0:
            log_lines.append(f"    Train outlier values: {t_info['values']}")
        log_lines.append("")
        total_train += t_info['count']
        total_test += te_info['count']
        print(f"    {name}: train={t_info['count']}, test={te_info['count']} outliers "
              f"(bounds [{b['lower']:.2f}, {b['upper']:.2f}])")
    
    log_lines.append("TREATMENT DECISION: KEEP all data points")
    log_lines.append("Justification:")
    log_lines.append("  - Old Faithful data are genuine physical measurements")
    log_lines.append("  - Bimodal structure means inter-cluster points are NOT errors")
    log_lines.append("  - GMM with full covariance handles non-spherical clusters")
    log_lines.append("  - Removing points would reduce already small dataset")
    log_lines.append("  - Ref: ML_PIPELINE_REFERENCE.md §5 — 'Never blindly remove outliers'")
    
    # Plot: boxplots with outlier annotations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, ax) in enumerate(zip(feature_names, axes)):
        col = X_train[:, i]
        b = bounds[name]
        t_info = train_outliers[name]
        
        bp = ax.boxplot(col, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#3498DB', alpha=0.6),
                        medianprops=dict(color='#E74C3C', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='#E74C3C',
                                       markersize=8, alpha=0.7))
        
        ax.axhline(b['lower'], color='#E74C3C', linestyle='--', alpha=0.5,
                   label=f'Lower={b["lower"]:.2f}')
        ax.axhline(b['upper'], color='#E74C3C', linestyle='--', alpha=0.5,
                   label=f'Upper={b["upper"]:.2f}')
        
        ax.set_ylabel(name)
        ax.set_title(f'{name}\n({t_info["count"]} outliers detected, KEPT)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Outlier Detection (IQR, bounds from TRAIN) — All Points Retained',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "06_outliers_boxplot.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")
    
    # Save log
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "06_outliers.txt")
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Log saved: {log_path}")
    
    return X_train, X_test


# ═══════════════════════════════════════════════════════════════
# FEATURE SCALING (§6) — fit on train, transform both
# ═══════════════════════════════════════════════════════════════

def fit_scaler(X_train):
    """
    Compute scaling parameters from TRAINING DATA ONLY (§6, §10).
    
    Uses z-score standardization: x' = (x - μ) / σ
    
    Decision: StandardScaler chosen over MinMaxScaler because:
    - GMM assumes Gaussian components → standardization is natural
    - No fixed known bounds for eruption/waiting features
    - Ref §6.4: "PCA, SVM, Linear models, GMM → Standardization"
    
    Args:
        X_train (numpy.ndarray): Training data of shape (N_train, D).
        
    Returns:
        dict: Scaler parameters {'mean': ndarray, 'std': ndarray}.
    """
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    
    # Compute mean manually
    mean = np.zeros(n_features)
    for i in range(n_samples):
        mean += X_train[i]
    mean /= n_samples
    
    # Compute std manually
    variance = np.zeros(n_features)
    for i in range(n_samples):
        diff = X_train[i] - mean
        variance += diff * diff
    variance /= n_samples
    std = np.sqrt(variance)
    
    # Prevent division by zero
    std = np.maximum(std, 1e-10)
    
    return {'mean': mean, 'std': std}


def transform_scaler(X, scaler_params):
    """
    Apply z-score standardization using pre-computed parameters.
    
    x' = (x - μ_train) / σ_train
    
    CRITICAL: Uses train-fitted parameters for BOTH train and test (§10).
    Never refit on test data.
    
    Args:
        X (numpy.ndarray): Data to transform of shape (N, D).
        scaler_params (dict): Parameters from fit_scaler.
        
    Returns:
        numpy.ndarray: Standardized data.
    """
    return (X - scaler_params['mean']) / scaler_params['std']


def log_scaling(scaler_params, X_train_scaled, X_test_scaled,
                feature_names, logs_dir):
    """
    Log scaling parameters and post-scaling verification.
    
    Args:
        scaler_params (dict): Scaler parameters from fit_scaler.
        X_train_scaled (numpy.ndarray): Scaled training data.
        X_test_scaled (numpy.ndarray): Scaled test data.
        feature_names (list): Feature names.
        logs_dir (str): Directory for log files.
    """
    log_lines = []
    log_lines.append("FEATURE SCALING")
    log_lines.append("=" * 50)
    log_lines.append("Method: Z-score Standardization (x' = (x - μ) / σ)")
    log_lines.append("Reference: ML_PIPELINE_REFERENCE.md §6")
    log_lines.append("⚠️  Parameters fitted on TRAINING DATA ONLY (§10)")
    log_lines.append("")
    log_lines.append("Justification:")
    log_lines.append("  - GMM requires standardization (§6.3, §6.4)")
    log_lines.append("  - Prevents covariance estimation distortion")
    log_lines.append("  - No fixed bounds → MinMax not appropriate")
    log_lines.append("")
    log_lines.append("Parameters (from training data):")
    for i, name in enumerate(feature_names):
        log_lines.append(f"  {name}: mean={scaler_params['mean'][i]:.6f}, "
                        f"std={scaler_params['std'][i]:.6f}")
    log_lines.append("")
    log_lines.append("Post-scaling verification (TRAIN):")
    for i, name in enumerate(feature_names):
        log_lines.append(f"  {name}: mean={np.mean(X_train_scaled[:, i]):.6f}, "
                        f"std={np.std(X_train_scaled[:, i]):.6f}")
    log_lines.append("")
    log_lines.append("Post-scaling verification (TEST — using TRAIN parameters):")
    for i, name in enumerate(feature_names):
        log_lines.append(f"  {name}: mean={np.mean(X_test_scaled[:, i]):.6f}, "
                        f"std={np.std(X_test_scaled[:, i]):.6f}")
    log_lines.append("")
    log_lines.append("NOTE: Test set mean/std will NOT be exactly 0/1 because")
    log_lines.append("scaling parameters come from train set. This is CORRECT —")
    log_lines.append("it means no leakage occurred.")
    
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "07_scaling.txt")
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Log saved: {log_path}")


# ═══════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════

def save_csv(data, filepath, header="eruptions,waiting"):
    """
    Save numpy array to CSV file using manual file I/O.
    
    Args:
        data (numpy.ndarray): Data array of shape (N, D).
        filepath (str): Output file path.
        header (str): CSV header line.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(header + '\n')
        for row in data:
            f.write(','.join(f"{v:.6f}" for v in row) + '\n')
    
    print(f"  Saved {len(data)} rows to {filepath}")
