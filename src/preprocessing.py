"""
preprocessing.py — Data cleaning, outlier detection, and standardization pipeline.

Implements:
  - Data cleaning (remove invalid rows)
  - Missing value analysis (ML_PIPELINE_REFERENCE.md §4)
  - Outlier detection via IQR rule (ML_PIPELINE_REFERENCE.md §5)
  - Z-score standardization (ML_PIPELINE_REFERENCE.md §6)

⚠️ No pandas, no sklearn — pure Python + NumPy arrays.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


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


def check_missing_values(X, feature_names, logs_dir):
    """
    Check for missing values (ML_PIPELINE_REFERENCE.md §4).
    
    For numerical features:
    - Gaussian distribution → mean imputation
    - Skewed distribution → median imputation
    - Multivariate pattern → KNN imputation
    
    For Old Faithful: both features are numeric, dataset is typically complete.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        feature_names (list): Feature names.
        logs_dir (str): Directory for log files.
    
    Returns:
        numpy.ndarray: Data with missing values handled (or unchanged if none).
    """
    log_lines = []
    log_lines.append("MISSING VALUE ANALYSIS")
    log_lines.append("=" * 50)
    
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
    log_path = os.path.join(logs_dir, "04_missing_values.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Log saved: {log_path}")
    
    return X


def detect_outliers_iqr(X, feature_names, iqr_multiplier=1.5):
    """
    Detect outliers using the IQR rule (ML_PIPELINE_REFERENCE.md §5.2).
    
    IQR = Q3 - Q1
    Lower bound = Q1 - 1.5 × IQR
    Upper bound = Q3 + 1.5 × IQR
    Points outside [Lower, Upper] are flagged.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        feature_names (list): Feature names.
        iqr_multiplier (float): IQR multiplier (default 1.5).
    
    Returns:
        dict: Per-feature outlier info {name: {indices, lower, upper, count}}.
    """
    outlier_info = {}
    
    for i, name in enumerate(feature_names):
        col = X[:, i]
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1
        
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        
        mask = (col < lower) | (col > upper)
        indices = np.where(mask)[0]
        
        outlier_info[name] = {
            'indices': indices,
            'lower': lower,
            'upper': upper,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'count': len(indices),
            'values': col[mask] if len(indices) > 0 else np.array([]),
        }
    
    return outlier_info


def handle_outliers(X, feature_names, iqr_multiplier, plots_dir, logs_dir):
    """
    Detect and document outliers (ML_PIPELINE_REFERENCE.md §5).
    
    Strategy for Old Faithful: Document outliers but DO NOT remove them.
    Reason: Old Faithful data points are genuine measurements, not errors.
    The bimodal structure means some points naturally fall between clusters.
    GMM with full covariance handles this gracefully.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        feature_names (list): Feature names.
        iqr_multiplier (float): IQR multiplier.
        plots_dir (str): Directory for plots.
        logs_dir (str): Directory for log files.
    
    Returns:
        numpy.ndarray: Data (unchanged — outliers documented but kept).
    """
    print("\n  Detecting outliers (IQR rule)...")
    outlier_info = detect_outliers_iqr(X, feature_names, iqr_multiplier)
    
    log_lines = []
    log_lines.append("OUTLIER DETECTION AND TREATMENT")
    log_lines.append("=" * 50)
    log_lines.append(f"Method: IQR Rule (multiplier = {iqr_multiplier})")
    log_lines.append(f"Reference: ML_PIPELINE_REFERENCE.md §5")
    log_lines.append("")
    
    total_outliers = 0
    for name, info in outlier_info.items():
        log_lines.append(f"  {name}:")
        log_lines.append(f"    Q1 = {info['q1']:.4f}, Q3 = {info['q3']:.4f}, IQR = {info['iqr']:.4f}")
        log_lines.append(f"    Bounds: [{info['lower']:.4f}, {info['upper']:.4f}]")
        log_lines.append(f"    Outliers found: {info['count']}")
        if info['count'] > 0:
            log_lines.append(f"    Outlier values: {info['values']}")
        log_lines.append("")
        total_outliers += info['count']
        print(f"    {name}: {info['count']} outliers (bounds [{info['lower']:.2f}, {info['upper']:.2f}])")
    
    log_lines.append("TREATMENT DECISION: KEEP all data points")
    log_lines.append("Justification:")
    log_lines.append("  - Old Faithful data are genuine physical measurements")
    log_lines.append("  - Bimodal structure means inter-cluster points are NOT errors")
    log_lines.append("  - GMM with full covariance handles non-spherical clusters")
    log_lines.append("  - Removing points would reduce already small dataset (N=272)")
    log_lines.append("  - Ref: ML_PIPELINE_REFERENCE.md §5 — 'Never blindly remove outliers'")
    
    # Plot: boxplots with outlier annotations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, ax) in enumerate(zip(feature_names, axes)):
        col = X[:, i]
        info = outlier_info[name]
        
        bp = ax.boxplot(col, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#3498DB', alpha=0.6),
                        medianprops=dict(color='#E74C3C', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='#E74C3C',
                                       markersize=8, alpha=0.7))
        
        # Annotate bounds
        ax.axhline(info['lower'], color='#E74C3C', linestyle='--', alpha=0.5,
                   label=f'Lower={info["lower"]:.2f}')
        ax.axhline(info['upper'], color='#E74C3C', linestyle='--', alpha=0.5,
                   label=f'Upper={info["upper"]:.2f}')
        
        ax.set_ylabel(name)
        ax.set_title(f'{name}\n({info["count"]} outliers detected, KEPT)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Outlier Detection (IQR Rule) — All Points Retained',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "05_outliers_boxplot.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")
    
    # Save log
    log_path = os.path.join(logs_dir, "05_outliers.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Log saved: {log_path}")
    
    return X


def compute_mean(data):
    """
    Compute column-wise mean manually.
    
    Formula: mean = (1/N) * sum(x_i)
    
    Args:
        data (numpy.ndarray): Data array of shape (N, D).
        
    Returns:
        numpy.ndarray: Mean vector of shape (D,).
    """
    n_samples = data.shape[0]
    mean = np.zeros(data.shape[1])
    for i in range(n_samples):
        mean += data[i]
    return mean / n_samples


def compute_std(data, mean):
    """
    Compute column-wise standard deviation manually.
    
    Formula: std = sqrt((1/N) * sum((x_i - mean)^2))
    
    Args:
        data (numpy.ndarray): Data array of shape (N, D).
        mean (numpy.ndarray): Mean vector of shape (D,).
        
    Returns:
        numpy.ndarray: Standard deviation vector of shape (D,).
    """
    n_samples = data.shape[0]
    variance = np.zeros(data.shape[1])
    for i in range(n_samples):
        diff = data[i] - mean
        variance += diff * diff
    variance = variance / n_samples
    return np.sqrt(variance)


def standardize(data, logs_dir=None):
    """
    Apply z-score standardization: x' = (x - mean) / std
    
    This transforms each feature to have mean=0 and std=1.
    Essential for GMM because (ref ML_PIPELINE_REFERENCE.md §6):
    - Features on different scales dominate Euclidean distance
    - Covariance matrix becomes ill-conditioned with mixed scales
    - EM convergence is faster with normalized features
    
    Decision: StandardScaler chosen over MinMaxScaler because:
    - GMM assumes Gaussian components → standardization is natural
    - No fixed known bounds for eruption/waiting features
    - Ref §6.4: "PCA, SVM, Linear models, GMM → Standardization"
    
    Args:
        data (numpy.ndarray): Raw data of shape (N, D).
        logs_dir (str, optional): Directory for log files.
        
    Returns:
        tuple: (standardized_data, mean, std) for inverse transform.
    """
    mean = compute_mean(data)
    std = compute_std(data, mean)
    
    print(f"  Feature means: eruptions={mean[0]:.4f}, waiting={mean[1]:.4f}")
    print(f"  Feature stds:  eruptions={std[0]:.4f}, waiting={std[1]:.4f}")
    
    standardized = (data - mean) / std
    
    # Log scaling parameters
    if logs_dir is not None:
        log_lines = []
        log_lines.append("FEATURE SCALING")
        log_lines.append("=" * 50)
        log_lines.append("Method: Z-score Standardization (x' = (x - μ) / σ)")
        log_lines.append("Reference: ML_PIPELINE_REFERENCE.md §6")
        log_lines.append("")
        log_lines.append("Justification:")
        log_lines.append("  - GMM requires standardization (§6.3, §6.4)")
        log_lines.append("  - Prevents covariance estimation distortion")
        log_lines.append("  - No fixed bounds → MinMax not appropriate")
        log_lines.append("")
        log_lines.append("Parameters:")
        log_lines.append(f"  eruptions: mean={mean[0]:.6f}, std={std[0]:.6f}")
        log_lines.append(f"  waiting:   mean={mean[1]:.6f}, std={std[1]:.6f}")
        log_lines.append("")
        log_lines.append("Post-scaling verification:")
        log_lines.append(f"  eruptions: mean={np.mean(standardized[:, 0]):.6f}, std={np.std(standardized[:, 0]):.6f}")
        log_lines.append(f"  waiting:   mean={np.mean(standardized[:, 1]):.6f}, std={np.std(standardized[:, 1]):.6f}")
        
        log_path = os.path.join(logs_dir, "06_scaling.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write("\n".join(log_lines))
        print(f"  Log saved: {log_path}")
    
    return standardized, mean, std


def save_csv(data, filepath, header="eruptions,waiting"):
    """
    Save numpy array to CSV file using manual file I/O.
    
    Args:
        data (numpy.ndarray): Data array of shape (N, 2).
        filepath (str): Output file path.
        header (str): CSV header line.
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(header + '\n')
        for row in data:
            f.write(f"{row[0]:.6f},{row[1]:.6f}\n")
    
    print(f"  Saved {len(data)} rows to {filepath}")


def run_pipeline(raw_data_path, processed_data_path, feature_names,
                 iqr_multiplier, plots_dir, logs_dir):
    """
    Execute the full preprocessing pipeline:
    1. Load raw CSV
    2. Clean invalid rows
    3. Check missing values (ref §4)
    4. Detect outliers (ref §5)
    5. Standardize features (ref §6)
    6. Save processed data
    
    Args:
        raw_data_path (str): Path to raw CSV.
        processed_data_path (str): Path to save processed CSV.
        feature_names (list): Feature names.
        iqr_multiplier (float): IQR multiplier for outlier detection.
        plots_dir (str): Directory for plots.
        logs_dir (str): Directory for logs.
    
    Returns:
        tuple: (raw_clean_data, standardized_data, mean, std)
    """
    from src.data_loader import load_csv
    
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load
    print("\n[Step 1] Loading raw data...")
    raw_data = load_csv(raw_data_path)
    
    # Log data loading
    log_lines = [
        "DATA LOADING",
        "=" * 50,
        f"Source: {raw_data_path}",
        f"Rows loaded: {len(raw_data)}",
        f"Features: {feature_names}",
        f"Data type: numeric (float)",
    ]
    log_path = os.path.join(logs_dir, "02_data_loading.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Log saved: {log_path}")
    
    # Step 2: Clean
    print("\n[Step 2] Cleaning data...")
    clean = clean_data(raw_data)
    
    # Step 3: Check missing values (ref §4)
    print("\n[Step 3] Checking missing values...")
    clean = check_missing_values(clean, feature_names, logs_dir)
    
    # Step 4: Detect outliers (ref §5)
    print("\n[Step 4] Outlier detection...")
    clean = handle_outliers(clean, feature_names, iqr_multiplier, plots_dir, logs_dir)
    
    # Step 5: Standardize (ref §6)
    print("\n[Step 5] Standardizing features (z-score)...")
    standardized, mean, std = standardize(clean, logs_dir)
    
    # Step 6: Save
    print("\n[Step 6] Saving processed data...")
    save_csv(standardized, processed_data_path)
    
    print("\n[DONE] Preprocessing complete.")
    return clean, standardized, mean, std
