"""
eda.py — Exploratory Data Analysis for the Old Faithful dataset.

Implements the three EDA lenses from ML_PIPELINE_REFERENCE.md §3:
  Lens 1: Global overview (shape, types, missing, ranges)
  Lens 2: Univariate distribution (histograms, KDE, skewness)
  Lens 3: Multivariate relationships (correlation, scatter)

All plots saved to results/plots/03_eda_*.png
All summaries logged to results/logs/03_eda_summary.txt

⚠️ Pure numpy + matplotlib only. No pandas, no scipy, no sklearn.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def compute_skewness(data):
    """
    Compute skewness manually: skew = (1/N) * sum(((x - mean) / std)^3)
    
    Interpretation:
        skew ≈ 0  → symmetric (Gaussian-like)
        skew > 0  → right-skewed (long right tail)
        skew < 0  → left-skewed (long left tail)
    
    Args:
        data (numpy.ndarray): 1D array of values.
    
    Returns:
        float: Skewness value.
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-10:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)


def compute_kurtosis(data):
    """
    Compute excess kurtosis: kurt = (1/N) * sum(((x - mean) / std)^4) - 3
    
    Interpretation:
        kurt ≈ 0  → mesokurtic (Gaussian-like tails)
        kurt > 0  → leptokurtic (heavy tails)
        kurt < 0  → platykurtic (light tails)
    
    Args:
        data (numpy.ndarray): 1D array of values.
    
    Returns:
        float: Excess kurtosis value.
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-10:
        return 0.0
    return np.mean(((data - mean) / std) ** 4) - 3.0


def compute_correlation_matrix(X):
    """
    Compute Pearson correlation matrix manually.
    
    r(i,j) = cov(i,j) / (std_i * std_j)
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
    
    Returns:
        numpy.ndarray: Correlation matrix of shape (D, D).
    """
    n_samples, n_features = X.shape
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    
    # Prevent division by zero
    stds = np.maximum(stds, 1e-10)
    
    corr = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            cov_ij = np.mean((X[:, i] - means[i]) * (X[:, j] - means[j]))
            corr[i, j] = cov_ij / (stds[i] * stds[j])
    
    return corr


def gaussian_kde_1d(data, x_grid, bandwidth=None):
    """
    Compute 1D Kernel Density Estimate using Gaussian kernels.
    
    KDE(x) = (1 / (N * h)) * sum_i K((x - x_i) / h)
    where K is the Gaussian kernel: K(u) = (1/sqrt(2*pi)) * exp(-u^2/2)
    
    Bandwidth selection: Silverman's rule h = 0.9 * min(std, IQR/1.34) * N^(-1/5)
    
    Args:
        data (numpy.ndarray): 1D data array.
        x_grid (numpy.ndarray): Points at which to evaluate the KDE.
        bandwidth (float, optional): Kernel bandwidth. Auto if None.
    
    Returns:
        numpy.ndarray: KDE values at x_grid points.
    """
    n = len(data)
    
    if bandwidth is None:
        # Silverman's rule of thumb
        std = np.std(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bandwidth = 0.9 * min(std, iqr / 1.34) * n ** (-0.2)
        bandwidth = max(bandwidth, 1e-6)
    
    kde_values = np.zeros_like(x_grid)
    for xi in data:
        u = (x_grid - xi) / bandwidth
        kde_values += np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    
    kde_values /= (n * bandwidth)
    return kde_values


def run_global_overview(X, feature_names, log_lines):
    """
    Lens 1 — Global Overview (ML_PIPELINE_REFERENCE.md §3.1)
    
    Questions answered:
    - How many samples? How many features?
    - What are the feature types?
    - Are there missing values?
    - What are the value ranges?
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        feature_names (list): Feature names.
        log_lines (list): Accumulator for log text.
    """
    n_samples, n_features = X.shape
    
    log_lines.append("=" * 60)
    log_lines.append("LENS 1 — GLOBAL OVERVIEW")
    log_lines.append("=" * 60)
    log_lines.append(f"Number of samples:  {n_samples}")
    log_lines.append(f"Number of features: {n_features}")
    log_lines.append(f"Feature names:      {feature_names}")
    log_lines.append(f"Data types:         All numeric (float64)")
    log_lines.append("")
    
    # Missing values
    n_missing = np.sum(np.isnan(X), axis=0)
    for i, name in enumerate(feature_names):
        log_lines.append(f"  {name}: {n_missing[i]} missing values")
    log_lines.append(f"  Total missing: {np.sum(n_missing)}")
    log_lines.append("")
    
    # Value ranges and summary statistics
    log_lines.append(f"{'Feature':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10} {'Std':>10}")
    log_lines.append("-" * 65)
    for i, name in enumerate(feature_names):
        col = X[:, i]
        log_lines.append(
            f"{name:<15} {np.min(col):>10.4f} {np.max(col):>10.4f} "
            f"{np.mean(col):>10.4f} {np.median(col):>10.4f} {np.std(col):>10.4f}"
        )
    log_lines.append("")


def run_univariate_analysis(X, feature_names, plots_dir, log_lines):
    """
    Lens 2 — Univariate Distribution (ML_PIPELINE_REFERENCE.md §3.2)
    
    For each feature:
    - Histogram + KDE overlay
    - Summary statistics (mean, median, std, skewness, kurtosis)
    - Distribution assessment (Gaussian vs skewed)
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        feature_names (list): Feature names.
        plots_dir (str): Directory to save plots.
        log_lines (list): Accumulator for log text.
    """
    log_lines.append("=" * 60)
    log_lines.append("LENS 2 — UNIVARIATE DISTRIBUTIONS")
    log_lines.append("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, (name, ax) in enumerate(zip(feature_names, axes)):
        col = X[:, i]
        
        # Statistics
        mean = np.mean(col)
        median = np.median(col)
        std = np.std(col)
        skew = compute_skewness(col)
        kurt = compute_kurtosis(col)
        
        log_lines.append(f"\n  {name}:")
        log_lines.append(f"    Mean:     {mean:.4f}")
        log_lines.append(f"    Median:   {median:.4f}")
        log_lines.append(f"    Std:      {std:.4f}")
        log_lines.append(f"    Skewness: {skew:.4f}")
        log_lines.append(f"    Kurtosis: {kurt:.4f}")
        
        # Distribution assessment (ref §3.2)
        if abs(skew) < 0.5:
            assessment = "Approximately symmetric → mean is representative"
        else:
            assessment = "Skewed → use median; consider log transform"
        log_lines.append(f"    Assessment ({name}): {assessment}")
        
        # Histogram
        ax.hist(col, bins=25, density=True, alpha=0.6, color='#3498DB',
                edgecolor='white', linewidth=0.5, label='Histogram')
        
        # KDE overlay
        x_grid = np.linspace(np.min(col) - 0.5 * std, np.max(col) + 0.5 * std, 200)
        kde = gaussian_kde_1d(col, x_grid)
        ax.plot(x_grid, kde, color='#E74C3C', linewidth=2, label='KDE')
        
        # Mark mean and median
        ax.axvline(mean, color='#2ECC71', linestyle='--', linewidth=1.5,
                   label=f'Mean={mean:.2f}')
        ax.axvline(median, color='#F39C12', linestyle=':', linewidth=1.5,
                   label=f'Median={median:.2f}')
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Distribution\n(skew={skew:.2f}, kurt={kurt:.2f})')
        ax.legend(fontsize=8)
    
    plt.suptitle('Univariate Distributions — Old Faithful', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "03_eda_histograms.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    log_lines.append("")


def run_multivariate_analysis(X, feature_names, plots_dir, log_lines):
    """
    Lens 3 — Multivariate Relationships (ML_PIPELINE_REFERENCE.md §3.3)
    
    - Correlation matrix heatmap
    - Scatter plot with bimodal structure visible
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        feature_names (list): Feature names.
        plots_dir (str): Directory to save plots.
        log_lines (list): Accumulator for log text.
    """
    log_lines.append("=" * 60)
    log_lines.append("LENS 3 — MULTIVARIATE RELATIONSHIPS")
    log_lines.append("=" * 60)
    
    # Correlation matrix
    corr = compute_correlation_matrix(X)
    
    log_lines.append("\n  Correlation Matrix:")
    header = f"{'':>15}" + "".join(f"{name:>15}" for name in feature_names)
    log_lines.append(header)
    for i, name in enumerate(feature_names):
        row = f"{name:>15}" + "".join(f"{corr[i, j]:>15.4f}" for j in range(len(feature_names)))
        log_lines.append(row)
    
    # Interpretation (ref §3.3)
    r = corr[0, 1]
    if abs(r) > 0.7:
        interp = "Strong correlation — features are highly co-linear"
    elif abs(r) > 0.3:
        interp = "Moderate correlation — consider interaction terms"
    else:
        interp = "Weak correlation — features are approximately independent"
    log_lines.append(f"\n  r(eruptions, waiting) = {r:.4f}")
    log_lines.append(f"  Interpretation: {interp}")
    log_lines.append("")
    
    # Plot: correlation heatmap + scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap
    im = axes[0].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_xticks(range(len(feature_names)))
    axes[0].set_yticks(range(len(feature_names)))
    axes[0].set_xticklabels(feature_names, fontsize=10)
    axes[0].set_yticklabels(feature_names, fontsize=10)
    axes[0].set_title('Correlation Matrix')
    
    # Annotate cells
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            axes[0].text(j, i, f'{corr[i, j]:.3f}', ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color='white' if abs(corr[i, j]) > 0.5 else 'black')
    
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    
    # Scatter plot
    axes[1].scatter(X[:, 0], X[:, 1], c='#34495E', alpha=0.6, s=20,
                    edgecolors='white', linewidths=0.5)
    axes[1].set_xlabel(feature_names[0])
    axes[1].set_ylabel(feature_names[1])
    axes[1].set_title(f'Scatter Plot (r={r:.3f})\nBimodal structure visible')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Multivariate Analysis — Old Faithful', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "03_eda_correlation.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def run_boxplots(X, feature_names, plots_dir):
    """
    Generate boxplots for visual outlier detection (ref §5.2).
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        feature_names (list): Feature names.
        plots_dir (str): Directory to save plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (name, ax) in enumerate(zip(feature_names, axes)):
        col = X[:, i]
        bp = ax.boxplot(col, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#3498DB', alpha=0.6),
                        medianprops=dict(color='#E74C3C', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='#E74C3C',
                                       markersize=6, alpha=0.7))
        ax.set_ylabel(name)
        ax.set_title(f'{name} — Box Plot')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Box Plots — Outlier Detection (EDA)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "03_eda_boxplots.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def run_eda(X, feature_names, plots_dir, logs_dir):
    """
    Run the complete EDA pipeline.
    
    Implements ML_PIPELINE_REFERENCE.md §3 (all three lenses).
    
    Args:
        X (numpy.ndarray): Raw (unscaled) data of shape (N, D).
        feature_names (list): Feature names.
        plots_dir (str): Directory for plots.
        logs_dir (str): Directory for log files.
    
    Returns:
        list: Log lines for reference.
    """
    
    log_lines = []
    log_lines.append("EXPLORATORY DATA ANALYSIS — OLD FAITHFUL GEYSER")
    log_lines.append(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    log_lines.append(f"Features: {feature_names}")
    log_lines.append("")
    
    # Lens 1: Global overview
    print("\n[Lens 1] Global overview...")
    run_global_overview(X, feature_names, log_lines)
    
    # Lens 2: Univariate distributions
    print("[Lens 2] Univariate distributions...")
    run_univariate_analysis(X, feature_names, plots_dir, log_lines)
    
    # Lens 3: Multivariate relationships
    print("[Lens 3] Multivariate relationships...")
    run_multivariate_analysis(X, feature_names, plots_dir, log_lines)
    
    # Box plots for outlier detection preview
    print("[Extra] Box plots for outlier preview...")
    run_boxplots(X, feature_names, plots_dir)
    
    # Save log
    log_path = os.path.join(logs_dir, "03_eda_summary.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Summary saved: {log_path}")
    
    # Print key findings
    print("\n  KEY EDA FINDINGS:")
    for line in log_lines:
        if "Assessment:" in line or "Interpretation:" in line or "missing" in line.lower():
            print(f"  {line.strip()}")
    
    return log_lines
