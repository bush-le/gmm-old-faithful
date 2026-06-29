"""
visualization.py — Plotting functions for GMM pipeline using Matplotlib only.

Generates publication-quality visualizations for:
- Raw data scatter plots
- GMM Gaussian ellipse overlays
- Convergence curves
- BIC/AIC model selection plots

All plots saved via plt.savefig() — no interactive plt.show().

⚠️ Pure matplotlib only. No seaborn, no plotly.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import os


# Color palette for clusters
CLUSTER_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']


def setup_plot_style():
    """Configure matplotlib for clean, professional plots."""
    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_raw_data(X, save_path):
    """
    Plot the raw (standardized) dataset.
    
    Args:
        X (numpy.ndarray): Data of shape (N, 2).
        save_path (str): Path to save the plot.
    """
    setup_plot_style()
    fig, ax = plt.subplots()
    
    ax.scatter(X[:, 0], X[:, 1], c='#34495E', alpha=0.6, s=20,
              edgecolors='white', linewidths=0.5)
    
    ax.set_xlabel('Eruption Duration (standardized)')
    ax.set_ylabel('Waiting Time (standardized)')
    ax.set_title('Old Faithful Geyser Dataset')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_clusters(X, labels, title, save_path, centroids=None):
    """
    Plot data points colored by cluster assignment.
    
    Args:
        X (numpy.ndarray): Data of shape (N, 2).
        labels (numpy.ndarray): Cluster labels of shape (N,).
        title (str): Plot title.
        save_path (str): Path to save the plot.
        centroids (numpy.ndarray, optional): Cluster centers of shape (K, 2).
    """
    setup_plot_style()
    fig, ax = plt.subplots()
    
    unique_labels = np.unique(labels)
    for k in unique_labels:
        mask = labels == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, s=20,
                  edgecolors='white', linewidths=0.5,
                  label=f'Cluster {k+1} (n={np.sum(mask)})')
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X',
                  s=200, edgecolors='white', linewidths=2, zorder=5,
                  label='Centroids')
    
    ax.set_xlabel('Eruption Duration (standardized)')
    ax.set_ylabel('Waiting Time (standardized)')
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.9)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def _compute_ellipse_points(mean, cov, n_std=2.0, n_points=100):
    """
    Compute points on a confidence ellipse for a 2D Gaussian.
    
    The ellipse is defined by the eigendecomposition of the covariance matrix.
    At n_std standard deviations, the ellipse contains ~95% of the density.
    
    Math:
    - Eigenvalues of Sigma give the variance along principal axes
    - Eigenvectors give the orientation of the ellipse
    - Semi-axes lengths = n_std * sqrt(eigenvalue)
    
    Args:
        mean (numpy.ndarray): Mean of shape (2,).
        cov (numpy.ndarray): Covariance matrix of shape (2, 2).
        n_std (float): Number of standard deviations for ellipse radius.
        n_points (int): Number of points to sample on the ellipse.
        
    Returns:
        numpy.ndarray: Ellipse points of shape (n_points, 2).
    """
    # Eigendecomposition of covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Ensure positive eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # Generate unit circle
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Scale by sqrt(eigenvalues) * n_std, rotate by eigenvectors
    scale = np.diag(n_std * np.sqrt(eigenvalues))
    ellipse = np.dot(circle, np.dot(scale, eigenvectors.T))
    
    # Translate to mean
    ellipse += mean
    
    return ellipse


def plot_gmm_ellipses(X, params, save_path):
    """
    Plot GMM result with Gaussian confidence ellipses.
    
    Shows data colored by most-likely component assignment,
    overlaid with 1-sigma and 2-sigma confidence ellipses.
    
    Args:
        X (numpy.ndarray): Data of shape (N, 2).
        params: GMMParams object with weights, means, covariances.
        save_path (str): Path to save the plot.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Assign each point to most likely component
    from src.gaussian import gaussian_pdf_batch
    n_samples = X.shape[0]
    K = params.K
    
    responsibilities = np.zeros((n_samples, K))
    for k in range(K):
        responsibilities[:, k] = params.weights[k] * gaussian_pdf_batch(
            X, params.means[k], params.covariances[k]
        )
    labels = np.argmax(responsibilities, axis=1)
    
    # Plot data colored by assignment
    for k in range(K):
        mask = labels == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.5, s=20,
                  edgecolors='white', linewidths=0.3,
                  label=f'Component {k+1} (n={np.sum(mask)}, '
                        f'π={params.weights[k]:.3f})')
    
    # Draw confidence ellipses
    for k in range(K):
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        
        # 1-sigma ellipse (~68% confidence)
        ellipse_1 = _compute_ellipse_points(params.means[k], params.covariances[k], n_std=1.0)
        ax.plot(ellipse_1[:, 0], ellipse_1[:, 1], color=color, linewidth=2, linestyle='-')
        
        # 2-sigma ellipse (~95% confidence)
        ellipse_2 = _compute_ellipse_points(params.means[k], params.covariances[k], n_std=2.0)
        ax.plot(ellipse_2[:, 0], ellipse_2[:, 1], color=color, linewidth=1.5, linestyle='--')
        
        # Mark center
        ax.scatter(params.means[k][0], params.means[k][1], c=color, marker='+',
                  s=200, linewidths=3, zorder=5)
    
    ax.set_xlabel('Eruption Duration (standardized)')
    ax.set_ylabel('Waiting Time (standardized)')
    ax.set_title('GMM Clustering with Gaussian Confidence Ellipses\n'
                 '(solid = 1σ ≈ 68%, dashed = 2σ ≈ 95%)')
    ax.legend(loc='upper left', framealpha=0.9)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_convergence(log_likelihoods, save_path):
    """
    Plot EM convergence curve (log-likelihood vs iteration).
    
    Args:
        log_likelihoods (list): Log-likelihood at each iteration.
        save_path (str): Path to save the plot.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations = range(1, len(log_likelihoods) + 1)
    
    # Log-likelihood curve
    axes[0].plot(iterations, log_likelihoods, 'o-', color='#2C3E50',
               markersize=4, linewidth=1.5)
    axes[0].set_xlabel('EM Iteration')
    axes[0].set_ylabel('Log-Likelihood')
    axes[0].set_title('EM Convergence: Log-Likelihood vs Iteration')
    
    # Delta log-likelihood (convergence rate)
    if len(log_likelihoods) > 1:
        deltas = [abs(log_likelihoods[i] - log_likelihoods[i-1])
                  for i in range(1, len(log_likelihoods))]
        delta_iters = range(2, len(log_likelihoods) + 1)
        axes[1].semilogy(delta_iters, deltas, 'o-', color='#8E44AD',
                        markersize=4, linewidth=1.5)
        axes[1].axhline(y=1e-6, color='#E74C3C', linestyle='--', alpha=0.7,
                       label='TOL = 1e-6')
        axes[1].set_xlabel('EM Iteration')
        axes[1].set_ylabel('|Δ Log-Likelihood| (log scale)')
        axes[1].set_title('Convergence Rate')
        axes[1].legend()
    
    plt.suptitle('EM Algorithm Convergence', fontsize=13, fontweight='bold')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bic_aic(k_values, bics, aics, save_path):
    """
    Plot BIC and AIC vs number of components for model selection.
    
    Args:
        k_values (list): K values tested.
        bics (list): BIC values for each K.
        aics (list): AIC values for each K.
        save_path (str): Path to save the plot.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(k_values, bics, 'o-', color='#E74C3C', markersize=8,
            linewidth=2, label='BIC (lower = better)')
    ax.plot(k_values, aics, 's-', color='#3498DB', markersize=8,
            linewidth=2, label='AIC (lower = better)')
    
    # Mark optimal K
    best_k_bic = k_values[np.argmin(bics)]
    ax.axvline(x=best_k_bic, color='#2ECC71', linestyle=':', alpha=0.7,
               label=f'Best K (BIC) = {best_k_bic}')
    
    ax.set_xlabel('Number of Components (K)')
    ax.set_ylabel('Information Criterion')
    ax.set_title('Model Selection: BIC and AIC vs K')
    ax.set_xticks(k_values)
    ax.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
