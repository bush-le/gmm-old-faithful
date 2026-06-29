"""
visualization.py — Plotting functions using Matplotlib only.

Generates publication-quality visualizations for:
- Raw data scatter plots
- Cluster assignment results  
- GMM Gaussian ellipse overlays
- Side-by-side model comparison
- Convergence curves
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
    
    ax.set_xlabel('Sepal Length (standardized)')
    ax.set_ylabel('Sepal Width (standardized)')
    ax.set_title('Iris Dataset')
    
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
    
    ax.set_xlabel('Sepal Length (standardized)')
    ax.set_ylabel('Sepal Width (standardized)')
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
    # Transform: ellipse = eigenvectors @ diag(sqrt(eigenvalues) * n_std) @ circle
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
        ellipse_1 = _compute_ellipse_points(params.means[k][:2], params.covariances[k][:2, :2], n_std=1.0)
        ax.plot(ellipse_1[:, 0], ellipse_1[:, 1], color=color, linewidth=2, linestyle='-')
        
        # 2-sigma ellipse (~95% confidence)
        ellipse_2 = _compute_ellipse_points(params.means[k][:2], params.covariances[k][:2, :2], n_std=2.0)
        ax.plot(ellipse_2[:, 0], ellipse_2[:, 1], color=color, linewidth=1.5, linestyle='--')
        
        # Mark center
        ax.scatter(params.means[k][0], params.means[k][1], c=color, marker='+',
                  s=200, linewidths=3, zorder=5)
    
    ax.set_xlabel('Sepal Length (standardized)')
    ax.set_ylabel('Sepal Width (standardized)')
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
    fig, ax = plt.subplots()
    
    iterations = range(1, len(log_likelihoods) + 1)
    ax.plot(iterations, log_likelihoods, 'o-', color='#2C3E50',
           markersize=4, linewidth=1.5)
    
    ax.set_xlabel('EM Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('EM Convergence: Log-Likelihood vs Iteration')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(X, results_dict, save_path):
    """
    Create side-by-side comparison of all clustering methods.
    
    Args:
        X (numpy.ndarray): Data of shape (N, 2).
        results_dict (dict): Maps method name to labels array.
        save_path (str): Path to save the plot.
    """
    setup_plot_style()
    n_methods = len(results_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (name, labels) in zip(axes, results_dict.items()):
        unique_labels = np.unique(labels)
        for k in unique_labels:
            mask = labels == k
            color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, s=15,
                      edgecolors='white', linewidths=0.3,
                      label=f'Cluster {k+1}')
        
        ax.set_xlabel('Sepal Length (std)')
        ax.set_ylabel('Sepal Width (std)')
        ax.set_title(name)
        ax.legend(loc='upper left', fontsize=9)
    
    plt.suptitle('Clustering Method Comparison', fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
