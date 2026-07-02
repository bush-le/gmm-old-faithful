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

def plot_cv_results(cv_results, save_path):
    """
    Plot cross-validation results (Log-Likelihood and Silhouette) across folds.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    folds = range(1, len(cv_results['ll_scores']) + 1)
    
    # Log-Likelihood
    axes[0].plot(folds, cv_results['ll_scores'], 'o-', color='#2980B9', linewidth=2, markersize=8)
    axes[0].axhline(cv_results['ll_mean'], color='#E74C3C', linestyle='--', 
                    label=f"Mean: {cv_results['ll_mean']:.4f}")
    axes[0].fill_between(folds, 
                         cv_results['ll_mean'] - cv_results['ll_std'], 
                         cv_results['ll_mean'] + cv_results['ll_std'], 
                         color='#E74C3C', alpha=0.2)
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Avg Log-Likelihood')
    axes[0].set_title('Cross-Validation: Log-Likelihood')
    axes[0].set_xticks(folds)
    axes[0].legend()
    
    # Silhouette Score
    axes[1].plot(folds, cv_results['sil_scores'], 's-', color='#27AE60', linewidth=2, markersize=8)
    axes[1].axhline(cv_results['sil_mean'], color='#E74C3C', linestyle='--',
                    label=f"Mean: {cv_results['sil_mean']:.4f}")
    axes[1].fill_between(folds, 
                         cv_results['sil_mean'] - cv_results['sil_std'], 
                         cv_results['sil_mean'] + cv_results['sil_std'], 
                         color='#E74C3C', alpha=0.2)
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Cross-Validation: Silhouette Score')
    axes[1].set_xticks(folds)
    axes[1].legend()
    
    plt.suptitle('K-Fold Cross Validation Results', fontsize=14, fontweight='bold')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_gmm_pdf_surface(X, params, save_path):
    """
    Plot the 3D surface and 2D contour of the fitted GMM probability density function.
    """
    setup_plot_style()
    from src.gaussian import gaussian_pdf_batch
    
    # Create a grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute PDF on grid
    Z = np.zeros(grid.shape[0])
    for k in range(params.K):
        Z += params.weights[k] * gaussian_pdf_batch(grid, params.means[k], params.covariances[k])
    Z = Z.reshape(xx.shape)
    
    fig = plt.figure(figsize=(14, 6))
    
    # 2D Contour
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(xx, yy, Z, 30, cmap='viridis', alpha=0.8)
    ax1.scatter(X[:, 0], X[:, 1], c='white', s=10, edgecolors='black', linewidths=0.5, alpha=0.5)
    ax1.set_xlabel('Eruption Duration (std)')
    ax1.set_ylabel('Waiting Time (std)')
    ax1.set_title('GMM PDF Contour')
    fig.colorbar(contour, ax=ax1)
    
    # 3D Surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(xx, yy, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax2.set_xlabel('Eruption Duration')
    ax2.set_ylabel('Waiting Time')
    ax2.set_zlabel('Density')
    ax2.set_title('GMM PDF 3D Surface')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_silhouette_profile(X, labels, save_path):
    """
    Plot the silhouette profile (score for each sample).
    """
    setup_plot_style()
    
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Compute distance matrix directly via broadcasting
    dist_matrix = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
    
    # Compute sample-wise silhouette scores
    sample_sil_scores = np.zeros(n_samples)
    for i in range(n_samples):
        label_i = labels[i]
        
        # a(i): average distance to same cluster
        same_cluster_mask = (labels == label_i)
        same_cluster_mask[i] = False
        if np.sum(same_cluster_mask) == 0:
            a_i = 0.0
        else:
            a_i = np.mean(dist_matrix[i, same_cluster_mask])
            
        # b(i): min average distance to other clusters
        b_i = np.inf
        for k in unique_labels:
            if k == label_i:
                continue
            other_cluster_mask = (labels == k)
            if np.sum(other_cluster_mask) == 0:
                continue
            avg_dist = np.mean(dist_matrix[i, other_cluster_mask])
            if avg_dist < b_i:
                b_i = avg_dist
                
        if np.isinf(b_i):
            sample_sil_scores[i] = 0.0
        else:
            sample_sil_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    # Sort scores within each cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    
    for i, k in enumerate(unique_labels):
        cluster_scores = sample_sil_scores[labels == k]
        cluster_scores.sort()
        size_cluster_i = cluster_scores.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_scores,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k+1))
        y_lower = y_upper + 10

    avg_score = np.mean(sample_sil_scores)
    ax.axvline(x=avg_score, color="red", linestyle="--", label=f"Average = {avg_score:.3f}")
    
    ax.set_title("Silhouette Profile for Each Sample")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.set_yticks([])
    ax.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_kmeans_result(X, labels, centroids, save_path):
    """
    Plot K-Means clustering result with centroids.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_labels = np.unique(labels)
    for k in unique_labels:
        mask = labels == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, s=30,
                   edgecolors='white', linewidths=0.5,
                   label=f'Cluster {k+1} (n={np.sum(mask)})')
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X',
                   s=200, edgecolors='white', linewidths=2, zorder=5,
                   label='Centroids')
    
    ax.set_xlabel('Eruption Duration (standardized)')
    ax.set_ylabel('Waiting Time (standardized)')
    ax.set_title('K-Means Clustering (Baseline)')
    ax.legend(loc='upper left', framealpha=0.9)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_gmm_decision_boundary(X, params, save_path):
    """
    Plot the GMM decision boundaries (hard assignment regions).
    """
    setup_plot_style()
    from src.gaussian import gaussian_pdf_batch
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    weighted_pdfs = np.zeros((grid.shape[0], params.K))
    for k in range(params.K):
        weighted_pdfs[:, k] = params.weights[k] * gaussian_pdf_batch(grid, params.means[k], params.covariances[k])
    
    Z = np.argmax(weighted_pdfs, axis=1)
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Custom colormap from CLUSTER_COLORS
    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap([c for c in CLUSTER_COLORS[:params.K]])
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)
    
    # Plot points
    responsibilities = np.zeros((X.shape[0], params.K))
    for k in range(params.K):
        responsibilities[:, k] = params.weights[k] * gaussian_pdf_batch(X, params.means[k], params.covariances[k])
    labels = np.argmax(responsibilities, axis=1)
    
    for k in range(params.K):
        mask = labels == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.8, s=20,
                   edgecolors='black', linewidths=0.5)
        # Plot centers
        ax.scatter(params.means[k][0], params.means[k][1], c='black', marker='*',
                   s=250, edgecolors='white', linewidths=1.5, zorder=5)
                   
    ax.set_xlabel('Eruption Duration (standardized)')
    ax.set_ylabel('Waiting Time (standardized)')
    ax.set_title('GMM Decision Boundaries')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_hierarchical_result(X, labels, save_path):
    """
    Plot Hierarchical Clustering result.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_labels = np.unique(labels)
    for k in unique_labels:
        mask = labels == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, s=30,
                   edgecolors='white', linewidths=0.5,
                   label=f'Cluster {k+1} (n={np.sum(mask)})')
    
    ax.set_xlabel('Eruption Duration (standardized)')
    ax.set_ylabel('Waiting Time (standardized)')
    ax.set_title('Hierarchical Clustering (Agglomerative)')
    ax.legend(loc='upper left', framealpha=0.9)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_knn_result(X, labels, save_path):
    """
    Plot KNN Clustering result.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_labels = np.unique(labels)
    for k in unique_labels:
        mask = labels == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, s=30,
                   edgecolors='white', linewidths=0.5,
                   label=f'Predicted {k+1} (n={np.sum(mask)})')
    
    ax.set_xlabel('Eruption Duration (standardized)')
    ax.set_ylabel('Waiting Time (standardized)')
    ax.set_title('KNN Predictions (Pseudo-labels from K-Means)')
    ax.legend(loc='upper left', framealpha=0.9)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_final_comparison(X, labels_gmm, labels_kmeans, labels_hc, save_path):
    """
    Side-by-side comparison of GMM, KMeans, and Hierarchical Clustering.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    titles = ['GMM', 'K-Means', 'Hierarchical']
    label_sets = [labels_gmm, labels_kmeans, labels_hc]
    
    for ax, title, labels in zip(axes, titles, label_sets):
        unique_labels = np.unique(labels)
        for k in unique_labels:
            mask = labels == k
            color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, s=20,
                       edgecolors='white', linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel('Eruption Duration')
        if ax == axes[0]:
            ax.set_ylabel('Waiting Time')
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
