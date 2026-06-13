import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.em import fit_gmm
from src.kmeans import fit_kmeans
from src.visualization import _compute_ellipse_points

def main():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "iris_clean.csv")
    X = np.loadtxt(data_path, delimiter=',', skiprows=1)
    X = X[:, :2]  # Use only Sepal Length and Sepal Width for 2D visualization
    
    # 2. Fit K-Means
    km_labels, km_centroids = fit_kmeans(X, K=3, max_iters=100, seed=42)
    
    # 3. Fit GMM
    gmm_params, gmm_resp, _, _ = fit_gmm(X, K=3, max_iters=100, tol=1e-6, reg_covar=1e-6, init_method="kmeans", seed=42)
    gmm_labels = np.argmax(gmm_resp, axis=1)
    
    # Align labels by sorting centroids by x-coordinate
    km_order = np.argsort(km_centroids[:, 0])
    km_labels_new = np.zeros_like(km_labels)
    for new_idx, old_idx in enumerate(km_order):
        km_labels_new[km_labels == old_idx] = new_idx
    km_labels = km_labels_new
    km_centroids = km_centroids[km_order]
        
    gmm_order = np.argsort(gmm_params.means[:, 0])
    gmm_labels_new = np.zeros_like(gmm_labels)
    for new_idx, old_idx in enumerate(gmm_order):
        gmm_labels_new[gmm_labels == old_idx] = new_idx
    gmm_labels = gmm_labels_new
    gmm_params.means = gmm_params.means[gmm_order]
    gmm_params.covariances = gmm_params.covariances[gmm_order]
    gmm_params.weights = gmm_params.weights[gmm_order]
        
    # Highlight points where K-means and GMM disagree
    highlight_mask = (km_labels != gmm_labels)
    
    # Define colors
    c0 = '#3498DB' # Blue
    c1 = '#2ECC71' # Green
    c2 = '#9B59B6' # Purple
    c_highlight = '#E74C3C' # Red
    
    # Set up plot
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(25, 9))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.15)
    
    # Helper to plot base points
    def plot_base(ax, labels=None, alpha=0.3):
        if labels is None:
            ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=alpha, s=40, edgecolors='white', linewidths=0.5)
        else:
            ax.scatter(X[labels==0, 0], X[labels==0, 1], c=c0, alpha=alpha, s=40, edgecolors='white', linewidths=0.5)
            ax.scatter(X[labels==1, 0], X[labels==1, 1], c=c1, alpha=alpha, s=40, edgecolors='white', linewidths=0.5)
            ax.scatter(X[labels==2, 0], X[labels==2, 1], c=c2, alpha=alpha, s=40, edgecolors='white', linewidths=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(X[:,0].min()-0.5, X[:,0].max()+0.5)
        ax.set_ylim(X[:,1].min()-0.5, X[:,1].max()+0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('#BDC3C7')

    # ---------------------------------------------------------
    # Panel A: Raw Data
    # ---------------------------------------------------------
    axA = axes[0]
    plot_base(axA, labels=None, alpha=0.6)
    
    axA.set_title("Panel A — Iris Dataset\nRaw data (Sepal Length vs Sepal Width)", fontsize=18, fontweight='bold', pad=20)
    
    # ---------------------------------------------------------
    # Panel B: K-Means Result
    # ---------------------------------------------------------
    axB = axes[1]
    plot_base(axB, km_labels, alpha=0.4)
    
    # Centroids
    axB.scatter(km_centroids[:,0], km_centroids[:,1], c='black', marker='X', s=300, edgecolors='white', linewidths=2.5, zorder=5)
    
    # KMeans Distance Boundaries
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                         np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    dists = np.linalg.norm(grid[:, None, :] - km_centroids[None, :, :], axis=2)
    preds = np.argmin(dists, axis=1).reshape(xx.shape)
    axB.contour(xx, yy, preds, levels=[0.5, 1.5], colors='#34495E', linewidths=3, linestyles='--')
    
    # Highlight misassigned
    if np.any(highlight_mask):
        axB.scatter(X[highlight_mask, 0], X[highlight_mask, 1], c=c_highlight, s=150, marker='D', edgecolors='black', linewidths=1.5, zorder=6)
        
        valid_idxs = np.where(highlight_mask)[0]
        idx = valid_idxs[0]
        axB.annotate("Different Assignment\n(Euclidean distance)",
                     xy=(X[idx, 0], X[idx, 1]), xycoords='data',
                     xytext=(X[idx, 0]-1.5, X[idx, 1]+1.5), textcoords='data',
                     arrowprops=dict(arrowstyle="->", color='black', lw=2.5),
                     fontsize=13, fontweight='bold', color=c_highlight,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
                     
    axB.set_title("Panel B — K-Means Result\nDistance-based hard clustering", fontsize=18, fontweight='bold', pad=20)
    axB.text(0.5, 0.03, "K-Means divides the space rigidly based on distance", 
             ha='center', va='bottom', transform=axB.transAxes, fontsize=15, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='#34495E', boxstyle='round,pad=0.6', lw=2))
             
    # ---------------------------------------------------------
    # Panel C: GMM Result
    # ---------------------------------------------------------
    axC = axes[2]
    plot_base(axC, gmm_labels, alpha=0.4)
    
    # GMM ellipses
    for m, c, col in zip(gmm_params.means, gmm_params.covariances, [c0, c1, c2]):
        for nsig in [1.5, 3.0]:
            ell = _compute_ellipse_points(m, c, n_std=nsig)
            axC.plot(ell[:,0], ell[:,1], color=col, lw=3 if nsig==1.5 else 1.5, ls='-' if nsig==1.5 else '--')
            
    # GMM Centers
    axC.scatter(gmm_params.means[:,0], gmm_params.means[:,1], c='black', marker='+', s=350, linewidths=4, zorder=5)
    
    # Highlight same points
    if np.any(highlight_mask):
        axC.scatter(X[highlight_mask, 0], X[highlight_mask, 1], c='#27AE60', s=150, marker='D', edgecolors='black', linewidths=1.5, zorder=6)
        
        axC.annotate("Different Assignment\n(GMM respects shape)",
                     xy=(X[idx, 0], X[idx, 1]), xycoords='data',
                     xytext=(X[idx, 0]-1.5, X[idx, 1]+1.5), textcoords='data',
                     arrowprops=dict(arrowstyle="->", color='black', lw=2.5),
                     fontsize=13, fontweight='bold', color='#27AE60',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    
    axC.set_title("Panel C — GMM Result\nDensity-based probabilistic clustering", fontsize=18, fontweight='bold', pad=20)
    axC.text(0.5, 0.03, "GMM respects the covariance structure", 
             ha='center', va='bottom', transform=axC.transAxes, fontsize=15, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='#34495E', boxstyle='round,pad=0.6', lw=2))
             
    # ---------------------------------------------------------
    # Global Title & Caption
    # ---------------------------------------------------------
    fig.suptitle("Comparison of K-Means and GMM on Iris Dataset", fontsize=26, fontweight='black', y=0.98)
    
    caption = ("In the Iris dataset, the clusters have slightly different shapes and densities. "
               "The highlighted points show where K-Means and GMM disagree. "
               "GMM assigns boundary points differently because it models the cluster shapes rather than just distance.")
    fig.text(0.5, 0.08, caption, ha='center', va='top', fontsize=18, style='italic', color='#2C3E50',
             bbox=dict(facecolor='#F8F9F9', edgecolor='#BDC3C7', boxstyle='round,pad=1.0', lw=2))
             
    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "presentation_iris.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
