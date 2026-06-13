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
    # 1. Generate Synthetic Data
    np.random.seed(42)
    # Cluster 0: Elongated, positive correlation
    mean0 = np.array([-1.0, -1.0])
    cov0 = np.array([[6.0, 4.5], [4.5, 5.0]])
    X0 = np.random.multivariate_normal(mean0, cov0, 350)
    
    # Cluster 1: Elongated, different correlation (more compact, rotated)
    mean1 = np.array([4.0, -2.0])
    cov1 = np.array([[2.5, -1.5], [-1.5, 3.0]])
    X1 = np.random.multivariate_normal(mean1, cov1, 350)
    
    X = np.vstack([X0, X1])
    y_true = np.concatenate([np.zeros(350), np.ones(350)]).astype(int)
    
    # 2. Fit K-Means
    km_labels, km_centroids = fit_kmeans(X, K=2, max_iters=100, seed=42)
    
    # 3. Fit GMM
    gmm_params, gmm_resp, _, _ = fit_gmm(X, K=2, max_iters=100, tol=1e-6, reg_covar=1e-6, init_method="kmeans", seed=42)
    gmm_labels = np.argmax(gmm_resp, axis=1)
    
    # Align K-Means labels with y_true
    if np.linalg.norm(km_centroids[0] - mean0) > np.linalg.norm(km_centroids[0] - mean1):
        km_labels = 1 - km_labels
        km_centroids = km_centroids[::-1]
        
    # Align GMM labels with y_true
    if np.linalg.norm(gmm_params.means[0] - mean0) > np.linalg.norm(gmm_params.means[0] - mean1):
        gmm_labels = 1 - gmm_labels
        gmm_params.means = gmm_params.means[::-1]
        gmm_params.covariances = gmm_params.covariances[::-1]
        gmm_params.weights = gmm_params.weights[::-1]
        
    # Identify misassigned points in K-Means that GMM gets right
    km_correct = (km_labels == y_true)
    gmm_correct = (gmm_labels == y_true)
    
    highlight_mask = (~km_correct) & gmm_correct
    
    # Define colors
    c0 = '#3498DB' # Blue
    c1 = '#2ECC71' # Green
    c_highlight = '#E74C3C' # Red
    
    # Set up plot
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(25, 9))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.15)
    
    # Helper to plot base points
    def plot_base(ax, labels, alpha=0.3):
        ax.scatter(X[labels==0, 0], X[labels==0, 1], c=c0, alpha=alpha, s=40, edgecolors='white', linewidths=0.5)
        ax.scatter(X[labels==1, 0], X[labels==1, 1], c=c1, alpha=alpha, s=40, edgecolors='white', linewidths=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(X[:,0].min()-1, X[:,0].max()+1)
        ax.set_ylim(X[:,1].min()-1, X[:,1].max()+1)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('#BDC3C7')

    # ---------------------------------------------------------
    # Panel A: True Structure
    # ---------------------------------------------------------
    axA = axes[0]
    plot_base(axA, y_true, alpha=0.6)
    
    # Draw true ellipses
    for m, c, col in zip([mean0, mean1], [cov0, cov1], [c0, c1]):
        for nsig in [1.5, 3.0]:
            ell = _compute_ellipse_points(m, c, n_std=nsig)
            axA.plot(ell[:,0], ell[:,1], color=col, lw=3 if nsig==1.5 else 1.5, ls='-' if nsig==1.5 else '--')
    
    # Shade overlap region
    overlap_center = (mean0 + mean1) / 2 + np.array([-0.5, 1.5])
    overlap_circle = plt.Circle(overlap_center, 3.5, color='#F1C40F', alpha=0.25, zorder=0)
    axA.add_patch(overlap_circle)
    axA.text(overlap_center[0], overlap_center[1] + 4.2, "Overlap Region", 
             ha='center', va='center', fontsize=15, fontweight='bold', color='#D4AC0D',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#F1C40F', boxstyle='round,pad=0.4', lw=2))
             
    axA.set_title("Panel A — Raw Dataset + True Structure\nOverlapping elliptical Gaussian clusters", fontsize=18, fontweight='bold', pad=20)
    
    # ---------------------------------------------------------
    # Panel B: K-Means Result
    # ---------------------------------------------------------
    axB = axes[1]
    plot_base(axB, km_labels, alpha=0.2)
    
    # Centroids
    axB.scatter(km_centroids[:,0], km_centroids[:,1], c='black', marker='X', s=300, edgecolors='white', linewidths=2.5, zorder=5)
    
    # Perpendicular bisector
    mid = (km_centroids[0] + km_centroids[1]) / 2
    dir_vec = km_centroids[1] - km_centroids[0]
    perp_vec = np.array([-dir_vec[1], dir_vec[0]])
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    t = np.linspace(-15, 15, 100)
    line = mid[:, None] + perp_vec[:, None] * t[None, :]
    axB.plot(line[0], line[1], color='#34495E', lw=3, ls='--', label="Distance Boundary")
    
    # Highlight misassigned
    axB.scatter(X[highlight_mask, 0], X[highlight_mask, 1], c=c_highlight, s=150, marker='D', edgecolors='black', linewidths=1.5, zorder=6)
    
    if np.any(highlight_mask):
        # Find a good point to annotate
        valid_idxs = np.where(highlight_mask)[0]
        # pick a point that is visibly high up in the overlap
        idx = valid_idxs[np.argmax(X[valid_idxs, 1])]
        axB.annotate("Misassigned by K-Means\n(Euclidean distance ignores cluster shape)",
                     xy=(X[idx, 0], X[idx, 1]), xycoords='data',
                     xytext=(X[idx, 0]-6, X[idx, 1]+5), textcoords='data',
                     arrowprops=dict(arrowstyle="->", color='black', lw=2.5),
                     fontsize=13, fontweight='bold', color=c_highlight,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
                     
    axB.set_title("Panel B — K-Means Result\nDistance-based hard clustering", fontsize=18, fontweight='bold', pad=20)
    axB.text(0.5, 0.03, "K-Means uses hard assignment based only on distance", 
             ha='center', va='bottom', transform=axB.transAxes, fontsize=15, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='#34495E', boxstyle='round,pad=0.6', lw=2))
             
    # ---------------------------------------------------------
    # Panel C: GMM Result
    # ---------------------------------------------------------
    axC = axes[2]
    plot_base(axC, gmm_labels, alpha=0.2)
    
    # GMM ellipses
    for m, c, col in zip(gmm_params.means, gmm_params.covariances, [c0, c1]):
        for nsig in [1.5, 3.0]:
            ell = _compute_ellipse_points(m, c, n_std=nsig)
            axC.plot(ell[:,0], ell[:,1], color=col, lw=3 if nsig==1.5 else 1.5, ls='-' if nsig==1.5 else '--')
            
    # GMM Centers
    axC.scatter(gmm_params.means[:,0], gmm_params.means[:,1], c='black', marker='+', s=350, linewidths=4, zorder=5)
    
    # Highlight same points (GMM got them right!)
    axC.scatter(X[highlight_mask, 0], X[highlight_mask, 1], c='#27AE60', s=150, marker='D', edgecolors='black', linewidths=1.5, zorder=6)
    
    if np.any(highlight_mask):
        axC.annotate("Correctly assigned by GMM\n(respects covariance structure)",
                     xy=(X[idx, 0], X[idx, 1]), xycoords='data',
                     xytext=(X[idx, 0]-6, X[idx, 1]+5), textcoords='data',
                     arrowprops=dict(arrowstyle="->", color='black', lw=2.5),
                     fontsize=13, fontweight='bold', color='#27AE60',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
                     
    axC.set_title("Panel C — GMM Result\nDensity-based probabilistic clustering", fontsize=18, fontweight='bold', pad=20)
    axC.text(0.5, 0.03, "GMM uses probability density + covariance structure", 
             ha='center', va='bottom', transform=axC.transAxes, fontsize=15, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='#34495E', boxstyle='round,pad=0.6', lw=2))
             
    # ---------------------------------------------------------
    # Global Title & Caption
    # ---------------------------------------------------------
    fig.suptitle("Comparison of K-Means and Gaussian Mixture Model (GMM)", fontsize=26, fontweight='black', y=0.98)
    
    caption = ("K-Means misclassifies boundary points because it partitions space using distance alone, "
               "while GMM models cluster shape and covariance, producing more realistic assignments in overlap regions.")
    fig.text(0.5, 0.08, caption, ha='center', va='top', fontsize=18, style='italic', color='#2C3E50',
             bbox=dict(facecolor='#F8F9F9', edgecolor='#BDC3C7', boxstyle='round,pad=1.0', lw=2))
             
    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "presentation_kmeans_vs_gmm.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
