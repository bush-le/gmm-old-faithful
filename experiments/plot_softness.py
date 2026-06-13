import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from src.data_loader import load_csv

DATA_PATH = os.path.join(BASE_DIR, "data/processed/faithful_clean.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/plots/presentation")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X = np.array(load_csv(DATA_PATH))
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X)
    
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X)
    
    # Calculate uncertainties
    # K-means: hard assignment, max prob = 1, uncertainty = 0
    km_uncertainty = np.zeros(len(X))
    
    # GMM: soft assignment
    gmm_probs = gmm.predict_proba(X)
    gmm_uncertainty = 1.0 - np.max(gmm_probs, axis=1) # max is 0.5 (highest uncertainty)
    # normalize uncertainty from 0 to 1 for visual scale (1 - max_p)*2
    gmm_uncertainty_norm = gmm_uncertainty * 2.0
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot K-Means
    axes[0].scatter(X[:, 0], X[:, 1], c=km_uncertainty, cmap='Reds', vmin=0, vmax=1, 
                          s=20, alpha=0.5, edgecolors='gray')
    axes[0].set_title("K-Means: Mù quáng trước độ bất định\n(Uncertainty = 0 cho mọi điểm)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Eruptions (minutes)")
    axes[0].set_ylabel("Waiting (minutes)")
    
    axes[0].annotate('Mọi điểm đều chắc chắn 100%', xy=(2.5, 60), xytext=(2.0, 50),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                     fontsize=11, fontweight='bold')
    
    # Plot GMM
    # Create grid for GMM uncertainty background
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_probs = gmm.predict_proba(grid)
    grid_uncert = (1.0 - np.max(grid_probs, axis=1)) * 2.0
    Z = grid_uncert.reshape(xx.shape)
    
    # Contour heatmap
    axes[1].contourf(xx, yy, Z, levels=50, cmap='Reds', alpha=0.3)
    
    # Scatter points sizes mapped to uncertainty
    sizes = 20 + gmm_uncertainty_norm * 150
    sc2 = axes[1].scatter(X[:, 0], X[:, 1], c=gmm_uncertainty_norm, cmap='Reds', vmin=0, vmax=1, 
                          s=sizes, alpha=0.8, edgecolors='black')
    
    # Annotate high uncertainty
    axes[1].annotate('Vùng giao thoa\n(Độ bất định cao, điểm to và đỏ hơn)', xy=(3.0, 70), xytext=(1.8, 85),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                     fontsize=11, fontweight='bold')
                     
    axes[1].set_title("GMM: Đo lường được độ bất định\n(Soft Clustering)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Eruptions (minutes)")
    axes[1].set_ylabel("Waiting (minutes)")
    
    # Add colorbar
    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label('Chỉ số bất định (Uncertainty)', fontsize=12, fontweight='bold')
    
    plt.suptitle("Biểu đồ Soft Clustering: So sánh Khả năng Đo lường Độ Bất Định", fontsize=18, fontweight='bold', y=0.98)
    
    out_path = os.path.join(OUTPUT_DIR, "figure5_soft_clustering_softness.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
