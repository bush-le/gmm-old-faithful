import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial import Voronoi, voronoi_plot_2d

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/faithful_clean.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/plots/presentation")
DPI = 300

# Color palette
COLOR_C1 = "#2980B9"  # Blue for cluster 1
COLOR_C2 = "#C0392B"  # Red for cluster 2
COLOR_NEUTRAL = "#F39C12" # Yellow/Neutral
COLOR_BG = "#FDFEFE"
COLOR_TEXT = "#2C3E50"

def configure_plot_style():
    """Set academic and clean plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'axes.edgecolor': COLOR_TEXT,
        'text.color': COLOR_TEXT,
        'xtick.color': COLOR_TEXT,
        'ytick.color': COLOR_TEXT,
        'figure.facecolor': COLOR_BG,
        'axes.facecolor': COLOR_BG,
    })

def load_data():
    """Load the Old Faithful dataset."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_csv
    from config import PROCESSED_DATA_PATH
    raw_data = load_csv(PROCESSED_DATA_PATH)
    return np.array(raw_data)

def fit_kmeans(X, k=2):
    """Train K-Means and return model."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans

def fit_gmm(X, k=2):
    """Train Gaussian Mixture Model and return model."""
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X)
    return gmm

def get_aligned_labels(X, kmeans, gmm):
    """Align labels between K-Means and GMM to ensure color consistency."""
    labels_km = kmeans.predict(X)
    labels_gmm = gmm.predict(X)
    
    # Check if cluster 0 in K-Means is cluster 1 in GMM by checking centroids
    mean_km_0 = kmeans.cluster_centers_[0]
    mean_gmm_0 = gmm.means_[0]
    mean_gmm_1 = gmm.means_[1]
    
    dist_00 = np.linalg.norm(mean_km_0 - mean_gmm_0)
    dist_01 = np.linalg.norm(mean_km_0 - mean_gmm_1)
    
    # If kmeans cluster 0 is closer to gmm cluster 1, swap gmm labels
    swap_needed = dist_01 < dist_00
    
    if swap_needed:
        # We will dynamically map GMM components during plotting
        pass
        
    return swap_needed

def plot_figure_1(X, ax=None, save=True):
    """Figure 1: Raw Data Scatter Plot."""
    is_subplot = ax is not None
    if not is_subplot:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X[:, 0], X[:, 1], c='#7F8C8D', alpha=0.7, edgecolors='white', s=50)
    
    # Annotations
    ax.annotate('Cluster-like Region A\n(Short eruptions)', xy=(2.0, 50), xytext=(1.5, 65),
                arrowprops=dict(facecolor=COLOR_TEXT, shrink=0.05, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', ha='center')
    ax.annotate('Cluster-like Region B\n(Long eruptions)', xy=(4.5, 80), xytext=(5.0, 65),
                arrowprops=dict(facecolor=COLOR_TEXT, shrink=0.05, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', ha='center')
    ax.annotate('Overlap region', xy=(3.2, 65), xytext=(3.2, 50),
                arrowprops=dict(facecolor=COLOR_NEUTRAL, shrink=0.05, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', ha='center', color=COLOR_NEUTRAL)

    ax.set_xlabel('Eruptions (minutes)')
    ax.set_ylabel('Waiting (minutes)')
    ax.set_title('Figure 1: Old Faithful Raw Data')

    if not is_subplot and save:
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_raw_scatter.png'), dpi=DPI)
        plt.close()

def plot_figure_2(X, kmeans, ax=None, save=True):
    """Figure 2: K-Means Visualization (Decision Boundary)."""
    is_subplot = ax is not None
    if not is_subplot:
        fig, ax = plt.subplots(figsize=(8, 6))

    labels = kmeans.predict(X)
    centroids = kmeans.cluster_centers_

    # Plot points
    colors = np.where(labels == 0, COLOR_C1, COLOR_C2)
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, edgecolors='white', s=50)
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=150, label='Centroids')

    # Draw decision boundary (perpendicular bisector)
    # Using a dense meshgrid for plotting the boundary line
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, colors='black', linewidths=2, linestyles='dashed')

    # Annotations
    ax.annotate('Hard assignment', xy=(2.5, 80), fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_TEXT, lw=1))
    ax.annotate('Distance-based boundary', xy=(3.3, 75), xytext=(4.0, 55),
                arrowprops=dict(facecolor=COLOR_TEXT, shrink=0.05, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', ha='center')
    ax.annotate('Straight boundary\n(Assumes spherical clusters)', xy=(3.0, 60), xytext=(1.8, 55),
                arrowprops=dict(facecolor=COLOR_TEXT, shrink=0.05, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', ha='center')

    ax.set_xlabel('Eruptions (minutes)')
    ax.set_ylabel('Waiting (minutes)')
    ax.set_title('Figure 2: K-Means Rigid Partitions')
    ax.legend(loc='lower right')

    if not is_subplot and save:
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_kmeans.png'), dpi=DPI)
        plt.close()

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance."""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw 1 sigma and 2 sigma
    for n_sig in [1, 2]:
        ax.add_patch(Ellipse(position, n_sig * width, n_sig * height, angle=angle, **kwargs))

def plot_figure_3(X, gmm, swap_gmm, ax=None, save=True):
    """Figure 3: GMM Covariance Visualization."""
    is_subplot = ax is not None
    if not is_subplot:
        fig, ax = plt.subplots(figsize=(8, 6))

    labels_gmm = gmm.predict(X)
    if swap_gmm:
        labels_gmm = 1 - labels_gmm

    # Plot points
    colors = np.where(labels_gmm == 0, COLOR_C1, COLOR_C2)
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, edgecolors='white', s=50)

    # Plot ellipses
    means = gmm.means_
    covariances = gmm.covariances_
    
    for i in range(2):
        c_color = COLOR_C1 if (i == 0 and not swap_gmm) or (i == 1 and swap_gmm) else COLOR_C2
        ax.scatter(means[i, 0], means[i, 1], c='black', marker='X', s=150, zorder=10)
        draw_ellipse(means[i], covariances[i], ax=ax, alpha=0.2, color=c_color)
        draw_ellipse(means[i], covariances[i], ax=ax, alpha=0.8, color=c_color, fill=False, linewidth=2)

    # Annotations
    ax.annotate('Covariance-aware', xy=(4.5, 80), fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_TEXT, lw=1))
    ax.annotate('Elliptical density', xy=(4.8, 85), xytext=(5.5, 95),
                arrowprops=dict(facecolor=COLOR_TEXT, shrink=0.05, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', ha='center')
    ax.annotate('Captures feature correlation\n(cov > 0)', xy=(2.0, 55), xytext=(2.0, 40),
                arrowprops=dict(facecolor=COLOR_TEXT, shrink=0.05, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', ha='center')

    ax.set_xlabel('Eruptions (minutes)')
    ax.set_ylabel('Waiting (minutes)')
    ax.set_title('Figure 3: GMM Probabilistic Ellipses')

    if not is_subplot and save:
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figure3_gmm_covariance.png'), dpi=DPI)
        plt.close()

def plot_figure_4(X, kmeans, gmm, swap_gmm, ax=None, save=True):
    """Figure 4: Probability Heatmap + Model Disagreement."""
    is_subplot = ax is not None
    if not is_subplot:
        fig, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # GMM Probabilities
    probs = gmm.predict_proba(grid)
    prob_c0 = probs[:, 0] if not swap_gmm else probs[:, 1]
    
    # Create a custom colormap: C2 (Red) -> Yellow -> C1 (Blue)
    from matplotlib.colors import LinearSegmentedColormap
    # Reverse order: 0 prob of C1 means 100% prob of C2 (Red)
    cmap_custom = LinearSegmentedColormap.from_list("GMM_Heatmap", [COLOR_C2, COLOR_NEUTRAL, COLOR_C1])
    
    # Reshape and plot heatmap
    Z_prob = prob_c0.reshape(xx.shape)
    img = ax.imshow(Z_prob, extent=(x_min, x_max, y_min, y_max), origin='lower',
                    cmap=cmap_custom, alpha=0.5, aspect='auto')
    if not is_subplot:
        plt.colorbar(img, ax=ax, label='Probability of Blue Cluster')

    # K-Means Boundary
    Z_km = kmeans.predict(grid).reshape(xx.shape)
    ax.contour(xx, yy, Z_km, colors='black', linewidths=2, linestyles='dashed')

    # Predictions
    pred_km = kmeans.predict(X)
    pred_gmm = gmm.predict(X)
    if swap_gmm:
        pred_gmm = 1 - pred_gmm

    # Disagreement points
    disagreements = pred_km != pred_gmm
    X_agree = X[~disagreements]
    X_disagree = X[disagreements]

    # Plot agreeing points small and faded
    ax.scatter(X_agree[:, 0], X_agree[:, 1], c='black', alpha=0.3, s=15, edgecolors='none')
    
    # Plot disagreeing points prominently
    ax.scatter(X_disagree[:, 0], X_disagree[:, 1], c='gold', marker='D', s=100, edgecolors='black', 
               linewidths=1.5, label='Model Disagreement', zorder=5)

    # Annotations
    ax.annotate('Overlap / uncertainty region\n(Probabilities ≈ 0.5)', xy=(3.1, 70), xytext=(2.0, 85),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', ha='center')
    
    if len(X_disagree) > 0:
        ax.annotate('Model disagreement region', xy=(X_disagree[0, 0], X_disagree[0, 1]), xytext=(4.5, 50),
                    arrowprops=dict(facecolor='gold', shrink=0.05, width=1.5, headwidth=6),
                    fontsize=11, fontweight='bold', ha='center')

    ax.set_xlabel('Eruptions (minutes)')
    ax.set_ylabel('Waiting (minutes)')
    ax.set_title('Figure 4: Soft Clustering & Disagreement')
    ax.legend(loc='lower right')

    if not is_subplot and save:
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figure4_probability_heatmap.png'), dpi=DPI)
        plt.close()

def create_combined_figure(X, kmeans, gmm, swap_gmm):
    """Combine all 4 figures into one large presentation plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    plot_figure_1(X, ax=axes[0, 0], save=False)
    plot_figure_2(X, kmeans, ax=axes[0, 1], save=False)
    plot_figure_3(X, gmm, swap_gmm, ax=axes[1, 0], save=False)
    plot_figure_4(X, kmeans, gmm, swap_gmm, ax=axes[1, 1], save=False)
    
    fig.suptitle('Gaussian Mixture Models vs K-Means on Old Faithful', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'gmm_vs_kmeans_old_faithful_comparison.png'), dpi=DPI)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configure_plot_style()
    
    print("Loading data...")
    X = load_data()
    
    print("Fitting models...")
    kmeans = fit_kmeans(X)
    gmm = fit_gmm(X)
    
    swap_gmm = get_aligned_labels(X, kmeans, gmm)
    
    print("Generating Figure 1: Raw Scatter...")
    plot_figure_1(X)
    
    print("Generating Figure 2: K-Means...")
    plot_figure_2(X, kmeans)
    
    print("Generating Figure 3: GMM Covariance...")
    plot_figure_3(X, gmm, swap_gmm)
    
    print("Generating Figure 4: Probability Heatmap...")
    plot_figure_4(X, kmeans, gmm, swap_gmm)
    
    print("Generating Combined Presentation Figure...")
    create_combined_figure(X, kmeans, gmm, swap_gmm)
    
    print(f"Done! Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
