import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import os

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# 1. Load the data
data = np.genfromtxt('data/raw/iris.csv', delimiter=',', skip_header=1)
X = data[:, :2]  # Use only the first two features (Sepal Length and Sepal Width) for 2D visualization

# 2. Train K-Means and GMM
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Grid for plotting
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
XY = np.array([xx.ravel(), yy.ravel()]).T

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    from matplotlib.patches import Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(xy=position, width=nsig * width, height=nsig * height, angle=angle, **kwargs))

# --- Plot 1: Decision Boundary and Density Mesh ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1a: K-Means
Z_kmeans = kmeans.predict(XY).reshape(xx.shape)
axes[0].pcolormesh(xx, yy, Z_kmeans, cmap='Pastel1', alpha=0.5, shading='auto')
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='Set1', edgecolor='k', s=40)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                marker='x', s=200, linewidths=3, color='k', zorder=10, label='Centroids')
axes[0].set_title('K-Means: Hard Decision Boundary')
axes[0].set_xlabel('Sepal Length')
axes[0].set_ylabel('Sepal Width')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# Plot 1b: GMM
# Compute PDF
Z_gmm_pdf = np.exp(gmm.score_samples(XY)).reshape(xx.shape)
# Use a density/likelihood mesh
cs = axes[1].contourf(xx, yy, Z_gmm_pdf, levels=20, cmap='Blues', alpha=0.7)
plt.colorbar(cs, ax=axes[1], label='Probability Density')

axes[1].scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap='Set1', edgecolor='k', s=40)
for pos, cov in zip(gmm.means_, gmm.covariances_):
    draw_ellipse(pos, cov, ax=axes[1], alpha=0.8, color='red', fill=False, linewidth=1.5)
axes[1].set_title('GMM: Density Mesh & Confidence Ellipses')
axes[1].set_xlabel('Sepal Length')
axes[1].set_ylabel('Sepal Width')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('outputs/1_boundary_and_mesh.png', dpi=150)
plt.close()


# --- Plot 2: Soft Clustering Softness ---
plt.figure(figsize=(8, 6))
gamma_nk = gmm.predict_proba(X)[:, 0] 
# color ranges from 0 to 1
scatter = plt.scatter(X[:, 0], X[:, 1], c=gamma_nk, cmap='coolwarm', edgecolor='k', s=50, vmin=0, vmax=1)
plt.colorbar(scatter, label='Probability of Belonging to Cluster 1')
plt.title('GMM Soft Clustering: Posterior Probabilities')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('outputs/2_soft_clustering.png', dpi=150)
plt.close()


# --- Plot 3: Generative Sampling Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
# Original
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.7, color='gray', edgecolor='k')
axes[0].set_title('Original Data')
axes[0].set_xlabel('Sepal Length')
axes[0].set_ylabel('Sepal Width')

# Synthetic K-Means
labels = kmeans.labels_
synth_km_list = []
n_samples = 300
for k in range(kmeans.n_clusters):
    var_k = np.mean(np.var(X[labels==k], axis=0))
    n_k = int(n_samples * np.mean(labels==k))
    if k == kmeans.n_clusters - 1:
        n_k = n_samples - sum(len(x) for x in synth_km_list)
    synth = np.random.normal(loc=kmeans.cluster_centers_[k], scale=np.sqrt(var_k), size=(n_k, 2))
    synth_km_list.append(synth)
synth_km = np.vstack(synth_km_list)
axes[1].scatter(synth_km[:, 0], synth_km[:, 1], alpha=0.7, color='C0', edgecolor='k')
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='black', s=100, linewidth=2)
axes[1].set_title('Synthetic K-Means (Spherical Variance)')
axes[1].set_xlabel('Sepal Length')

# Synthetic GMM
synth_gmm, _ = gmm.sample(300)
axes[2].scatter(synth_gmm[:, 0], synth_gmm[:, 1], alpha=0.7, color='C2', edgecolor='k')
for pos, cov in zip(gmm.means_, gmm.covariances_):
    draw_ellipse(pos, cov, ax=axes[2], alpha=0.6, color='black', fill=False, linewidth=1.5)
axes[2].set_title('Synthetic GMM (Full Covariance)')
axes[2].set_xlabel('Sepal Length')

for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
plt.tight_layout()
plt.savefig('outputs/3_generative_sampling.png', dpi=150)
plt.close()


# --- Plot 4: Marginal Density Plots ---
plt.figure(figsize=(10, 6))
# Histogram of Sepal Width
sepal_width = X[:, 1]
plt.hist(sepal_width, bins=25, density=True, alpha=0.5, color='gray', edgecolor='k', label='Data Histogram')

# GMM Marginal Curve
x_eval = np.linspace(y_min, y_max, 500)
gmm_marginal_pdf = np.zeros_like(x_eval)

colors = ['C0', 'C1', 'C2']
for k in range(gmm.n_components):
    weight = gmm.weights_[k]
    mean = gmm.means_[k, 1]  # 1 is the index for 'Sepal Width'
    variance = gmm.covariances_[k, 1, 1]
    
    component_pdf = weight * norm.pdf(x_eval, mean, np.sqrt(variance))
    gmm_marginal_pdf += component_pdf
    
    plt.plot(x_eval, component_pdf, '--', color=colors[k], label=f'Gaussian {k+1} (Weight: {weight:.2f})')

plt.plot(x_eval, gmm_marginal_pdf, color='red', linewidth=2.5, label='GMM Total Density Curve')

plt.title('Marginal Density: Sepal Width')
plt.xlabel('Sepal Width')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('outputs/4_marginal_density.png', dpi=150)
plt.close()

print("Plots successfully generated in 'outputs/' directory.")
