"""
config.py — Global hyperparameters and file paths for GMM Iris project.

Every hyperparameter is justified by domain knowledge, mathematical theory,
and empirical observations about the Iris dataset.
"""
import os

# ─── File Paths ───
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "iris.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "iris_clean.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
LOGS_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

# ─── Clustering ───
# K=3: Iris dataset has 3 species (setosa, versicolor, virginica).
K = 3

# ─── EM Algorithm ───
# MAX_ITERS=100: EM on this dataset (150 points, 4D, K=3) converges in ~15-30
# iterations. 100 is a generous safety margin. Standard default in sklearn.
MAX_ITERS = 100

# TOL=1e-6: Convergence threshold for |delta log-likelihood|. Small enough for
# precise convergence, large enough to avoid floating-point noise issues.
# Well above float64 machine epsilon (~2.2e-16).
TOL = 1e-6

# REG_COVAR=1e-2: Added to diagonal of covariance matrices (epsilon*I) to prevent
# singularity when a component collapses onto a point. Set to 1e-2 to prevent
# overfitting/singularity on human rounding discretization artifacts.
REG_COVAR = 1e-2

# INIT_METHOD="kmeans": KMeans provides good initial means close to cluster centers,
# reducing EM iterations and avoiding poor local optima from random init.
# Required by instructor specification (Section 9).
INIT_METHOD = "kmeans"

# ─── K-Means ───
# Same iteration cap as EM for consistency. K-Means on 150 points with K=3
# converges in ~5-15 iterations.
KMEANS_MAX_ITERS = 100

# ─── KNN ───
# K_NEIGHBORS=5: 5 uses ~3.3% of
# dataset per prediction — local enough to respect boundaries, robust to noise.
# Standard default in KNN implementations.
K_NEIGHBORS = 5

# ─── Reproducibility ───
# Fixed seed ensures deterministic results across runs.
RANDOM_SEED = 42
