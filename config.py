"""
config.py — Global hyperparameters and file paths for GMM Old Faithful project.

Every hyperparameter is justified by domain knowledge, mathematical theory,
and empirical observations about the Old Faithful dataset.

⚠️ No scikit-learn, PyTorch, or any pre-built ML class. Pure numpy only.
"""
import os

# ─── File Paths ───
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "faithful.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "faithful_clean.csv")

# ─── Output Directories (organized by type) ───
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# ─── Clustering ───
# K=2: Old Faithful has two eruption regimes (short ~2min + short wait ~55min,
# long ~4.5min + long wait ~80min). Bimodal structure is well-documented in
# geophysics literature. K=1 underfits, K>=3 overfits without physical basis.
K = 2

# ─── EM Algorithm ───
# MAX_ITERS=100: EM on this dataset (272 points, 2D, K=2) converges in ~15-30
# iterations. 100 is a generous safety margin.
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
INIT_METHOD = "kmeans"

# ─── K-Means (used for GMM initialization only) ───
# Same iteration cap as EM for consistency. K-Means on 272 points with K=2
# converges in ~5-15 iterations.
KMEANS_MAX_ITERS = 100

# ─── Outlier Detection ───
# IQR multiplier: standard 1.5x IQR rule for outlier detection
IQR_MULTIPLIER = 1.5

# ─── Reproducibility ───
# Fixed seed ensures deterministic results across runs.
RANDOM_SEED = 42
