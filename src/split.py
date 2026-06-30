"""
split.py — Train/test split utilities for GMM Old Faithful project.

Implements stratification-free random splitting for unsupervised density
estimation.  No scikit-learn — pure numpy only.

Reference: ML_PIPELINE_REFERENCE.md §10 — Train/Test Split & Data Leakage
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def train_test_split(X, test_ratio=0.2, seed=42):
    """Split feature matrix *X* into train and test sets.

    Uses a Fisher–Yates shuffle via ``np.random.default_rng`` for
    reproducible, unbiased permutation of row indices.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix to split.
    test_ratio : float, default 0.2
        Fraction of samples reserved for the test set.
    seed : int, default 42
        Seed for the random-number generator (reproducibility).

    Returns
    -------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test  : np.ndarray, shape (n_test, n_features)
    train_idx : np.ndarray of int, sorted
    test_idx  : np.ndarray of int, sorted

    Reference: ML_PIPELINE_REFERENCE.md §10
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    # Shuffle a copy of the index array
    indices = np.arange(n)
    rng.shuffle(indices)

    # Compute split point
    n_test = int(np.ceil(n * test_ratio))
    test_idx = np.sort(indices[:n_test])
    train_idx = np.sort(indices[n_test:])

    X_train = X[train_idx]
    X_test = X[test_idx]

    return X_train, X_test, train_idx, test_idx


def log_split_info(X_train, X_test, feature_names, logs_dir, plots_dir):
    """Log split diagnostics and create distribution-comparison plots.

    Writes a plain-text report to ``logs_dir/04_split.txt`` and saves a
    histogram overlay figure to ``plots_dir/04_split_distribution.png``.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test  : np.ndarray, shape (n_test, n_features)
    feature_names : list[str]
        Human-readable names for each column of *X*.
    logs_dir : str
        Directory for the text log file.
    plots_dir : str
        Directory for the histogram plot.

    Reference: ML_PIPELINE_REFERENCE.md §10
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_total = n_train + n_test
    n_features = X_train.shape[1]

    # ── Build text report ──────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("STEP 04 — Train / Test Split Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total samples : {n_total}")
    lines.append(f"Train samples : {n_train}  ({n_train / n_total * 100:.1f}%)")
    lines.append(f"Test  samples : {n_test}  ({n_test / n_total * 100:.1f}%)")
    lines.append("")

    # Per-feature distribution comparison
    lines.append("-" * 60)
    lines.append("Feature distribution comparison (train vs test):")
    lines.append("-" * 60)
    header = f"{'Feature':<20} {'Set':<6} {'Mean':>10} {'Std':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, name in enumerate(feature_names):
        tr_mean = np.mean(X_train[:, i])
        tr_std = np.std(X_train[:, i], ddof=1)
        te_mean = np.mean(X_test[:, i])
        te_std = np.std(X_test[:, i], ddof=1)
        lines.append(f"{name:<20} {'train':<6} {tr_mean:>10.4f} {tr_std:>10.4f}")
        lines.append(f"{'':<20} {'test':<6} {te_mean:>10.4f} {te_std:>10.4f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("⚠️  LEAKAGE BOUNDARY WARNING")
    lines.append("-" * 60)
    lines.append("All feature engineering, scaling, and model fitting must be")
    lines.append("computed on the TRAINING set only, then applied to the test")
    lines.append("set.  Violating this boundary inflates performance estimates")
    lines.append("and invalidates evaluation metrics.")
    lines.append("")

    report = "\n".join(lines)

    # ── Write log ──────────────────────────────────────────────────
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "04_split.txt")
    with open(log_path, "w") as f:
        f.write(report)

    # ── Print report to stdout as well ─────────────────────────────
    print(report)

    # ── Histogram overlay plot ─────────────────────────────────────
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 5))
    if n_features == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.hist(
            X_train[:, i],
            bins=20,
            alpha=0.6,
            label=f"Train (n={n_train})",
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.hist(
            X_test[:, i],
            bins=20,
            alpha=0.6,
            label=f"Test (n={n_test})",
            color="#DD8452",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        ax.set_title(f"{name} — Train vs Test Distribution")
        ax.legend()

    fig.suptitle("Train / Test Split — Distribution Comparison", fontsize=14, y=1.02)
    fig.tight_layout()

    plot_path = os.path.join(plots_dir, "04_split_distribution.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n📊 Distribution plot saved → {plot_path}")
    print(f"📝 Split log saved         → {log_path}")
