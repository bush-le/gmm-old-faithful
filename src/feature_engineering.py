"""
feature_engineering.py — Feature engineering for Old Faithful GMM pipeline.

Implements ML_PIPELINE_REFERENCE.md §9.9 (Feature Engineering Strategies):
  - Interaction features: eruptions × waiting (captures relationship structure)
  - Polynomial features: eruptions² (captures nonlinear geometric patterns)

These engineered features can help the GMM capture more nuanced cluster structure
beyond what the two original features provide.

Design Rationale (§9.9):
  - Interaction feature (eruptions × waiting): Old Faithful exhibits a strong
    positive correlation between eruption duration and waiting time. The product
    term captures this multiplicative relationship, helping separate the two
    eruption regimes (short-short vs long-long) more cleanly.
  - Polynomial feature (eruptions²): The eruption duration shows bimodal behavior
    with clusters at ~2min and ~4.5min. Squaring amplifies the separation between
    these modes (4 vs 20.25), making cluster boundaries more distinct.

⚠️ Pure numpy + matplotlib only. No pandas, no scipy, no sklearn.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def create_engineered_features(X, feature_names):
    """
    Create engineered features from original Old Faithful data.

    Implements §9.9 feature engineering strategies:
      1. Interaction feature: eruptions × waiting
         - Captures multiplicative relationship between the two variables
         - Short eruptions with short waits → small product (~110)
         - Long eruptions with long waits → large product (~360)
         - This amplifies the separation between the two regimes

      2. Polynomial feature: eruptions²
         - Captures nonlinear pattern in eruption duration
         - Short eruptions (~2min): eruptions² ≈ 4
         - Long eruptions (~4.5min): eruptions² ≈ 20.25
         - Squaring amplifies the ~2.5x gap to ~5x, aiding cluster separation

    Args:
        X (numpy.ndarray): Original data of shape (N, 2) with columns
            [eruptions, waiting].
        feature_names (list): Original feature names, e.g. ['eruptions', 'waiting'].

    Returns:
        tuple: (X_augmented, augmented_names) where:
            - X_augmented (numpy.ndarray): Augmented data of shape (N, 4)
              with columns [eruptions, waiting, eruptions×waiting, eruptions²].
            - augmented_names (list): List of 4 feature names.
    """
    n_samples = X.shape[0]

    # Interaction feature: eruptions × waiting (§9.9 — relationship feature)
    interaction = X[:, 0] * X[:, 1]  # (N,)

    # Polynomial feature: eruptions² (§9.9 — geometric feature)
    eruptions_sq = X[:, 0] ** 2  # (N,)

    # Stack into augmented array: [eruptions, waiting, eruptions×waiting, eruptions²]
    X_augmented = np.column_stack([X, interaction, eruptions_sq])

    # Build augmented feature names
    augmented_names = list(feature_names) + [
        f'{feature_names[0]}×{feature_names[1]}',
        f'{feature_names[0]}²',
    ]

    print(f"  Feature engineering: {X.shape[1]} → {X_augmented.shape[1]} features")
    print(f"  New features: {augmented_names[2:]}")

    return X_augmented, augmented_names


def log_feature_engineering(X_train, X_train_aug, feature_names, aug_names,
                            logs_dir, plots_dir):
    """
    Log feature engineering decisions and save distribution plots.

    Documents each engineered feature with:
      - Mathematical formula
      - Domain justification (why it helps for Old Faithful)
      - Summary statistics (mean, std, min, max)
      - Distribution comparison plot

    Outputs:
      - Log file: {logs_dir}/10_feature_engineering.txt
      - Plot:     {plots_dir}/10_engineered_features.png

    Reference: ML_PIPELINE_REFERENCE.md §9.9

    Args:
        X_train (numpy.ndarray): Original training data of shape (N, 2).
        X_train_aug (numpy.ndarray): Augmented training data of shape (N, 4).
        feature_names (list): Original feature names (length 2).
        aug_names (list): Augmented feature names (length 4).
        logs_dir (str): Directory for log files.
        plots_dir (str): Directory for plot files.
    """
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    n_samples = X_train_aug.shape[0]
    n_original = X_train.shape[1]
    n_augmented = X_train_aug.shape[1]

    # ── Build log content ──
    log_lines = []
    log_lines.append("FEATURE ENGINEERING")
    log_lines.append("=" * 60)
    log_lines.append(f"Reference: ML_PIPELINE_REFERENCE.md §9.9")
    log_lines.append(f"Dataset: {n_samples} training samples")
    log_lines.append(f"Original features: {n_original} → Augmented features: {n_augmented}")
    log_lines.append("")

    # Justification for each engineered feature
    log_lines.append("ENGINEERED FEATURES")
    log_lines.append("-" * 60)
    log_lines.append("")

    log_lines.append(f"1. {aug_names[2]}  (Interaction Feature)")
    log_lines.append(f"   Formula:        {feature_names[0]} × {feature_names[1]}")
    log_lines.append(f"   Type:           Relationship / Interaction (§9.9)")
    log_lines.append(f"   Justification:  Old Faithful's two eruption regimes exhibit")
    log_lines.append(f"                   a strong positive correlation (r ≈ 0.90).")
    log_lines.append(f"                   The product term captures the multiplicative")
    log_lines.append(f"                   relationship: short eruptions pair with short")
    log_lines.append(f"                   waits (product ~110), long eruptions pair with")
    log_lines.append(f"                   long waits (product ~360). This creates a")
    log_lines.append(f"                   single feature that encodes the regime identity.")
    log_lines.append("")

    log_lines.append(f"2. {aug_names[3]}  (Polynomial Feature)")
    log_lines.append(f"   Formula:        {feature_names[0]}²")
    log_lines.append(f"   Type:           Geometric / Polynomial (§9.9)")
    log_lines.append(f"   Justification:  Eruption duration is bimodal (~2min vs ~4.5min).")
    log_lines.append(f"                   Squaring amplifies the gap between modes:")
    log_lines.append(f"                   2² = 4  vs  4.5² = 20.25 (5× separation vs 2.25×).")
    log_lines.append(f"                   This makes cluster boundaries more distinct")
    log_lines.append(f"                   for the GMM to detect.")
    log_lines.append("")

    # Summary statistics for all features
    log_lines.append("FEATURE STATISTICS")
    log_lines.append("-" * 60)
    header = f"{'Feature':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
    log_lines.append(header)
    log_lines.append("-" * 65)

    for i, name in enumerate(aug_names):
        col = X_train_aug[:, i]
        log_lines.append(
            f"{name:<25} {np.mean(col):>10.4f} {np.std(col):>10.4f} "
            f"{np.min(col):>10.4f} {np.max(col):>10.4f}"
        )

    log_lines.append("")

    # Correlation of engineered features with originals
    log_lines.append("CORRELATION WITH ORIGINAL FEATURES")
    log_lines.append("-" * 60)
    for i in range(n_original, n_augmented):
        eng_col = X_train_aug[:, i]
        for j in range(n_original):
            orig_col = X_train_aug[:, j]
            # Manual Pearson correlation
            mean_e = np.mean(eng_col)
            mean_o = np.mean(orig_col)
            std_e = np.std(eng_col)
            std_o = np.std(orig_col)
            if std_e < 1e-10 or std_o < 1e-10:
                r = 0.0
            else:
                r = np.mean((eng_col - mean_e) * (orig_col - mean_o)) / (std_e * std_o)
            log_lines.append(f"  r({aug_names[i]}, {aug_names[j]}) = {r:.4f}")
    log_lines.append("")

    log_lines.append("NOTE: Engineered features are computed on raw (unscaled) data.")
    log_lines.append("They should be standardized before use in the GMM pipeline.")

    # Save log
    log_path = os.path.join(logs_dir, "10_feature_engineering.txt")
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Log saved: {log_path}")

    # ── Distribution plots ──
    n_features = n_augmented
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']

    for i in range(n_features):
        ax = axes[i]
        col = X_train_aug[:, i]
        name = aug_names[i]

        # Histogram
        ax.hist(col, bins=25, density=True, alpha=0.6, color=colors[i],
                edgecolor='white', linewidth=0.5, label='Histogram')

        # Manual KDE (Gaussian kernel, Silverman's rule)
        std = np.std(col)
        q75, q25 = np.percentile(col, [75, 25])
        iqr = q75 - q25
        bw = 0.9 * min(std, iqr / 1.34) * len(col) ** (-0.2)
        bw = max(bw, 1e-6)

        x_grid = np.linspace(np.min(col) - 0.5 * std,
                             np.max(col) + 0.5 * std, 200)
        kde_vals = np.zeros_like(x_grid)
        for xi in col:
            u = (x_grid - xi) / bw
            kde_vals += np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        kde_vals /= (len(col) * bw)

        ax.plot(x_grid, kde_vals, color='#2C3E50', linewidth=2, label='KDE')

        # Annotate stats
        mean_val = np.mean(col)
        ax.axvline(mean_val, color='#E74C3C', linestyle='--', linewidth=1.5,
                   label=f'Mean={mean_val:.2f}')

        # Mark if engineered
        if i >= n_original:
            ax.set_title(f'{name} (ENGINEERED)\n'
                         f'mean={mean_val:.2f}, std={std:.2f}',
                         fontweight='bold')
        else:
            ax.set_title(f'{name} (ORIGINAL)\n'
                         f'mean={mean_val:.2f}, std={std:.2f}')

        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Feature Engineering — Original vs Engineered Distributions\n'
                 '(ML_PIPELINE_REFERENCE.md §9.9)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(plots_dir, "10_engineered_features.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")
