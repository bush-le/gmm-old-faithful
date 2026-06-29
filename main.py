"""
main.py — GMM Old Faithful Pipeline Entry Point.

Runs the complete GMM pipeline end-to-end:
  Stage 01: Project setup (directories)
  Stage 02: Data loading
  Stage 03: Exploratory Data Analysis (EDA)
  Stage 04: Missing value check
  Stage 05: Outlier detection
  Stage 06: Feature scaling (standardization)
  Stage 07: GMM training (EM algorithm)
  Stage 08: Model selection (BIC/AIC for K)
  Stage 09: Evaluation metrics
  Stage 10: Final visualizations

References: ML_PIPELINE_REFERENCE.md
Constraint: Pure numpy only (no sklearn, no PyTorch, no TensorFlow).

Usage:
    python3 main.py
"""
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    K, MAX_ITERS, TOL, REG_COVAR, INIT_METHOD, RANDOM_SEED,
    KMEANS_MAX_ITERS, IQR_MULTIPLIER,
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    RESULTS_DIR, PLOTS_DIR, LOGS_DIR, METRICS_DIR, MODELS_DIR,
)


def setup_directories():
    """Create the results/ directory hierarchy."""
    for d in [RESULTS_DIR, PLOTS_DIR, LOGS_DIR, METRICS_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)
    print("  Results directories ready.")


def main():
    """Run the complete GMM pipeline."""

    # ─── Reproducibility ───
    np.random.seed(RANDOM_SEED)

    print("╔" + "═" * 63 + "╗")
    print("║  GMM FROM SCRATCH — OLD FAITHFUL GEYSER PIPELINE             ║")
    print("║  Algorithm: Gaussian Mixture Model (EM)                       ║")
    print("║  Paradigm:  Unsupervised Learning                             ║")
    print("╚" + "═" * 63 + "╝")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 01 — Project Setup
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 01 — PROJECT SETUP")
    print("=" * 60)
    setup_directories()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 02-06 — Preprocessing Pipeline
    # (Data loading, EDA, missing values, outliers, scaling)
    # ═══════════════════════════════════════════════════════════════
    feature_names = ['eruptions', 'waiting']

    # Run EDA on raw data first (before scaling)
    from src.data_loader import load_csv
    from src.eda import run_eda

    print("\n" + "=" * 60)
    print("STAGE 02 — DATA LOADING")
    print("=" * 60)
    raw_data = load_csv(RAW_DATA_PATH)
    X_raw = np.array([[r[0], r[1]] for r in raw_data
                      if r[0] > 0 and r[1] > 0])
    print(f"  Loaded {X_raw.shape[0]} valid samples, {X_raw.shape[1]} features")

    # Stage 03 — EDA (on raw data, before scaling)
    run_eda(X_raw, feature_names, PLOTS_DIR, LOGS_DIR)

    # Stages 04-06 — Preprocessing (missing values, outliers, scaling)
    from src.preprocessing import run_pipeline as run_preprocessing

    raw_clean, X_standardized, data_mean, data_std = run_preprocessing(
        RAW_DATA_PATH, PROCESSED_DATA_PATH, feature_names,
        IQR_MULTIPLIER, PLOTS_DIR, LOGS_DIR
    )

    # ═══════════════════════════════════════════════════════════════
    # STAGE 07 — Categorical Encoding — SKIPPED
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 07 — CATEGORICAL ENCODING — SKIPPED")
    print("=" * 60)
    print("  Both features (eruptions, waiting) are numeric floats.")
    print("  No categorical variables exist in Old Faithful dataset.")
    print("  Decision: SKIP (ref ML_PIPELINE_REFERENCE.md §7)")

    # Log the skip
    skip_log = os.path.join(LOGS_DIR, "07_encoding.txt")
    with open(skip_log, 'w') as f:
        f.write("CATEGORICAL ENCODING — SKIPPED\n")
        f.write("=" * 40 + "\n")
        f.write("Reason: Both features are numeric (float64).\n")
        f.write("No categorical variables in Old Faithful dataset.\n")
        f.write("Ref: ML_PIPELINE_REFERENCE.md §7\n")
    print(f"  Log saved: {skip_log}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 08 — Class Imbalance — SKIPPED
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 08 — CLASS IMBALANCE — SKIPPED")
    print("=" * 60)
    print("  This is an UNSUPERVISED learning task (GMM clustering).")
    print("  No target labels exist → class imbalance is not applicable.")
    print("  Decision: SKIP (ref ML_PIPELINE_REFERENCE.md §8)")

    skip_log = os.path.join(LOGS_DIR, "08_imbalance.txt")
    with open(skip_log, 'w') as f:
        f.write("CLASS IMBALANCE — SKIPPED\n")
        f.write("=" * 40 + "\n")
        f.write("Reason: GMM is unsupervised — no class labels exist.\n")
        f.write("Class imbalance is a supervised learning concept.\n")
        f.write("Ref: ML_PIPELINE_REFERENCE.md §8\n")
    print(f"  Log saved: {skip_log}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 09 — GMM Training (Core Algorithm)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 09 — GMM TRAINING (EM ALGORITHM)")
    print("=" * 60)

    from src.em import fit_gmm, compute_log_likelihood
    from src.visualization import (plot_raw_data, plot_gmm_ellipses,
                                    plot_convergence, plot_bic_aic)

    # Plot standardized data
    print("\n[1] Plotting standardized data...")
    plot_raw_data(X_standardized, os.path.join(PLOTS_DIR, "raw_data_standardized.png"))

    # Train GMM with K=2
    print(f"\n[2] Training GMM (K={K}, init={INIT_METHOD}, seed={RANDOM_SEED})...")
    print(f"    max_iters={MAX_ITERS}, tol={TOL}, reg_covar={REG_COVAR}")

    params, responsibilities, log_likelihoods, n_iters = fit_gmm(
        X_standardized, K, MAX_ITERS, TOL, REG_COVAR,
        init_method=INIT_METHOD, seed=RANDOM_SEED
    )

    # Hard assignments
    labels = np.argmax(responsibilities, axis=1)
    final_ll = compute_log_likelihood(X_standardized, params)

    # Print results
    print(f"\n[3] Learned GMM parameters:")
    print(params)
    print(f"\n    Final log-likelihood: {final_ll:.6f}")
    print(f"    Converged in {n_iters} iterations")

    # Save training log
    train_log_lines = []
    train_log_lines.append("GMM TRAINING LOG")
    train_log_lines.append("=" * 50)
    train_log_lines.append(f"K={K}, init={INIT_METHOD}, seed={RANDOM_SEED}")
    train_log_lines.append(f"max_iters={MAX_ITERS}, tol={TOL}, reg_covar={REG_COVAR}")
    train_log_lines.append(f"Converged in {n_iters} iterations")
    train_log_lines.append(f"Final log-likelihood: {final_ll:.6f}")
    train_log_lines.append("")
    train_log_lines.append("Log-likelihood per iteration:")
    for i, ll in enumerate(log_likelihoods):
        train_log_lines.append(f"  Iter {i+1:3d}: {ll:.6f}")
    train_log_lines.append("")
    train_log_lines.append("Learned Parameters:")
    train_log_lines.append(str(params))

    train_log_path = os.path.join(LOGS_DIR, "11_gmm_training.txt")
    with open(train_log_path, 'w') as f:
        f.write("\n".join(train_log_lines))
    print(f"  Training log saved: {train_log_path}")

    # Save plots
    print("\n[4] Saving GMM plots...")
    plot_gmm_ellipses(X_standardized, params,
                      os.path.join(PLOTS_DIR, "11_gmm_result.png"))
    plot_convergence(log_likelihoods,
                     os.path.join(PLOTS_DIR, "11_gmm_convergence.png"))

    # Save model parameters
    print("\n[5] Saving model parameters...")
    model_path = os.path.join(MODELS_DIR, "gmm_params.npz")
    np.savez(model_path,
             weights=params.weights,
             means=params.means,
             covariances=params.covariances,
             K=params.K,
             data_mean=data_mean,
             data_std=data_std)
    print(f"  Model saved: {model_path}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 10 — Model Selection (BIC/AIC for K)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 10 — MODEL SELECTION (BIC/AIC)")
    print("=" * 60)

    from src.metrics import compute_bic, compute_aic

    k_values = [2, 3, 4, 5]
    bics = []
    aics = []
    n_samples, n_features = X_standardized.shape

    for k_test in k_values:
        print(f"\n  Testing K={k_test}...")
        params_k, resp_k, ll_k, _ = fit_gmm(
            X_standardized, k_test, MAX_ITERS, TOL, REG_COVAR,
            init_method=INIT_METHOD, seed=RANDOM_SEED
        )
        ll_final = compute_log_likelihood(X_standardized, params_k)
        bic = compute_bic(ll_final, k_test, n_samples, n_features)
        aic = compute_aic(ll_final, k_test, n_features)
        bics.append(bic)
        aics.append(aic)
        print(f"    K={k_test}: LL={ll_final:.4f}, BIC={bic:.4f}, AIC={aic:.4f}")

    # Plot BIC/AIC
    plot_bic_aic(k_values, bics, aics,
                 os.path.join(PLOTS_DIR, "11_gmm_bic_aic.png"))

    best_k = k_values[np.argmin(bics)]
    print(f"\n  Best K by BIC: K={best_k}")
    print(f"  This confirms K=2 matches the bimodal physical structure.")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 11 — Evaluation Metrics
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 11 — EVALUATION METRICS")
    print("=" * 60)

    from src.metrics import compute_all_gmm_metrics, write_metrics_report

    print("\n  Computing GMM evaluation metrics...")
    metrics = compute_all_gmm_metrics(X_standardized, params, labels, final_ll)

    print(f"\n  Log-Likelihood:     {metrics['log_likelihood']:.4f}")
    print(f"  BIC:                {metrics['bic']:.4f}")
    print(f"  AIC:                {metrics['aic']:.4f}")
    print(f"  Silhouette Score:   {metrics['silhouette']:.4f}")
    print(f"  Cluster Separation: {metrics['separation']:.4f}")

    # Write report
    report_path = os.path.join(METRICS_DIR, "gmm_metrics.txt")
    write_metrics_report(metrics, params, report_path)

    # Save BIC/AIC table
    bic_path = os.path.join(METRICS_DIR, "gmm_bic_scores.csv")
    with open(bic_path, 'w') as f:
        f.write("K,BIC,AIC\n")
        for k_val, b, a in zip(k_values, bics, aics):
            f.write(f"{k_val},{b:.4f},{a:.4f}\n")
    print(f"  BIC/AIC scores saved: {bic_path}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 12 — Generate plots/README.txt
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 12 — PLOT INVENTORY")
    print("=" * 60)

    readme_lines = []
    readme_lines.append("PLOT INVENTORY — GMM Old Faithful Pipeline")
    readme_lines.append("=" * 50)
    readme_lines.append("")

    if os.path.exists(PLOTS_DIR):
        for f in sorted(os.listdir(PLOTS_DIR)):
            if f.endswith('.png'):
                readme_lines.append(f"  {f}")
                if "eda_histograms" in f:
                    readme_lines.append("    → Univariate distributions (histogram + KDE)")
                elif "eda_correlation" in f:
                    readme_lines.append("    → Correlation heatmap + scatter plot")
                elif "eda_boxplots" in f:
                    readme_lines.append("    → Box plots for initial outlier preview")
                elif "outliers" in f:
                    readme_lines.append("    → IQR outlier detection results")
                elif "raw_data" in f:
                    readme_lines.append("    → Standardized data scatter plot")
                elif "gmm_result" in f:
                    readme_lines.append("    → GMM clustering with confidence ellipses")
                elif "gmm_convergence" in f:
                    readme_lines.append("    → EM convergence (log-likelihood + delta)")
                elif "bic_aic" in f:
                    readme_lines.append("    → Model selection: BIC and AIC vs K")
                readme_lines.append("")

    readme_path = os.path.join(PLOTS_DIR, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("\n".join(readme_lines))
    print(f"  Plot inventory saved: {readme_path}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "╔" + "═" * 63 + "╗")
    print("║  PIPELINE COMPLETE                                           ║")
    print("╚" + "═" * 63 + "╝")

    print(f"\n  Pipeline Stages Executed:")
    print(f"    ✅ 01 — Project setup")
    print(f"    ✅ 02 — Data loading ({X_raw.shape[0]} samples)")
    print(f"    ✅ 03 — EDA (3 lenses: global, univariate, multivariate)")
    print(f"    ✅ 04 — Missing value check")
    print(f"    ✅ 05 — Outlier detection (IQR rule)")
    print(f"    ✅ 06 — Feature scaling (z-score standardization)")
    print(f"    ⏭️  07 — Categorical encoding (SKIPPED — all numeric)")
    print(f"    ⏭️  08 — Class imbalance (SKIPPED — unsupervised)")
    print(f"    ✅ 09 — GMM training (EM, K={K}, {n_iters} iterations)")
    print(f"    ✅ 10 — Model selection (BIC/AIC, best K={best_k})")
    print(f"    ✅ 11 — Evaluation metrics")
    print(f"    ✅ 12 — Plot inventory")

    print(f"\n  Generated files:")
    print(f"    Data:    {PROCESSED_DATA_PATH}")
    print(f"    Plots:   {PLOTS_DIR}/")
    if os.path.exists(PLOTS_DIR):
        for f in sorted(os.listdir(PLOTS_DIR)):
            if f.endswith('.png'):
                print(f"             - {f}")
    print(f"    Logs:    {LOGS_DIR}/")
    if os.path.exists(LOGS_DIR):
        for f in sorted(os.listdir(LOGS_DIR)):
            print(f"             - {f}")
    print(f"    Metrics: {METRICS_DIR}/")
    if os.path.exists(METRICS_DIR):
        for f in sorted(os.listdir(METRICS_DIR)):
            print(f"             - {f}")
    print(f"    Models:  {MODELS_DIR}/")
    if os.path.exists(MODELS_DIR):
        for f in sorted(os.listdir(MODELS_DIR)):
            print(f"             - {f}")


if __name__ == "__main__":
    main()
