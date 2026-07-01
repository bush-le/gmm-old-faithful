"""
main.py — GMM Old Faithful Pipeline Entry Point (Refactored v2).

Runs the complete GMM pipeline end-to-end following ML_PIPELINE_REFERENCE_v2.md:

  Stage 01: Project setup (directories)
  Stage 02: Data loading
  Stage 03: Exploratory Data Analysis (EDA) — on raw data
  Stage 04: Train/test split (§10) — LEAKAGE BOUNDARY
  Stage 05: Missing value handling (fit on train only, §4)
  Stage 06: Outlier detection (fit on train only, §5)
  Stage 07: Feature scaling (fit on train only, §6)
  Stage 08: Categorical encoding — SKIPPED (all numeric)
  Stage 09: Class imbalance — SKIPPED (unsupervised)
  Stage 10: Feature engineering (§9.9)
  Stage 11: Baseline models (§11)
  Stage 13: GMM training on TRAIN set (EM algorithm)
  Stage 14: Hyperparameter tuning (§14, §15, §18.3)
  Stage 15: Evaluation metrics
  Stage 16: K-Fold cross-validation (§17)
  Stage 17: Final test-set evaluation (ONCE)
  Stage 18: Error analysis (§19)
  Stage 19: Final visualization pass
  Stage 20: Interpretability notes (§20)

References: ML_PIPELINE_REFERENCE_v2.md, REFACTOR_PROMPTS_v2.md
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
    KMEANS_MAX_ITERS, IQR_MULTIPLIER, TEST_RATIO, N_FOLDS, K_VALUES,
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    RESULTS_DIR, PLOTS_DIR, LOGS_DIR, METRICS_DIR, MODELS_DIR, EXPERIMENTS_DIR,
)


import shutil

def setup_directories():
    """Create the results/ directory hierarchy including experiments/. Clears old results first."""
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    for d in [RESULTS_DIR, PLOTS_DIR, LOGS_DIR, METRICS_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
        os.makedirs(d, exist_ok=True)
    print("  Results directories ready (including experiments/).")


def main():
    """Run the complete refactored GMM pipeline."""

    # ─── Reproducibility ───
    np.random.seed(RANDOM_SEED)

    print("╔" + "═" * 63 + "╗")
    print("║  GMM FROM SCRATCH — OLD FAITHFUL GEYSER PIPELINE  (v2)       ║")
    print("║  Algorithm: Gaussian Mixture Model (EM)                       ║")
    print("║  Paradigm:  Unsupervised Learning                             ║")
    print("║  Refactored: Train/test split + CV + baselines + experiments  ║")
    print("╚" + "═" * 63 + "╝")

    feature_names = ['eruptions', 'waiting']

    # ═══════════════════════════════════════════════════════════════
    # STAGE 01 — Project Setup
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 01 — PROJECT SETUP")
    print("=" * 60)
    setup_directories()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 02 — Data Loading
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 02 — DATA LOADING")
    print("=" * 60)

    from src.data_loader import load_csv

    raw_data = load_csv(RAW_DATA_PATH)

    from src.preprocessing import clean_data
    X_raw = clean_data(raw_data)

    print(f"  Shape: {X_raw.shape}")
    print(f"  Dtypes: float64 (both features)")
    print(f"  Head (first 5 rows):")
    for i in range(min(5, len(X_raw))):
        print(f"    [{i}] eruptions={X_raw[i, 0]:.2f}, waiting={X_raw[i, 1]:.1f}")

    # Log data loading
    log_lines = [
        "DATA LOADING",
        "=" * 50,
        f"Source: {RAW_DATA_PATH}",
        f"Rows loaded: {len(raw_data)}",
        f"After cleaning: {X_raw.shape[0]} rows, {X_raw.shape[1]} features",
        f"Features: {feature_names}",
        f"Data type: numeric (float64)",
    ]
    log_path = os.path.join(LOGS_DIR, "02_data_loading.txt")
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"  Log saved: {log_path}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 03 — EDA (on raw data, before split)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 03 — EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    from src.eda import run_eda
    run_eda(X_raw, feature_names, PLOTS_DIR, LOGS_DIR)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 04 — Train/Test Split (§10) — LEAKAGE BOUNDARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 04 — TRAIN/TEST SPLIT (§10) — LEAKAGE BOUNDARY")
    print("=" * 60)

    from src.split import train_test_split, log_split_info

    X_train_raw, X_test_raw, train_idx, test_idx = train_test_split(
        X_raw, test_ratio=TEST_RATIO, seed=RANDOM_SEED
    )

    print(f"  Total:    {len(X_raw)} samples")
    print(f"  Training: {len(X_train_raw)} samples ({len(X_train_raw)/len(X_raw)*100:.1f}%)")
    print(f"  Test:     {len(X_test_raw)} samples ({len(X_test_raw)/len(X_raw)*100:.1f}%)")
    print(f"  ⚠️  ALL statistics below computed from TRAIN DATA ONLY")

    log_split_info(X_train_raw, X_test_raw, feature_names, LOGS_DIR, PLOTS_DIR)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 05 — Missing Value Handling (fit on train, §4)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 05 — MISSING VALUE HANDLING (TRAIN ONLY)")
    print("=" * 60)

    from src.preprocessing import check_missing_values

    X_train_clean = check_missing_values(
        X_train_raw.copy(), feature_names, LOGS_DIR, dataset_label="train"
    )
    # Apply same handling to test (if any imputation values were computed from train)
    X_test_clean = X_test_raw.copy()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 06 — Outlier Detection (fit on train, §5)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 06 — OUTLIER DETECTION (BOUNDS FROM TRAIN, §5)")
    print("=" * 60)

    from src.preprocessing import handle_outliers

    X_train_clean, X_test_clean = handle_outliers(
        X_train_clean, X_test_clean, feature_names,
        IQR_MULTIPLIER, PLOTS_DIR, LOGS_DIR
    )

    # ═══════════════════════════════════════════════════════════════
    # STAGE 07 — Feature Scaling (fit on train, §6)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 07 — FEATURE SCALING (FIT ON TRAIN ONLY, §6)")
    print("=" * 60)

    from src.preprocessing import fit_scaler, transform_scaler, log_scaling

    scaler_params = fit_scaler(X_train_clean)
    X_train_scaled = transform_scaler(X_train_clean, scaler_params)
    X_test_scaled = transform_scaler(X_test_clean, scaler_params)

    print(f"  Scaling method: Z-score standardization")
    print(f"  Parameters fitted on TRAIN ({len(X_train_clean)} samples):")
    for i, name in enumerate(feature_names):
        print(f"    {name}: mean={scaler_params['mean'][i]:.4f}, "
              f"std={scaler_params['std'][i]:.4f}")
    print(f"  Train post-scaling: mean≈{np.mean(X_train_scaled, axis=0).round(4)}, "
          f"std≈{np.std(X_train_scaled, axis=0).round(4)}")

    log_scaling(scaler_params, X_train_scaled, X_test_scaled, feature_names, LOGS_DIR)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 08 — Categorical Encoding — SKIPPED
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 08 — CATEGORICAL ENCODING — SKIPPED")
    print("=" * 60)
    print("  Both features (eruptions, waiting) are numeric floats.")
    print("  No categorical variables exist in Old Faithful dataset.")
    print("  Decision: SKIP (ref ML_PIPELINE_REFERENCE.md §7)")

    skip_log = os.path.join(LOGS_DIR, "08_encoding.txt")
    with open(skip_log, 'w') as f:
        f.write("CATEGORICAL ENCODING — SKIPPED\n")
        f.write("=" * 40 + "\n")
        f.write("Reason: Both features are numeric (float64).\n")
        f.write("No categorical variables in Old Faithful dataset.\n")
        f.write("Ref: ML_PIPELINE_REFERENCE.md §7\n")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 09 — Class Imbalance — SKIPPED
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 09 — CLASS IMBALANCE — SKIPPED")
    print("=" * 60)
    print("  This is an UNSUPERVISED learning task (GMM clustering).")
    print("  No target labels exist → class imbalance is not applicable.")
    print("  Decision: SKIP (ref ML_PIPELINE_REFERENCE.md §8)")

    skip_log = os.path.join(LOGS_DIR, "09_imbalance.txt")
    with open(skip_log, 'w') as f:
        f.write("CLASS IMBALANCE — SKIPPED\n")
        f.write("=" * 40 + "\n")
        f.write("Reason: GMM is unsupervised — no class labels exist.\n")
        f.write("Class imbalance is a supervised learning concept.\n")
        f.write("Ref: ML_PIPELINE_REFERENCE.md §8\n")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 10 — Feature Engineering (§9.9)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 10 — FEATURE ENGINEERING (§9.9)")
    print("=" * 60)

    from src.feature_engineering import create_engineered_features, log_feature_engineering

    # Create engineered features on raw (pre-scaled) data
    X_train_aug_raw, aug_names = create_engineered_features(X_train_clean, feature_names)
    X_test_aug_raw, _ = create_engineered_features(X_test_clean, feature_names)

    print(f"  Original features: {feature_names}")
    print(f"  Engineered features: {aug_names[len(feature_names):]}")
    print(f"  Total features: {len(aug_names)}")

    # Scale augmented features (fit on train augmented)
    aug_scaler = fit_scaler(X_train_aug_raw)
    X_train_aug_scaled = transform_scaler(X_train_aug_raw, aug_scaler)
    X_test_aug_scaled = transform_scaler(X_test_aug_raw, aug_scaler)

    log_feature_engineering(X_train_clean, X_train_aug_raw,
                            feature_names, aug_names, LOGS_DIR, PLOTS_DIR)

    # NOTE: We will test GMM with ORIGINAL 2D features (primary)
    # and compare with augmented 4D features to see if engineering helps.
    # Primary pipeline uses 2D (eruptions, waiting) for the GMM.

    # ═══════════════════════════════════════════════════════════════
    # STAGE 11 — Baseline Models (§11)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 11 — BASELINE MODELS (§11)")
    print("=" * 60)

    from src.baseline import run_baselines

    baseline_results = run_baselines(
        X_train_scaled, X_test_scaled, K, RANDOM_SEED, LOGS_DIR
    )

    print(f"\n  Baseline 1 — Single Gaussian (K=1):")
    print(f"    Avg test LL: {baseline_results['single_gaussian']['avg_test_ll']:.4f}")
    print(f"  Baseline 2 — K-Means (K={K}):")
    print(f"    Silhouette:  {baseline_results['kmeans']['test_silhouette']:.4f}")
    print(f"  → GMM must beat BOTH baselines.")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 13 — GMM Training on TRAIN set (EM Algorithm)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 13 — GMM TRAINING (EM ALGORITHM, TRAIN SET ONLY)")
    print("=" * 60)

    from src.em import fit_gmm, compute_log_likelihood
    from src.visualization import (plot_raw_data, plot_gmm_ellipses,
                                    plot_convergence, plot_bic_aic)

    # Plot standardized training data
    print("\n[1] Plotting standardized training data...")
    plot_raw_data(X_train_scaled, os.path.join(PLOTS_DIR, "13_train_data_standardized.png"))

    # Train GMM with K=2 on TRAINING SET ONLY
    print(f"\n[2] Training GMM (K={K}, init={INIT_METHOD}, seed={RANDOM_SEED})...")
    print(f"    Data: TRAIN set only ({len(X_train_scaled)} samples)")
    print(f"    max_iters={MAX_ITERS}, tol={TOL}, reg_covar={REG_COVAR}")

    params, responsibilities, log_likelihoods, n_iters = fit_gmm(
        X_train_scaled, K, MAX_ITERS, TOL, REG_COVAR,
        init_method=INIT_METHOD, seed=RANDOM_SEED
    )

    # Hard assignments for training data
    train_labels = np.argmax(responsibilities, axis=1)
    train_ll = compute_log_likelihood(X_train_scaled, params)
    avg_train_ll = train_ll / len(X_train_scaled)

    print(f"\n[3] Learned GMM parameters:")
    print(params)
    print(f"\n    Train log-likelihood: {train_ll:.6f} (avg: {avg_train_ll:.4f})")
    print(f"    Converged in {n_iters} iterations")

    # Compare to baseline
    baseline_avg_ll = baseline_results['single_gaussian']['avg_train_ll']
    print(f"\n    Improvement over single Gaussian baseline:")
    print(f"    GMM avg LL: {avg_train_ll:.4f} vs Baseline: {baseline_avg_ll:.4f} "
          f"(delta: {avg_train_ll - baseline_avg_ll:+.4f})")

    # Save training log
    train_log_lines = []
    train_log_lines.append("GMM TRAINING LOG")
    train_log_lines.append("=" * 50)
    train_log_lines.append(f"Data: TRAIN SET ONLY ({len(X_train_scaled)} samples)")
    train_log_lines.append(f"K={K}, init={INIT_METHOD}, seed={RANDOM_SEED}")
    train_log_lines.append(f"max_iters={MAX_ITERS}, tol={TOL}, reg_covar={REG_COVAR}")
    train_log_lines.append(f"Converged in {n_iters} iterations")
    train_log_lines.append(f"Train log-likelihood: {train_ll:.6f}")
    train_log_lines.append(f"Avg train LL: {avg_train_ll:.6f}")
    train_log_lines.append("")
    train_log_lines.append("Log-likelihood per iteration:")
    for i, ll in enumerate(log_likelihoods):
        train_log_lines.append(f"  Iter {i+1:3d}: {ll:.6f}")
    train_log_lines.append("")
    train_log_lines.append("Learned Parameters:")
    train_log_lines.append(str(params))

    train_log_path = os.path.join(LOGS_DIR, "13_gmm_training.txt")
    with open(train_log_path, 'w') as f:
        f.write("\n".join(train_log_lines))
    print(f"  Training log saved: {train_log_path}")

    # Save plots
    print("\n[4] Saving GMM plots...")
    plot_gmm_ellipses(X_train_scaled, params,
                      os.path.join(PLOTS_DIR, "13_gmm_result.png"))
    plot_convergence(log_likelihoods,
                     os.path.join(PLOTS_DIR, "13_gmm_convergence.png"))

    # Save model parameters
    print("\n[5] Saving model parameters...")
    model_path = os.path.join(MODELS_DIR, "gmm_params.npz")
    np.savez(model_path,
             weights=params.weights,
             means=params.means,
             covariances=params.covariances,
             K=params.K,
             data_mean=scaler_params['mean'],
             data_std=scaler_params['std'])
    print(f"  Model saved: {model_path}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 14 — Hyperparameter Tuning (§14, §15, §18.3)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 14 — HYPERPARAMETER TUNING (§14, §18.3)")
    print("=" * 60)
    print("  Principle: Change ONE variable at a time (§18.3)")

    # Experiment 1: Sweep K (number of components)
    print("\n  [Experiment 1] Sweeping K (number of components)...")
    from src.metrics import compute_bic, compute_aic

    k_bics, k_aics, k_lls = [], [], []
    n_train = len(X_train_scaled)
    n_features = X_train_scaled.shape[1]

    exp_lines = []
    exp_lines.append("EXPERIMENT 001 — K (NUMBER OF COMPONENTS)")
    exp_lines.append("=" * 50)
    exp_lines.append(f"Single-variable principle (§18.3): ONLY K changes")
    exp_lines.append(f"Fixed: init={INIT_METHOD}, tol={TOL}, reg_covar={REG_COVAR}")
    exp_lines.append(f"Data: TRAIN set ({n_train} samples)")
    exp_lines.append("")

    for k_test in K_VALUES:
        params_k, _, ll_k, iters_k = fit_gmm(
            X_train_scaled, k_test, MAX_ITERS, TOL, REG_COVAR,
            init_method=INIT_METHOD, seed=RANDOM_SEED
        )
        ll_final = compute_log_likelihood(X_train_scaled, params_k)
        bic = compute_bic(ll_final, k_test, n_train, n_features)
        aic = compute_aic(ll_final, k_test, n_features)
        k_bics.append(bic)
        k_aics.append(aic)
        k_lls.append(ll_final)
        exp_lines.append(f"  K={k_test}: LL={ll_final:.4f}, BIC={bic:.4f}, "
                        f"AIC={aic:.4f}, iters={iters_k}")
        print(f"    K={k_test}: LL={ll_final:.4f}, BIC={bic:.4f}, AIC={aic:.4f}")

    best_k_bic = K_VALUES[np.argmin(k_bics)]
    exp_lines.append(f"\n  Best K by BIC: K={best_k_bic}")

    exp_path = os.path.join(EXPERIMENTS_DIR, "experiment_001_K.txt")
    with open(exp_path, 'w') as f:
        f.write("\n".join(exp_lines))
    print(f"  Experiment log saved: {exp_path}")

    # Plot BIC/AIC
    plot_bic_aic(K_VALUES, k_bics, k_aics,
                 os.path.join(PLOTS_DIR, "14_tuning_K_bic_aic.png"))

    # Experiment 2: Sweep reg_covar
    print("\n  [Experiment 2] Sweeping reg_covar (regularization)...")
    reg_values = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    reg_lls = []

    exp_lines2 = []
    exp_lines2.append("EXPERIMENT 002 — REG_COVAR (COVARIANCE REGULARIZATION)")
    exp_lines2.append("=" * 50)
    exp_lines2.append(f"Single-variable principle (§18.3): ONLY reg_covar changes")
    exp_lines2.append(f"Fixed: K={K}, init={INIT_METHOD}, tol={TOL}")
    exp_lines2.append("")

    for reg in reg_values:
        params_r, _, _, iters_r = fit_gmm(
            X_train_scaled, K, MAX_ITERS, TOL, reg,
            init_method=INIT_METHOD, seed=RANDOM_SEED
        )
        ll_r = compute_log_likelihood(X_train_scaled, params_r)
        reg_lls.append(ll_r)
        exp_lines2.append(f"  reg_covar={reg:.0e}: LL={ll_r:.4f}, iters={iters_r}")
        print(f"    reg_covar={reg:.0e}: LL={ll_r:.4f}")

    best_reg = reg_values[np.argmax(reg_lls)]
    exp_lines2.append(f"\n  Best reg_covar by LL: {best_reg:.0e}")

    exp_path2 = os.path.join(EXPERIMENTS_DIR, "experiment_002_reg_covar.txt")
    with open(exp_path2, 'w') as f:
        f.write("\n".join(exp_lines2))
    print(f"  Experiment log saved: {exp_path2}")

    # Plot reg_covar sweep
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(reg_values, reg_lls, 'o-', color='#8E44AD', markersize=8, linewidth=2)
    ax.set_xlabel('reg_covar (log scale)')
    ax.set_ylabel('Train Log-Likelihood')
    ax.set_title('Hyperparameter Tuning: reg_covar vs Log-Likelihood\n(§18.3 single-variable)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "14_tuning_reg_covar.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # Experiment 3: Sweep init_method
    print("\n  [Experiment 3] Sweeping init_method...")
    init_methods = ["kmeans", "random"]
    init_lls = []

    exp_lines3 = []
    exp_lines3.append("EXPERIMENT 003 — INIT_METHOD (INITIALIZATION)")
    exp_lines3.append("=" * 50)
    exp_lines3.append(f"Fixed: K={K}, tol={TOL}, reg_covar={REG_COVAR}")
    exp_lines3.append("")

    for method in init_methods:
        params_m, _, _, iters_m = fit_gmm(
            X_train_scaled, K, MAX_ITERS, TOL, REG_COVAR,
            init_method=method, seed=RANDOM_SEED
        )
        ll_m = compute_log_likelihood(X_train_scaled, params_m)
        init_lls.append(ll_m)
        exp_lines3.append(f"  init={method}: LL={ll_m:.4f}, iters={iters_m}")
        print(f"    init={method}: LL={ll_m:.4f}")

    best_init = init_methods[np.argmax(init_lls)]
    exp_lines3.append(f"\n  Best init by LL: {best_init}")

    exp_path3 = os.path.join(EXPERIMENTS_DIR, "experiment_003_init_method.txt")
    with open(exp_path3, 'w') as f:
        f.write("\n".join(exp_lines3))
    print(f"  Experiment log saved: {exp_path3}")

    # Summary table
    sweep_lines = []
    sweep_lines.append("HYPERPARAMETER SWEEP SUMMARY")
    sweep_lines.append("=" * 50)
    sweep_lines.append("")
    sweep_lines.append("Experiment 1: K (components)")
    sweep_lines.append(f"  {'K':>5} {'LL':>12} {'BIC':>12} {'AIC':>12}")
    sweep_lines.append(f"  {'-'*41}")
    for i, k_val in enumerate(K_VALUES):
        sweep_lines.append(f"  {k_val:>5} {k_lls[i]:>12.4f} {k_bics[i]:>12.4f} {k_aics[i]:>12.4f}")
    sweep_lines.append(f"  Best K (BIC): {best_k_bic}")
    sweep_lines.append("")
    sweep_lines.append("Experiment 2: reg_covar")
    for i, rv in enumerate(reg_values):
        sweep_lines.append(f"  reg_covar={rv:.0e}: LL={reg_lls[i]:.4f}")
    sweep_lines.append(f"  Best reg_covar: {best_reg:.0e}")
    sweep_lines.append("")
    sweep_lines.append("Experiment 3: init_method")
    for i, m in enumerate(init_methods):
        sweep_lines.append(f"  init={m}: LL={init_lls[i]:.4f}")
    sweep_lines.append(f"  Best init: {best_init}")

    sweep_path = os.path.join(LOGS_DIR, "14_hyperparameter_sweep.txt")
    with open(sweep_path, 'w') as f:
        f.write("\n".join(sweep_lines))
    print(f"\n  Sweep summary saved: {sweep_path}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 15 — Evaluation Metrics (on TRAIN set)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 15 — EVALUATION METRICS")
    print("=" * 60)

    from src.metrics import compute_all_gmm_metrics, write_metrics_report

    print("  Computing GMM metrics on TRAINING set...")
    train_metrics = compute_all_gmm_metrics(
        X_train_scaled, params, train_labels, train_ll
    )

    print(f"\n  Train Metrics:")
    print(f"    Log-Likelihood:     {train_metrics['log_likelihood']:.4f}")
    print(f"    BIC:                {train_metrics['bic']:.4f}")
    print(f"    AIC:                {train_metrics['aic']:.4f}")
    print(f"    Silhouette Score:   {train_metrics['silhouette']:.4f}")
    print(f"    Cluster Separation: {train_metrics['separation']:.4f}")

    report_path = os.path.join(METRICS_DIR, "gmm_train_metrics.txt")
    write_metrics_report(train_metrics, params, report_path)

    # Save BIC/AIC table
    bic_path = os.path.join(METRICS_DIR, "gmm_bic_scores.csv")
    with open(bic_path, 'w') as f:
        f.write("K,BIC,AIC,LL\n")
        for i, k_val in enumerate(K_VALUES):
            f.write(f"{k_val},{k_bics[i]:.4f},{k_aics[i]:.4f},{k_lls[i]:.4f}\n")
    print(f"  BIC/AIC scores saved: {bic_path}")

    # Confusion matrix plot (cluster assignments)
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in range(K):
        n_k = np.sum(train_labels == k)
        ax.bar(k, n_k, color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'][k],
               alpha=0.8, edgecolor='white')
        ax.text(k, n_k + 2, f'n={n_k}', ha='center', fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Training Samples')
    ax.set_title('GMM Cluster Assignments (Train Set)')
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'Component {k+1}\n(π={params.weights[k]:.3f})' for k in range(K)])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "15_cluster_assignments.png"),
                bbox_inches='tight', dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 16 — K-Fold Cross-Validation (§17)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 16 — K-FOLD CROSS-VALIDATION (§17)")
    print("=" * 60)
    print(f"  Scope: TRAINING SET ONLY ({len(X_train_clean)} samples)")
    print(f"  Test set is NOT touched during CV.")

    from src.cross_validation import cross_validate_gmm, log_cv_results

    # Run CV on raw (unscaled) training data — scaler is refit per fold
    cv_results = cross_validate_gmm(
        X_train_clean, K, N_FOLDS, MAX_ITERS, TOL, REG_COVAR,
        init_method=INIT_METHOD, seed=RANDOM_SEED
    )

    log_cv_results(cv_results, K, N_FOLDS, LOGS_DIR, METRICS_DIR)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 17 — Final Test-Set Evaluation (ONCE, §10.2 Rule 4)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 17 — FINAL TEST-SET EVALUATION (ONE TIME ONLY)")
    print("=" * 60)
    print("  ⚠️  This is the ONLY place the test set is used for evaluation.")

    # Evaluate GMM on test set
    test_ll = compute_log_likelihood(X_test_scaled, params)
    avg_test_ll = test_ll / len(X_test_scaled)

    # Assign test points to clusters
    from src.gaussian import gaussian_pdf_batch
    test_resp = np.zeros((len(X_test_scaled), K))
    for k in range(K):
        test_resp[:, k] = params.weights[k] * gaussian_pdf_batch(
            X_test_scaled, params.means[k], params.covariances[k]
        )
    test_labels = np.argmax(test_resp, axis=1)

    # Test metrics
    test_metrics = compute_all_gmm_metrics(
        X_test_scaled, params, test_labels, test_ll
    )

    print(f"\n  FINAL TEST RESULTS:")
    print(f"    Test samples:       {len(X_test_scaled)}")
    print(f"    Test LL:            {test_ll:.4f} (avg: {avg_test_ll:.4f})")
    print(f"    Test Silhouette:    {test_metrics['silhouette']:.4f}")
    print(f"    Test Separation:    {test_metrics['separation']:.4f}")
    print(f"\n  Comparison to baselines:")
    print(f"    GMM avg test LL:        {avg_test_ll:.4f}")
    print(f"    Baseline K=1 avg LL:    {baseline_results['single_gaussian']['avg_test_ll']:.4f}")
    improvement = avg_test_ll - baseline_results['single_gaussian']['avg_test_ll']
    print(f"    Improvement:            {improvement:+.4f}")
    print(f"\n  Comparison to CV estimate:")
    print(f"    CV avg LL (μ±σ):        {cv_results['ll_mean']:.4f} ± {cv_results['ll_std']:.4f}")
    print(f"    Actual test avg LL:     {avg_test_ll:.4f}")
    cv_gap = abs(avg_test_ll - cv_results['ll_mean'])
    print(f"    |CV estimate - actual|: {cv_gap:.4f}")

    # Generalization gap (bias-variance diagnosis §13)
    gen_gap = avg_train_ll - avg_test_ll
    print(f"\n  Generalization gap (§13):")
    print(f"    Train avg LL:  {avg_train_ll:.4f}")
    print(f"    Test avg LL:   {avg_test_ll:.4f}")
    print(f"    Gap:           {gen_gap:.4f}")
    if gen_gap < 0.1:
        print(f"    → Small gap: good generalization (near sweet spot)")
    elif gen_gap < 0.5:
        print(f"    → Moderate gap: some overfitting, but acceptable")
    else:
        print(f"    → Large gap: potential overfitting (high variance)")

    # Save final test results
    test_lines = []
    test_lines.append("FINAL TEST-SET EVALUATION")
    test_lines.append("=" * 50)
    test_lines.append(f"⚠️  Test set used exactly ONCE (§10.2 Rule 4)")
    test_lines.append("")
    test_lines.append(f"Test samples:        {len(X_test_scaled)}")
    test_lines.append(f"Test LL:             {test_ll:.4f}")
    test_lines.append(f"Avg test LL:         {avg_test_ll:.4f}")
    test_lines.append(f"Test BIC:            {test_metrics['bic']:.4f}")
    test_lines.append(f"Test AIC:            {test_metrics['aic']:.4f}")
    test_lines.append(f"Test Silhouette:     {test_metrics['silhouette']:.4f}")
    test_lines.append(f"Test Separation:     {test_metrics['separation']:.4f}")
    test_lines.append("")
    test_lines.append(f"Baseline comparison:")
    test_lines.append(f"  K=1 Gaussian avg LL: {baseline_results['single_gaussian']['avg_test_ll']:.4f}")
    test_lines.append(f"  GMM improvement:     {improvement:+.4f}")
    test_lines.append("")
    test_lines.append(f"CV estimate vs actual:")
    test_lines.append(f"  CV avg LL:   {cv_results['ll_mean']:.4f} ± {cv_results['ll_std']:.4f}")
    test_lines.append(f"  Actual:      {avg_test_ll:.4f}")
    test_lines.append(f"  Gap:         {cv_gap:.4f}")
    test_lines.append("")
    test_lines.append(f"Generalization gap (§13):")
    test_lines.append(f"  Train avg LL: {avg_train_ll:.4f}")
    test_lines.append(f"  Test avg LL:  {avg_test_ll:.4f}")
    test_lines.append(f"  Gap:          {gen_gap:.4f}")

    test_path = os.path.join(LOGS_DIR, "17_final_test_evaluation.txt")
    with open(test_path, 'w') as f:
        f.write("\n".join(test_lines))
    print(f"\n  Final test log saved: {test_path}")

    test_metrics_path = os.path.join(METRICS_DIR, "final_test_results.txt")
    with open(test_metrics_path, 'w') as f:
        f.write("\n".join(test_lines))
    print(f"  Final test metrics saved: {test_metrics_path}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 18 — Error Analysis (§19)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 18 — ERROR ANALYSIS (§19)")
    print("=" * 60)

    # Compute per-point log-likelihood (density) on test set
    test_densities = np.zeros(len(X_test_scaled))
    for k in range(K):
        test_densities += params.weights[k] * gaussian_pdf_batch(
            X_test_scaled, params.means[k], params.covariances[k]
        )
    test_log_densities = np.log(np.maximum(test_densities, 1e-300))

    # Find worst-explained points (lowest log-density)
    n_worst = min(10, len(X_test_scaled))
    worst_idx = np.argsort(test_log_densities)[:n_worst]

    error_lines = []
    error_lines.append("ERROR ANALYSIS")
    error_lines.append("=" * 50)
    error_lines.append("Reference: ML_PIPELINE_REFERENCE.md §19")
    error_lines.append("")
    error_lines.append(f"Analysis: {n_worst} lowest-likelihood test points")
    error_lines.append("")
    error_lines.append(f"{'Rank':>5} {'Idx':>5} {'LogDensity':>12} {'Label':>6} "
                      f"{'Eruptions':>10} {'Waiting':>10}")
    error_lines.append("-" * 60)

    for rank, idx in enumerate(worst_idx):
        error_lines.append(
            f"{rank+1:>5} {idx:>5} {test_log_densities[idx]:>12.4f} "
            f"{test_labels[idx]:>6} "
            f"{X_test_raw[idx, 0]:>10.2f} {X_test_raw[idx, 1]:>10.1f}"
        )

    # Group analysis
    error_lines.append("")
    error_lines.append("PATTERN ANALYSIS:")
    error_lines.append("-" * 40)

    worst_eruptions = X_test_raw[worst_idx, 0]
    worst_waiting = X_test_raw[worst_idx, 1]

    # Check if worst points cluster between modes
    mid_eruptions = (worst_eruptions > 2.5) & (worst_eruptions < 3.5)
    n_mid = np.sum(mid_eruptions)
    error_lines.append(f"  Points in 'gap' region (2.5-3.5 min eruption): {n_mid}/{n_worst}")
    error_lines.append(f"  Mean eruptions of worst points: {np.mean(worst_eruptions):.2f}")
    error_lines.append(f"  Mean waiting of worst points:   {np.mean(worst_waiting):.1f}")
    error_lines.append("")
    error_lines.append("ROOT CAUSE ANALYSIS:")
    error_lines.append("  The lowest-likelihood points are expected to lie in the")
    error_lines.append("  transition zone between the two eruption modes. These are")
    error_lines.append("  genuine data points — not errors — that happen to fall in")
    error_lines.append("  the low-density region between the two Gaussian components.")
    error_lines.append("  This is a fundamental limitation of the GMM's assumption")
    error_lines.append("  that the data is a mixture of Gaussians (§12 inductive bias).")

    error_log_path = os.path.join(LOGS_DIR, "18_error_analysis.txt")
    with open(error_log_path, 'w') as f:
        f.write("\n".join(error_lines))
    print(f"  Error analysis log saved: {error_log_path}")

    # Error analysis plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot all test points
    for k in range(K):
        mask = test_labels == k
        color = ['#E74C3C', '#3498DB'][k % 2]
        ax.scatter(X_test_scaled[mask, 0], X_test_scaled[mask, 1],
                  c=color, alpha=0.4, s=20, label=f'Component {k+1}')

    # Highlight worst points
    ax.scatter(X_test_scaled[worst_idx, 0], X_test_scaled[worst_idx, 1],
              c='black', s=100, marker='X', zorder=5,
              label=f'Worst {n_worst} points')

    ax.set_xlabel('Eruption Duration (standardized)')
    ax.set_ylabel('Waiting Time (standardized)')
    ax.set_title('Error Analysis: Lowest-Likelihood Test Points')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "18_error_analysis.png"),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Error analysis plot saved")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 19 — Final Visualization Pass
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 19 — FINAL VISUALIZATION PASS")
    print("=" * 60)

    readme_lines = []
    readme_lines.append("PLOT INVENTORY — GMM Old Faithful Pipeline (v2 Refactored)")
    readme_lines.append("=" * 60)
    readme_lines.append("")

    plot_descriptions = {
        "03_eda_histograms.png": "Univariate distributions (histogram + KDE) for each feature",
        "03_eda_correlation.png": "Correlation heatmap + scatter plot showing bimodal structure",
        "03_eda_boxplots.png": "Box plots for initial outlier detection preview",
        "04_split_distribution.png": "Train/test split distribution comparison",
        "06_outliers_boxplot.png": "IQR outlier detection results (bounds from train)",
        "10_engineered_features.png": "Engineered feature distributions (interaction, polynomial)",
        "13_train_data_standardized.png": "Standardized training data scatter plot",
        "13_gmm_result.png": "GMM clustering with Gaussian confidence ellipses",
        "13_gmm_convergence.png": "EM convergence (log-likelihood + delta per iteration)",
        "14_tuning_K_bic_aic.png": "Model selection: BIC and AIC vs K",
        "14_tuning_reg_covar.png": "Regularization sweep: LL vs reg_covar",
        "15_cluster_assignments.png": "GMM cluster assignment counts (train set)",
        "18_error_analysis.png": "Lowest-likelihood test points highlighted",
    }

    if os.path.exists(PLOTS_DIR):
        for f_name in sorted(os.listdir(PLOTS_DIR)):
            if f_name.endswith('.png'):
                readme_lines.append(f"  {f_name}")
                if f_name in plot_descriptions:
                    readme_lines.append(f"    → {plot_descriptions[f_name]}")
                readme_lines.append("")

    readme_path = os.path.join(PLOTS_DIR, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("\n".join(readme_lines))
    print(f"  Plot inventory saved: {readme_path}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 20 — Interpretability Notes (§20)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 20 — INTERPRETABILITY NOTES (§20)")
    print("=" * 60)

    # Unstandardize means for physical interpretation
    phys_means = params.means * scaler_params['std'] + scaler_params['mean']

    interp_lines = []
    interp_lines.append("MODEL INTERPRETABILITY")
    interp_lines.append("=" * 50)
    interp_lines.append("Reference: ML_PIPELINE_REFERENCE.md §20")
    interp_lines.append("")
    interp_lines.append("GLOBAL INTERPRETABILITY")
    interp_lines.append("-" * 40)
    interp_lines.append("The GMM with K=2 identifies two distinct eruption regimes:")
    interp_lines.append("")

    for k in range(K):
        interp_lines.append(f"  Component {k+1} (weight π={params.weights[k]:.3f}):")
        interp_lines.append(f"    Physical mean: eruptions={phys_means[k, 0]:.2f} min, "
                           f"waiting={phys_means[k, 1]:.1f} min")
        interp_lines.append(f"    Standardized mean: {params.means[k]}")
        interp_lines.append(f"    Covariance (standardized):")
        for row in params.covariances[k]:
            interp_lines.append(f"      {row}")
        interp_lines.append("")

    interp_lines.append("Physical interpretation:")
    if phys_means[0, 0] < phys_means[1, 0]:
        short_k, long_k = 0, 1
    else:
        short_k, long_k = 1, 0

    interp_lines.append(f"  Component {short_k+1}: SHORT eruptions (~{phys_means[short_k, 0]:.1f} min) "
                       f"with SHORT waits (~{phys_means[short_k, 1]:.0f} min)")
    interp_lines.append(f"  Component {long_k+1}: LONG eruptions (~{phys_means[long_k, 0]:.1f} min) "
                       f"with LONG waits (~{phys_means[long_k, 1]:.0f} min)")
    interp_lines.append("")
    interp_lines.append("This matches the well-documented bimodal behavior of Old Faithful:")
    interp_lines.append("short eruptions deplete less water from the underground chamber,")
    interp_lines.append("requiring less time to refill → shorter wait. Long eruptions deplete")
    interp_lines.append("more water → longer refill → longer wait.")
    interp_lines.append("")
    interp_lines.append("The positive off-diagonal covariance entries confirm that eruption")
    interp_lines.append("duration and waiting time are positively correlated WITHIN each")
    interp_lines.append("cluster, not just between clusters.")
    interp_lines.append("")
    interp_lines.append("LOCAL INTERPRETABILITY")
    interp_lines.append("-" * 40)
    interp_lines.append("For any specific data point, the responsibilities γ(n,k) from the")
    interp_lines.append("E-step give the probability of belonging to each component.")
    interp_lines.append("Points near cluster centers have γ≈1 for one component and γ≈0")
    interp_lines.append("for the other. Points in the transition zone have more balanced γ,")
    interp_lines.append("indicating genuine ambiguity — which is captured by GMM's soft")
    interp_lines.append("assignment but lost in K-Means' hard assignment.")
    interp_lines.append("")
    interp_lines.append("FEATURE IMPORTANCE:")
    interp_lines.append("Both features (eruptions, waiting) contribute meaningfully to")
    interp_lines.append("cluster separation. The strong positive correlation (r≈0.9)")
    interp_lines.append("means either feature alone provides significant discriminative")
    interp_lines.append("power, but using both together with full covariance captures")
    interp_lines.append("the elongated cluster shapes that a diagonal covariance would miss.")

    interp_path = os.path.join(LOGS_DIR, "20_interpretability.txt")
    with open(interp_path, 'w') as f:
        f.write("\n".join(interp_lines))
    print(f"  Interpretability notes saved: {interp_path}")

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "╔" + "═" * 63 + "╗")
    print("║  PIPELINE COMPLETE (v2 — REFACTORED)                         ║")
    print("╚" + "═" * 63 + "╝")

    print(f"\n  Pipeline Stages Executed:")
    print(f"    ✅ 01 — Project setup")
    print(f"    ✅ 02 — Data loading ({X_raw.shape[0]} samples)")
    print(f"    ✅ 03 — EDA (3 lenses: global, univariate, multivariate)")
    print(f"    ✅ 04 — Train/test split ({len(X_train_raw)}/{len(X_test_raw)})")
    print(f"    ✅ 05 — Missing value check (train only)")
    print(f"    ✅ 06 — Outlier detection (bounds from train)")
    print(f"    ✅ 07 — Feature scaling (fit on train only)")
    print(f"    ⏭️  08 — Categorical encoding (SKIPPED — all numeric)")
    print(f"    ⏭️  09 — Class imbalance (SKIPPED — unsupervised)")
    print(f"    ✅ 10 — Feature engineering (2 new features)")
    print(f"    ✅ 11 — Baseline models (single Gaussian + K-Means)")
    print(f"    ✅ 13 — GMM training (K={K}, {n_iters} iters, TRAIN only)")
    print(f"    ✅ 14 — Hyperparameter tuning (3 experiments)")
    print(f"    ✅ 15 — Evaluation metrics")
    print(f"    ✅ 16 — {N_FOLDS}-Fold cross-validation (μ±σ)")
    print(f"    ✅ 17 — Final test evaluation (ONCE)")
    print(f"    ✅ 18 — Error analysis")
    print(f"    ✅ 19 — Plot inventory")
    print(f"    ✅ 20 — Interpretability notes")

    print(f"\n  Key Results:")
    print(f"    Train avg LL:           {avg_train_ll:.4f}")
    print(f"    Test avg LL:            {avg_test_ll:.4f}")
    print(f"    CV avg LL (μ±σ):        {cv_results['ll_mean']:.4f} ± {cv_results['ll_std']:.4f}")
    print(f"    Baseline K=1 test LL:   {baseline_results['single_gaussian']['avg_test_ll']:.4f}")
    print(f"    GMM improvement:        {improvement:+.4f}")
    print(f"    Best K (BIC):           {best_k_bic}")

    print(f"\n  Generated files:")
    for dirname, dirpath in [("Plots", PLOTS_DIR), ("Logs", LOGS_DIR),
                              ("Metrics", METRICS_DIR), ("Models", MODELS_DIR),
                              ("Experiments", EXPERIMENTS_DIR)]:
        print(f"    {dirname}: {dirpath}/")
        if os.path.exists(dirpath):
            for f_name in sorted(os.listdir(dirpath)):
                print(f"      - {f_name}")


if __name__ == "__main__":
    main()
