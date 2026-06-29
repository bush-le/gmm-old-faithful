"""
run_experiments.py — Master orchestrator for the entire GMM Iris experiment pipeline.

Runs everything end-to-end:
1. Data preprocessing (load, clean, standardize, save CSV)
2. GMM training with EM algorithm
3. Baseline models (K-Means, Hierarchical, KNN)
4. Hyperparameter justification experiments
5. Model comparison and metrics report

Usage:
    python3 experiments/run_experiments.py
"""
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH, PLOTS_DIR, LOGS_DIR


def main():
    """Run the complete experimental pipeline."""

    print("╔" + "═" * 63 + "╗")
    print("║  GMM FROM SCRATCH — IRIS EXPERIMENT PIPELINE                 ║")
    print("╚" + "═" * 63 + "╝")

    # ─── Step 1: Preprocessing ───
    from src.preprocessing import run_pipeline
    standardized, mean, std = run_pipeline()

    # ─── Step 2: GMM Training ───
    from experiments.run_gmm import run_gmm_experiment
    gmm_params, gmm_resp, gmm_labels, gmm_ll_history = run_gmm_experiment()

    # ─── Step 3: Baselines ───
    from experiments.run_baselines import run_all_baselines
    baseline_results = run_all_baselines()

    # ─── Step 4: Hyperparameter Justification Experiments ───
    from experiments.hyperparameter_study import run_all_hyperparameter_experiments
    run_all_hyperparameter_experiments()

    # ─── Step 5: Model Comparison ───
    from experiments.compare_models import run_comparison
    from src.data_loader import load_csv

    X = np.array(load_csv(PROCESSED_DATA_PATH))

    metrics = run_comparison(
        X,
        gmm_labels=gmm_labels,
        gmm_params=gmm_params,
        kmeans_labels=baseline_results['kmeans_labels'],
        hierarchical_labels=baseline_results['hierarchical_labels'],
        knn_consistency=baseline_results['knn_consistency'],
    )

    # ─── Summary ───
    print("\n" + "╔" + "═" * 63 + "╗")
    print("║  PIPELINE COMPLETE                                           ║")
    print("╚" + "═" * 63 + "╝")
    print(f"\n  Generated files:")
    print(f"    Data:    {PROCESSED_DATA_PATH}")
    print(f"    Plots:   {PLOTS_DIR}/")

    # List plot files
    if os.path.exists(PLOTS_DIR):
        for f in sorted(os.listdir(PLOTS_DIR)):
            print(f"             - {f}")

    print(f"    Report:  {os.path.join(LOGS_DIR, 'metrics.txt')}")


if __name__ == "__main__":
    main()
