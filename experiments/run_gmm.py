"""
run_gmm.py — Driver script for GMM training and evaluation.

Loads preprocessed Iris data, trains the GMM using EM,
prints learned parameters, and saves visualization plots.
"""
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (K, MAX_ITERS, TOL, REG_COVAR, INIT_METHOD,
                    RANDOM_SEED, PROCESSED_DATA_PATH, PLOTS_DIR)
from src.data_loader import load_csv
from src.em import fit_gmm, compute_log_likelihood
from src.visualization import plot_gmm_ellipses, plot_convergence, plot_raw_data


def run_gmm_experiment():
    """
    Run the full GMM training experiment.

    Steps:
    1. Load preprocessed (standardized) data
    2. Plot raw data scatter
    3. Train GMM using EM algorithm
    4. Print learned parameters
    5. Save GMM clustering plot with Gaussian ellipses
    6. Save convergence curve

    Returns:
        tuple: (params, responsibilities, labels, log_likelihoods)
    """
    print("\n" + "=" * 60)
    print("GMM EXPERIMENT")
    print("=" * 60)

    # Load processed data
    print("\n[1] Loading processed data...")
    raw_data = load_csv(PROCESSED_DATA_PATH)
    X = np.array(raw_data)
    print(f"    Data shape: {X.shape}")

    # Plot raw data
    print("\n[2] Plotting raw data scatter...")
    plot_raw_data(X, os.path.join(PLOTS_DIR, "raw_data.png"))

    # Train GMM
    print("\n[3] Training GMM with EM algorithm...")
    print(f"    K={K}, max_iters={MAX_ITERS}, tol={TOL}")
    print(f"    reg_covar={REG_COVAR}, init={INIT_METHOD}, seed={RANDOM_SEED}")

    params, responsibilities, log_likelihoods, n_iters = fit_gmm(
        X, K, MAX_ITERS, TOL, REG_COVAR,
        init_method=INIT_METHOD, seed=RANDOM_SEED
    )

    # Print results
    print("\n[4] Learned GMM parameters:")
    print(params)

    final_ll = compute_log_likelihood(X, params)
    print(f"\n    Final log-likelihood: {final_ll:.6f}")
    print(f"    Converged in {n_iters} iterations")

    # Get hard assignments from responsibilities
    labels = np.argmax(responsibilities, axis=1)

    # Plot results
    print("\n[5] Saving plots...")
    plot_gmm_ellipses(X, params, os.path.join(PLOTS_DIR, "gmm_result.png"))
    plot_convergence(log_likelihoods, os.path.join(PLOTS_DIR, "em_convergence.png"))

    print("\n[DONE] GMM experiment complete.")
    return params, responsibilities, labels, log_likelihoods


if __name__ == "__main__":
    run_gmm_experiment()
