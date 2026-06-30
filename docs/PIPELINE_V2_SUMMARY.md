# GMM Pipeline v2 (Refactored) — Architecture Review

This document summarizes the complete refactoring of the Old Faithful GMM project to adhere to the zero-leakage requirements outlined in `ML_PIPELINE_REFERENCE_v2.md` and `REFACTOR_PROMPTS_v2.md`. 

The core constraint of this project was to implement all ML operations—from data splitting and preprocessing to the Expectation-Maximization algorithm and Cross-Validation—from scratch using **pure NumPy**, strictly prohibiting libraries such as scikit-learn.

## 1. Architectural Changes

The pipeline was heavily refactored to enforce a strict boundary between training and testing data. In previous iterations, operations like scaling and outlier detection were mistakenly applied globally across the entire dataset. In Pipeline v2, these operations were decoupled into separate `fit` and `transform` steps.

### New Modules Introduced
The following modules were created to support the robust v2 pipeline:

1. **`src/split.py`**:
   - Implements a manual `train_test_split` using a seeded numpy random generator.
   - Provides functions to log split distributions and warn against leakage.
2. **`src/baseline.py`**:
   - Implements a Single Gaussian (K=1) MLE fit as a lower-bound generative baseline.
   - Implements K-Means (utilizing `src/kmeans.py`) as a hard-assignment baseline for clustering comparison.
3. **`src/feature_engineering.py`**:
   - Applies feature engineering specific to the Old Faithful dataset (e.g., polynomial transformations and interaction terms).
4. **`src/cross_validation.py`**:
   - Implements K-Fold cross-validation manually.
   - For each fold, it creates sub-splits, scales data using ONLY the fold-train subset (preventing intra-CV leakage), fits the GMM, and evaluates performance (Log-Likelihood and Silhouette) on the fold-val subset.

## 2. Prevention of Data Leakage

The pipeline enforces data hygiene through the following mechanisms:

- **Stage 04 (Train/Test Split)**: The dataset is split into `X_train` and `X_test` (80/20) before any transformations occur.
- **Stage 05 & 06 (Imputation & Outliers)**: Statistics for imputation (e.g., medians) and outlier boundaries (e.g., IQR thresholds) are calculated strictly on `X_train`.
- **Stage 07 (Scaling)**: Mean and Standard Deviation are computed from `X_train` (`fit_scaler`). The test dataset is transformed using these exact train-derived statistics (`transform_scaler`), ensuring the test set remains completely unseen by the model until final evaluation.

## 3. The 20-Stage Pipeline (`main.py`)

The `main.py` script was completely rewritten to act as an orchestrator for the 20 stages:

1. **Stage 01 — Project Setup**: Generates results directories (plots, logs, metrics, models, experiments).
2. **Stage 02 — Data Loading**: Loads data via `src/data_loader.py` and cleans raw invalid rows.
3. **Stage 03 — EDA**: Performs exploratory data analysis on the initial dataset prior to transformations.
4. **Stage 04 — Train/Test Split**: Enforces the primary leakage boundary (80/20 split).
5. **Stage 05 — Missing Values**: Fits and applies missing value logic (train only).
6. **Stage 06 — Outliers**: Calculates IQR bounds (train only). 
7. **Stage 07 — Feature Scaling**: Z-score standardizes both arrays utilizing train-set statistics.
8. **Stage 08 — Categorical Encoding**: Skipped automatically (no categorical features).
9. **Stage 09 — Class Imbalance**: Skipped automatically (unsupervised learning).
10. **Stage 10 — Feature Engineering**: Appends polynomial and interaction features.
11. **Stage 11 — Baselines**: Trains Baseline 1 (Single Gaussian) and Baseline 2 (K-Means).
12. **Stage 13 — GMM Training**: Fits the GMM utilizing the EM Algorithm on `X_train` alone.
13. **Stage 14 — Hyperparameter Tuning**: Executes experiments varying components ($K$), regularization (`reg_covar`), and `init_method`.
14. **Stage 15 — Evaluation**: Computes BIC, AIC, and Silhouette scores for the GMM.
15. **Stage 16 — K-Fold CV**: Calculates model stability ($\mu \pm \sigma$) over $5$ folds entirely within `X_train`.
16. **Stage 17 — Final Test Evaluation**: Exposes the test set to the model for the first and only time to assess real-world generalization gap.
17. **Stage 18 — Error Analysis**: Analyzes test points with the lowest likelihood densities.
18. **Stage 19 — Visualization**: Outputs a README index of all auto-generated plots.
19. **Stage 20 — Interpretability**: Maps the standardized learned means and covariances back to physical eruption durations and wait times.

## 4. Execution Integrity

The entire script can be executed sequentially via `python3 main.py`. It requires no interaction and generates all assets (CSV reports, `matplotlib` plots, `.npz` parameter dumps, and `.txt` logging) entirely within the `/results/` directory, adhering to the pure NumPy constraint required for the course.
