# ML Project Refactor — Two-Phase Prompt

---

## PHASE 1 — Audit (DO NOT MODIFY ANY CODE)

```
You are an experienced Machine Learning Engineer doing a code review.

I have attached two things:
1. My current ML project (MLP + Gaussian Mixture Model).
2. ML_PIPELINE_REFERENCE.md — my instructor's pipeline, used as conceptual reference only.

Your task in this phase is ONLY to audit. Do not modify, do not suggest fixes yet.
Produce a structured engineering report with the sections below.

---

⚠️ CRITICAL IMPLEMENTATION CONSTRAINT
This project is for a university course. We are NOT allowed to use high-level ML libraries.
Forbidden: scikit-learn, PyTorch, TensorFlow, Keras, XGBoost, LightGBM, imblearn, or any library
that provides a model, scaler, encoder, imputer, or sampler as a pre-built class.

Allowed: numpy, pandas, matplotlib, scipy.stats (for statistical tests only), and standard Python.

Everything — MLP forward/backward pass, GMM EM algorithm, scaling, imputation, encoding,
outlier detection — must be implemented from scratch using numpy math.

When referencing ML_PIPELINE_REFERENCE.md:
- Use it to understand WHAT should be done and WHY.
- Do NOT use its sklearn/imblearn code patterns for implementation.
- Translate every concept into numpy-level math.

---

## Step 1 — Project Structure

List every file in the project. For each file state:
- filename
- responsibility (one sentence)
- key classes and functions inside it

---

## Step 2 — Execution Flow

Trace the complete execution order from entry point to final output.
Use an arrow diagram. Do NOT assume the order is correct — describe what actually happens.

---

## Step 3 — Pipeline Coverage Audit

Compare the project against ML_PIPELINE_REFERENCE.md.
Produce this exact table:

| Pipeline Stage            | Exists? | Where in code? | Implemented correctly? | Missing or broken? | Notes |
|---------------------------|---------|----------------|------------------------|--------------------|-------|
| Problem type identified   |         |                |                        |                    |       |
| EDA — global overview     |         |                |                        |                    |       |
| EDA — univariate dist.    |         |                |                        |                    |       |
| EDA — multivariate corr.  |         |                |                        |                    |       |
| Missing value analysis    |         |                |                        |                    |       |
| Missing value handling    |         |                |                        |                    |       |
| Outlier detection         |         |                |                        |                    |       |
| Outlier treatment         |         |                |                        |                    |       |
| Feature scaling           |         |                |                        |                    |       |
| Categorical encoding      |         |                |                        |                    |       |
| Class imbalance handling  |         |                |                        |                    |       |
| Feature engineering       |         |                |                        |                    |       |
| Model training — MLP      |         |                |                        |                    |       |
| Model training — GMM      |         |                |                        |                    |       |
| Evaluation metrics        |         |                |                        |                    |       |
| Visualizations            |         |                |                        |                    |       |

---

## Step 4 — Data Flow

Trace how data moves from source to model output.
For every transformation write:
    Input → shape/type → Operation (numpy function used) → Output → shape/type

---

## Step 5 — Feature Analysis

For every feature in the dataset, fill this table:

| Feature Name | Data Type | Missing Values? | Outliers? | Scaling Applied? | Encoding Applied? | Engineered? |
|--------------|-----------|-----------------|-----------|------------------|-------------------|-------------|

---

## Step 6 — Output Inventory

List every file the project currently saves or prints.
Classify each as: plot / log / metric / model weight / other.
Note whether it is currently organized or dumped into the root directory.

---

## Step 7 — Model Analysis

For each model (MLP, GMM):
- Algorithm and mathematical formulation (brief)
- Architecture / hyperparameters
- Loss function
- Optimization procedure (gradient descent / EM — specify exactly what is implemented)
- Stopping criterion
- Current weaknesses based on what you see in the code

---

## Step 8 — Code Quality Issues

List concrete problems found:
- Magic numbers (hard-coded values with no explanation)
- Hard-coded paths
- Dead code or unreachable branches
- Duplicate logic
- Missing random seed (reproducibility)
- Missing docstrings on key functions
Do not fix yet.

---

## Step 9 — Missing Pipeline Components

For each missing stage from Step 3, explain:
- What is missing
- Why it matters for this specific dataset
- What the numpy-level implementation would conceptually involve
- What the expected effect on MLP and GMM performance would be

---

## Step 10 — Refactoring Plan

Produce a numbered implementation roadmap.
For each step: what changes, what file it touches, what the output will be.
Do NOT write code. This is a plan only.

Follow this output folder structure in your plan:
    results/
        logs/        ← training logs, metric tables, console output
        plots/       ← all matplotlib figures, organized by pipeline stage
        metrics/     ← final quantitative results (accuracy, loss, BIC, etc.)
        models/      ← saved weights or parameters if applicable
```

---

## PHASE 2 — Refactor (one pipeline stage at a time)

```
Using your Phase 1 engineering report, now refactor the project.

⚠️ SAME CONSTRAINT AS PHASE 1:
Do NOT use scikit-learn, PyTorch, TensorFlow, Keras, imblearn, or any pre-built ML class.
Implement everything from scratch in numpy.
ML_PIPELINE_REFERENCE.md is the conceptual guide — translate its ideas into numpy math.

Rules:
- Implement exactly ONE pipeline stage per reply.
- Do not touch code outside the current stage.
- Before writing code, state in one paragraph: what you are implementing and why.
- After writing code, produce a summary block (format below).
- Justify every decision by referencing the relevant section of ML_PIPELINE_REFERENCE.md.
- Fix random seeds for reproducibility: np.random.seed(42).

---

Output folder structure — enforce this strictly:

    results/
        logs/        ← print all training progress, epoch losses, EM iterations here
        plots/       ← save all matplotlib figures here, named by stage
                        Naming: {stage}_{description}.png
                        Examples: 03_eda_histograms.png, 05_outliers_boxplot.png
        metrics/     ← save final quantitative results as .txt or .csv
                        Examples: mlp_final_metrics.txt, gmm_bic_scores.csv
        models/      ← save final model parameters as .npy if applicable

Every plt.savefig() call must save into results/plots/.
Every print() of a metric or score must also be written to results/logs/.
Never show plots interactively (plt.show()) — always save to file.

---

Implement in this order. Wait for my confirmation before moving to the next stage.

Stage 01 — Project structure cleanup
         Create the results/ folder hierarchy. Rename files if needed.
         Add a config.py or constants section with all magic numbers.

Stage 02 — Data loading
         Robust CSV/data loading. Print shape, dtypes, head.
         Log output → results/logs/02_data_loading.txt

Stage 03 — EDA
         Global overview: shape, types, missing count, value ranges.
         Univariate: histogram + KDE for every numerical feature.
         Multivariate: correlation heatmap.
         All figures → results/plots/03_eda_*.png
         Summary → results/logs/03_eda_summary.txt

Stage 04 — Missing value handling
         Check distribution of each feature first (numpy: mean vs median decision).
         Implement mean imputation, median imputation, and KNN imputation in pure numpy.
         Log which strategy was chosen per feature → results/logs/04_imputation.txt

Stage 05 — Outlier detection and treatment
         Implement IQR detection in numpy.
         Generate boxplots before and after treatment.
         Plots → results/plots/05_outliers_*.png
         Log → results/logs/05_outliers.txt (which points flagged, what action taken)

Stage 06 — Feature scaling
         Implement StandardScaler and MinMaxScaler in numpy (no sklearn).
         Choose method per feature based on ML_PIPELINE_REFERENCE.md §6.4.
         Apply fit on train, transform on val/test.
         Log scaling parameters (mean, std per feature) → results/logs/06_scaling.txt

Stage 07 — Encoding
         Identify any categorical features. Apply appropriate encoding in numpy.
         If no categorical features exist, document this explicitly and skip.
         Log → results/logs/07_encoding.txt

Stage 08 — Class imbalance
         Compute class distribution. Plot class bar chart → results/plots/08_class_dist.png
         If imbalanced: implement oversampling or class weighting in numpy.
         Log → results/logs/08_imbalance.txt

Stage 09 — Feature engineering
         Based on EDA findings, create at least one engineered feature.
         Justify each new feature using domain knowledge.
         Plot: distribution of engineered features → results/plots/09_engineered_*.png
         Log → results/logs/09_feature_engineering.txt

Stage 10 — MLP training
         Refactor the existing MLP to be clean and modular.
         Log every epoch: loss, accuracy → results/logs/10_mlp_training.txt
         Save loss curve → results/plots/10_mlp_loss_curve.png
         Save final weights → results/models/mlp_weights.npy

Stage 11 — GMM training
         Refactor the existing GMM (EM algorithm) to be clean and modular.
         Log every EM iteration: log-likelihood → results/logs/11_gmm_training.txt
         Save BIC/AIC curve → results/plots/11_gmm_bic.png
         Save final parameters (means, covariances, weights) → results/models/gmm_params.npy

Stage 12 — Evaluation
         Compute and save all metrics.
         MLP: accuracy, precision, recall, F1, confusion matrix.
         GMM: BIC, AIC, log-likelihood, silhouette score (numpy implementation).
         All metrics → results/metrics/
         Confusion matrix plot → results/plots/12_confusion_matrix.png

Stage 13 — Final visualization pass
         Review all saved plots. Ensure every plot has:
           - title
           - axis labels
           - legend if applicable
           - consistent style (set once at top of file)
         Generate a results/plots/README.txt listing every plot and its interpretation.

---

After each stage, produce this exact summary block:

## Stage [N] Summary
**What changed:** [list of modified/added functions and files]
**Why:** [reference to ML_PIPELINE_REFERENCE.md section]
**numpy implementation note:** [key formula or algorithm used, e.g. x' = (x - μ) / σ]
**Expected impact:** [effect on MLP/GMM performance]
**Essay note:** [one sentence describing this stage for use in the essay methodology section]
```

---

## AI Implementation Checklist

Add this section at the end of `ML_PIPELINE_REFERENCE.md` for the agent to verify before training:

```markdown
## AI Implementation Checklist

⚠️ No scikit-learn, PyTorch, or any pre-built ML class. Pure numpy only.

Before training any model, verify every box is checked:

### Data Understanding
- [ ] Problem type identified (Supervised / Unsupervised / RL)
- [ ] Dataset shape printed (n_samples, n_features)
- [ ] Feature types identified (numeric / categorical / bool)
- [ ] Missing value counts per feature logged
- [ ] Value ranges and distributions visualized

### Preprocessing
- [ ] Missing values handled (strategy chosen based on distribution, not default)
- [ ] Outliers detected (IQR) and decision documented (remove / cap / transform)
- [ ] Feature scaling applied (StandardScaler or MinMaxScaler in numpy — fit on train only)
- [ ] Categorical encoding applied or explicitly skipped with justification
- [ ] Class imbalance checked and addressed if present

### Feature Engineering
- [ ] At least one engineered feature created and justified
- [ ] Redundant features removed or documented

### Reproducibility
- [ ] np.random.seed(42) set at entry point
- [ ] All file paths use os.path.join(), not hard-coded strings

### Output Organization
- [ ] results/logs/    ← all training logs and metric prints
- [ ] results/plots/   ← all matplotlib figures (no plt.show())
- [ ] results/metrics/ ← final scores saved as .txt or .csv
- [ ] results/models/  ← saved numpy arrays for weights/parameters

### Essay Readiness
- [ ] Every plot has title, axis labels, legend
- [ ] Every metric is saved to file (not just printed)
- [ ] results/plots/README.txt exists, listing every figure and its purpose
- [ ] Stage summaries written (the "Essay note:" line from each Stage Summary block)
```
