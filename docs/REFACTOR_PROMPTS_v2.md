# ML Project Refactor — Two-Phase Prompt (v2)

**What changed from v1:** ML_PIPELINE_REFERENCE.md now covers the full pipeline through Cross-Validation, Scientific Methodology, Error Analysis, and Interpretability (previously it stopped at Feature Engineering). The biggest addition is **§10 — Train/Test Split and Data Leakage**, which directly targets the bug you already know your old code has (transforming before splitting). This version's audit table and refactor stage order are restructured around that split boundary, since it changes *when* every other stage is allowed to run.

---

## PHASE 1 — Audit (DO NOT MODIFY ANY CODE)

```
You are an experienced Machine Learning Engineer doing a code review.

I have attached two things:
1. My current ML project (MLP + Gaussian Mixture Model).
2. ML_PIPELINE_REFERENCE.md — my instructor's pipeline, used as conceptual reference only.
   It now covers the full pipeline: EDA through Model Interpretability.

Your task in this phase is ONLY to audit. Do not modify, do not suggest fixes yet.
Produce a structured engineering report with the sections below.

---

⚠️ CRITICAL IMPLEMENTATION CONSTRAINT
This project is for a university course. We are NOT allowed to use high-level ML libraries.
Forbidden: scikit-learn, PyTorch, TensorFlow, Keras, XGBoost, LightGBM, imblearn, or any library
that provides a model, scaler, encoder, imputer, sampler, splitter, or cross-validator as a
pre-built class. This includes sklearn.model_selection (train_test_split, KFold, StratifiedKFold,
cross_val_score) — these must also be implemented from scratch.

Allowed: numpy, pandas, matplotlib, scipy.stats (for statistical tests only — t-test, bootstrap
confidence intervals — not for model fitting), and standard Python.

Everything — MLP forward/backward pass, GMM EM algorithm, scaling, imputation, encoding,
outlier detection, train/test split, stratified split, K-Fold cross-validation — must be
implemented from scratch using numpy math.

When referencing ML_PIPELINE_REFERENCE.md:
- Use it to understand WHAT should be done and WHY.
- Do NOT use its sklearn/imblearn code patterns for implementation — those exist in the
  document only as conceptual illustration, marked explicitly as such.
- Translate every concept into numpy-level math.

---

⚠️ SPECIFIC BUG TO LOOK FOR — DATA LEAKAGE
I already suspect my old code computes preprocessing statistics (mean, std, scaling
parameters, etc.) on the FULL dataset BEFORE splitting into train/test. This is described
in ML_PIPELINE_REFERENCE.md §10. Explicitly check for this in Step 4 and Step 9 below —
do not skip this check even if nothing else looks wrong.

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

Specifically identify: at what point in the execution does the train/test split occur,
relative to scaling, imputation, encoding, and outlier handling? State this explicitly —
this is the most important fact in this entire audit.

---

## Step 3 — Pipeline Coverage Audit

Compare the existing implementation against ML_PIPELINE_REFERENCE.md.
Produce this exact table:

| Pipeline Stage                       | Exists? | Where in code? | Correct? | Missing/Broken? | Notes |
|---------------------------------------|---------|-----------------|----------|-------------------|-------|
| Problem type identified               |         |                 |          |                   |       |
| EDA — global overview                 |         |                 |          |                   |       |
| EDA — univariate distribution         |         |                 |          |                   |       |
| EDA — multivariate correlation        |         |                 |          |                   |       |
| Missing value analysis                |         |                 |          |                   |       |
| Missing value handling                |         |                 |          |                   |       |
| Outlier detection                     |         |                 |          |                   |       |
| Outlier treatment                     |         |                 |          |                   |       |
| **Train/test split (§10)**            |         |                 |          |                   |       |
| **Split occurs BEFORE all fitting?**  |         |                 |          |                   |       |
| **Stratified split used (if needed)** |         |                 |          |                   |       |
| Feature scaling                       |         |                 |          |                   |       |
| — fit on train only?                  |         |                 |          |                   |       |
| Categorical encoding                  |         |                 |          |                   |       |
| — fit on train only?                  |         |                 |          |                   |       |
| Class imbalance handling              |         |                 |          |                   |       |
| — applied to train only?              |         |                 |          |                   |       |
| Feature engineering / construction    |         |                 |          |                   |       |
| **Baseline model (§11)**              |         |                 |          |                   |       |
| **Model selection rationale (§12)**   |         |                 |          |                   |       |
| **Bias-variance diagnosis (§13)**     |         |                 |          |                   |       |
| **Hyperparameter tuning method (§15)**|         |                 |          |                   |       |
| Model training — MLP                  |         |                 |          |                   |       |
| Model training — GMM                  |         |                 |          |                   |       |
| Evaluation metrics                    |         |                 |          |                   |       |
| **K-Fold cross-validation (§17)**     |         |                 |          |                   |       |
| **Results reported as μ ± σ?**        |         |                 |          |                   |       |
| **Single-variable experiment log (§18)**|       |                 |          |                   |       |
| Visualizations                        |         |                 |          |                   |       |
| **Error analysis (§19)**              |         |                 |          |                   |       |
| **Interpretability discussion (§20)** |         |                 |          |                   |       |

Bold rows are new in this audit version — pay particular attention to them, since they were
not covered in the previous audit and are likely fully absent from the current code.

---

## Step 4 — Data Flow & Leakage Check

Trace how data moves from source to model output.
For every transformation write:
    Input → shape/type → Operation (numpy function used) → Output → shape/type

For each transformation that involves fitting a statistic (mean, std, min/max, category
frequency, IQR bounds, etc.), explicitly answer:
    - Is this statistic computed before or after the train/test split?
    - If before: this is a LEAKAGE BUG. Flag it clearly and explain the consequence
      using the reasoning in ML_PIPELINE_REFERENCE.md §10.1.

---

## Step 5 — Feature Analysis

For every feature in the dataset, fill this table:

| Feature Name | Data Type | Missing Values? | Outliers? | Scaling Applied? | Encoding Applied? | Engineered? |
|--------------|-----------|-----------------|-----------|------------------|--------------------|--------------|

---

## Step 6 — Output Inventory

List every file the project currently saves or prints.
Classify each as: plot / log / metric / model weight / experiment-log / other.
Note whether it is currently organized or dumped into the root directory.

---

## Step 7 — Model Analysis

For each model (MLP, GMM):
- Algorithm and mathematical formulation (brief)
- Architecture / hyperparameters
- Loss function
- Optimization procedure (gradient descent / EM — specify exactly what is implemented)
- Stopping criterion
- Is there a documented baseline this model is being compared against? (§11)
- Is there any hyperparameter tuning logic, and if so, does it change one variable at a
  time or several at once? (§18.3)
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

For each missing or broken stage from Step 3, explain:
- What is missing or broken
- Why it matters for this specific dataset and these two models
- What the numpy-level implementation would conceptually involve
- What the expected effect on MLP and GMM performance/validity would be

Pay special attention to anything flagged as a leakage bug in Step 4 — explain concretely
how it would have inflated previously reported metrics, and by roughly how much you'd
expect the corrected numbers to drop (qualitative estimate is fine).

---

## Step 10 — Refactoring Plan

Produce a numbered implementation roadmap.
For each step: what changes, what file it touches, what the output will be.
Do NOT write code. This is a plan only.

The plan must respect this constraint: the train/test split (and stratified split, if the
data is imbalanced) happens FIRST, before any other preprocessing stage is implemented or
re-implemented. Every later stage in the plan operates within that split boundary.

Follow this output folder structure in your plan:
    results/
        logs/          ← training logs, metric tables, console output, experiment logs
        plots/         ← all matplotlib figures, organized by pipeline stage
        metrics/       ← final quantitative results (accuracy, loss, BIC, F1, μ±σ, etc.)
        models/        ← saved weights or parameters if applicable
        experiments/   ← one file per single-variable experiment run (§18.3)
```

---

## PHASE 2 — Refactor (one pipeline stage at a time)

```
Using your Phase 1 engineering report, now refactor the project.

⚠️ SAME CONSTRAINTS AS PHASE 1:
Do NOT use scikit-learn, PyTorch, TensorFlow, Keras, imblearn, or any pre-built ML class,
including sklearn.model_selection. Implement everything from scratch in numpy.
ML_PIPELINE_REFERENCE.md is the conceptual guide — translate its ideas into numpy math.

Rules:
- Implement exactly ONE pipeline stage per reply.
- Do not touch code outside the current stage.
- Before writing code, state in one paragraph: what you are implementing and why.
- After writing code, produce a summary block (format below).
- Justify every decision by referencing the relevant section of ML_PIPELINE_REFERENCE.md.
- Fix random seeds for reproducibility: np.random.seed(42).
- Stop after each stage. Wait for my confirmation before continuing to the next one.

---

Output folder structure — enforce this strictly:

    results/
        logs/          ← print all training progress, epoch losses, EM iterations,
                          experiment summaries here
        plots/         ← save all matplotlib figures here, named by stage
                          Naming: {stage}_{description}.png
                          Examples: 03_eda_histograms.png, 10_split_class_balance.png
        metrics/       ← save final quantitative results as .txt or .csv
                          Examples: mlp_final_metrics.txt, gmm_bic_scores.csv,
                          cv_results_mu_sigma.txt
        models/        ← save final model parameters as .npy if applicable
        experiments/   ← one file per single-variable tuning run, e.g.
                          experiment_001_lr.txt, experiment_002_hidden_units.txt

Every plt.savefig() call must save into results/plots/.
Every print() of a metric or score must also be written to results/logs/.
Never show plots interactively (plt.show()) — always save to file.

---

Implement in this order. Wait for my confirmation before moving to the next stage.

Stage 01 — Project structure cleanup
         Create the results/ folder hierarchy. Rename files if needed.
         Add a config.py or constants section with all magic numbers, including
         RANDOM_SEED = 42 used everywhere.

Stage 02 — Data loading
         Robust CSV/data loading. Print shape, dtypes, head.
         Log output → results/logs/02_data_loading.txt

Stage 03 — EDA
         Global overview: shape, types, missing count, value ranges.
         Univariate: histogram + KDE for every numerical feature.
         Multivariate: correlation heatmap.
         All figures → results/plots/03_eda_*.png
         Summary → results/logs/03_eda_summary.txt

Stage 04 — Train/test split (§10) — MOVED EARLY, BEFORE ALL PREPROCESSING
         Implement train/test split from scratch in numpy.
         If classification / imbalanced data: implement stratified split (§10.3).
         This is the leakage boundary — nothing after this stage may compute a fitted
         statistic using both train and test data simultaneously.
         Log split sizes and class balance before/after → results/logs/04_split.txt
         Plot class distribution train vs test → results/plots/04_split_balance.png

Stage 05 — Missing value handling
         Check distribution of each feature using TRAIN DATA ONLY.
         Implement mean imputation, median imputation, and KNN imputation in pure numpy.
         Fit imputation statistics on train, apply to both train and test.
         Log which strategy was chosen per feature → results/logs/05_imputation.txt

Stage 06 — Outlier detection and treatment
         Implement IQR detection in numpy, computed from TRAIN DATA ONLY.
         Apply the same bounds to test data (do not recompute on test).
         Generate boxplots before and after treatment.
         Plots → results/plots/06_outliers_*.png
         Log → results/logs/06_outliers.txt (which points flagged, what action taken)

Stage 07 — Feature scaling
         Implement StandardScaler and MinMaxScaler in numpy (no sklearn).
         Choose method per feature based on ML_PIPELINE_REFERENCE.md §6.4.
         Fit on train only. Transform train and test using train-fitted parameters.
         Log scaling parameters (mean, std per feature) → results/logs/07_scaling.txt

Stage 08 — Encoding
         Identify any categorical features. Apply appropriate encoding in numpy,
         with any fitted mapping (e.g. frequency, target encoding) computed from
         TRAIN DATA ONLY.
         If no categorical features exist, document this explicitly and skip.
         Log → results/logs/08_encoding.txt

Stage 09 — Class imbalance
         Compute class distribution on TRAIN DATA ONLY.
         Plot class bar chart → results/plots/09_class_dist.png
         If imbalanced: implement oversampling or class weighting in numpy,
         applied to train data only — never to test.
         Log → results/logs/09_imbalance.txt

Stage 10 — Feature engineering
         Based on EDA findings, create at least one engineered feature using the
         three construction strategies in §9.9 (relationship, structural, or
         geometric/polynomial) — whichever fits this dataset.
         Any fitted statistic used in construction (e.g. an aggregation) must be
         computed from train data only and applied consistently to test.
         Justify each new feature using domain knowledge.
         Plot: distribution of engineered features → results/plots/10_engineered_*.png
         Log → results/logs/10_feature_engineering.txt

Stage 11 — Baseline model (§11)
         Implement the simplest reasonable baseline for each task:
           - MLP task: simple linear/logistic model with default settings
           - GMM task: K-Means with a small k, OR a single-Gaussian fit
         Evaluate the baseline using the metrics relevant to the task.
         Log baseline performance → results/logs/11_baseline.txt
         This number becomes the reference point every later result is compared against.

Stage 12 — MLP training
         Refactor the existing MLP to be clean and modular.
         Log every epoch: loss, accuracy → results/logs/12_mlp_training.txt
         Save loss curve → results/plots/12_mlp_loss_curve.png
         Save final weights → results/models/mlp_weights.npy
         Compare final result to the Stage 11 baseline explicitly in the log.

Stage 13 — GMM training
         Refactor the existing GMM (EM algorithm) to be clean and modular.
         Log every EM iteration: log-likelihood → results/logs/13_gmm_training.txt
         Save BIC/AIC curve → results/plots/13_gmm_bic.png
         Save final parameters (means, covariances, weights) → results/models/gmm_params.npy
         Compare final result to the Stage 11 baseline explicitly in the log.

Stage 14 — Hyperparameter tuning (§14, §15, §18.3)
         For at least one hyperparameter per model (e.g. MLP: learning rate or hidden
         units; GMM: number of components k), run a single-variable experiment sweep:
         change ONLY that one hyperparameter across several values, holding everything
         else constant.
         Save one file per run → results/experiments/experiment_NNN_{param}.txt
         Summarize the sweep in a table (param value → metric) →
         results/logs/14_hyperparameter_sweep.txt
         Plot metric vs. hyperparameter value → results/plots/14_tuning_{param}.png

Stage 15 — Evaluation metrics
         Compute and save all metrics.
         MLP: accuracy, precision, recall, F1, confusion matrix.
         GMM: BIC, AIC, log-likelihood, silhouette score (numpy implementation).
         All metrics → results/metrics/
         Confusion matrix plot → results/plots/15_confusion_matrix.png

Stage 16 — Cross-validation (§17)
         Implement K-Fold (and Stratified K-Fold if classification) from scratch in numpy,
         operating only within the training set.
         Run the final chosen model configuration across K folds.
         Report results as μ ± σ, with the stability interpretation from §17.4.
         Log → results/logs/16_cross_validation.txt
         Save → results/metrics/cv_results_mu_sigma.txt

Stage 17 — Final test-set evaluation
         Using the model and hyperparameters selected via Stages 14–16, evaluate ONCE
         on the held-out test set from Stage 04. This must be the only place in the
         entire codebase where the test set is used for evaluation.
         Log → results/logs/17_final_test_evaluation.txt
         Save → results/metrics/final_test_results.txt

Stage 18 — Error analysis (§19)
         Identify the samples with the largest errors (MLP) or lowest likelihood (GMM).
         Group them by shared characteristics and report patterns found.
         Log → results/logs/18_error_analysis.txt
         Plot: distribution of errors by relevant subgroup → results/plots/18_error_*.png

Stage 19 — Final visualization pass
         Review all saved plots. Ensure every plot has:
           - title
           - axis labels
           - legend if applicable
           - consistent style (set once at top of file)
         Generate results/plots/README.txt listing every plot and its interpretation.

Stage 20 — Interpretability notes (§20)
         Write a short global and local interpretability discussion for each model
         (e.g. which features matter most overall for the MLP; which features define
         each Gaussian component in the GMM). This can be analysis/markdown rather
         than new code, but save it as results/logs/20_interpretability.txt so it is
         available alongside everything else for the essay.

---

After each stage, produce this exact summary block:

## Stage [N] Summary
**What changed:** [list of modified/added functions and files]
**Why:** [reference to ML_PIPELINE_REFERENCE.md section]
**numpy implementation note:** [key formula or algorithm used, e.g. x' = (x - μ) / σ]
**Expected impact:** [effect on MLP/GMM performance or validity]
**Essay note:** [one sentence describing this stage for use in the essay methodology section]
```

---

## AI Implementation Checklist (Updated)

Replace the previous checklist at the end of `ML_PIPELINE_REFERENCE.md` with this version:

```markdown
## AI Implementation Checklist

⚠️ No scikit-learn, PyTorch, or any pre-built ML class — including sklearn.model_selection.
Pure numpy only.

Before training any model, verify every box is checked:

### Data Understanding
- [ ] Problem type identified (Supervised / Unsupervised / RL)
- [ ] Dataset shape printed (n_samples, n_features)
- [ ] Feature types identified (numeric / categorical / bool)
- [ ] Missing value counts per feature logged
- [ ] Value ranges and distributions visualized

### Split & Leakage (do this BEFORE any other preprocessing)
- [ ] Train/test split implemented from scratch
- [ ] Stratified split used if data is imbalanced or classification
- [ ] Every fitted statistic (mean, std, IQR bounds, encoding maps) computed from
      TRAIN DATA ONLY
- [ ] Test set is transformed using train-fitted parameters, never refit
- [ ] Test set evaluated exactly once, at the very end of the pipeline

### Preprocessing (fit on train, applied to train + test)
- [ ] Missing values handled (strategy chosen based on train-set distribution)
- [ ] Outliers detected (IQR, from train) and decision documented
- [ ] Feature scaling applied (fit on train only)
- [ ] Categorical encoding applied or explicitly skipped with justification
- [ ] Class imbalance checked and addressed on train data only

### Feature Engineering
- [ ] At least one engineered feature created and justified (§9.9)
- [ ] Redundant features removed or documented

### Modeling Discipline
- [ ] A baseline model is implemented and its performance logged before the main model
- [ ] Model choice is justified against data geometry (§12 — No Free Lunch)
- [ ] Bias/variance behavior is discussed using train vs. validation gap (§13)
- [ ] Hyperparameter tuning changes ONE variable at a time, logged per run (§18.3)
- [ ] K-Fold cross-validation implemented from scratch, used only within train data
- [ ] Final metrics reported as μ ± σ across folds

### Reproducibility
- [ ] np.random.seed(42) set at entry point
- [ ] All file paths use os.path.join(), not hard-coded strings

### Output Organization
- [ ] results/logs/          ← all training logs and metric prints
- [ ] results/plots/         ← all matplotlib figures (no plt.show())
- [ ] results/metrics/       ← final scores saved as .txt or .csv
- [ ] results/models/        ← saved numpy arrays for weights/parameters
- [ ] results/experiments/   ← one file per hyperparameter sweep run

### Essay Readiness
- [ ] Every plot has title, axis labels, legend
- [ ] Every metric is saved to file (not just printed)
- [ ] results/plots/README.txt exists, listing every figure and its purpose
- [ ] Stage summaries written (the "Essay note:" line from each Stage Summary block)
- [ ] Error analysis findings documented (§19)
- [ ] Interpretability notes written for both models (§20)
```
