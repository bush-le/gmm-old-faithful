# ML Pipeline Reference
**Version:** 1.0  
**Scope:** Steps 1–9 (Problem Framing → Feature Engineering). Steps 10–20 are placeholders — not yet covered in class.  
**Audience:** Human practitioners and AI coding agents.  
**How to use:** Follow this document top-to-bottom before writing a single line of training code. Every section states *what*, *why*, and *when not to* — read all three.

---

## ⚠️ Not Yet Covered (Placeholders)

The following topics belong to the full pipeline but have **not been taught yet**. Do not implement or assume defaults for these until the corresponding lecture notes are added:

| # | Topic |
|---|-------|
| 10 | Data Leakage — definition, causes, prevention |
| 11 | Train / Validation / Test Split — stratification, distribution matching |
| 12 | Baseline Thinking — problem framing, resource constraints, "good enough" |
| 13 | No Free Lunch Theorem |
| 14 | Bias–Variance Tradeoff, systematic error |
| 15 | Hyperparameter Tuning |
| 16 | Evaluation Metrics |
| 17 | Cross-Validation |
| 18 | Experimental Methodology |
| 19 | Reliability of Performance Numbers |
| 20 | Error Analysis |
| 21 | Model Interpretability / Inferability |

---

## Full Pipeline at a Glance

```
1.  Identify the ML paradigm
         ↓
2.  Understand the three pillars
         ↓
3.  Geometric survey of the data (EDA)
         ↓
4.  Handle missing values
         ↓
5.  Detect & treat outliers
         ↓
6.  Feature scaling
         ↓
7.  Encode categorical variables
         ↓
8.  Handle class imbalance
         ↓
9.  Feature engineering
         ↓
   [Steps 10–21 — not yet covered]
```

> **Rule #0 — Never start from the algorithm.**  
> Start from the data. Wrong assumptions at step 3 corrupt every later step. A mediocre algorithm on well-prepared data beats a fancy algorithm on raw data.

---

## 1. ML Mind Map — Identify the Problem Type

The very first question is not *"which model do I use?"*

The first question is:

```
Does the data have labels?
    YES → Supervised Learning
    NO  → Does a reward signal exist?
              YES → Reinforcement Learning
              NO  → Unsupervised Learning
```

Choosing the wrong paradigm invalidates every downstream decision. Lock this in before touching the data.

---

## 2. Three Pillars of ML

### 2.1 Supervised Learning

- Data: `(X, y)` pairs — inputs with known labels.
- Goal: Learn `f: X → y` that generalizes to unseen samples.
- Tasks: Classification, Regression.
- Typical algorithms: Linear/Logistic Regression, Decision Tree, Random Forest, SVM, MLP (Neural Network), Gradient Boosting.

#### ⚠️ Pitfall — Label Quality in the Real World

Public datasets (Kaggle, Hugging Face, UCI) come pre-labeled. Real industry data does **not**.

```
Raw unlabeled data
    ↓
Manual labeling by human judgment
    ↓
Subjective / inconsistent labels
    ↓
Model learns human bias, not ground truth
    ↓
Biased predictions in production
```

**Consequence:** Label quality is the hard ceiling on supervised model performance. No algorithm can fix corrupted labels.

---

### 2.2 Unsupervised Learning

- Data: `X` only — no labels.
- Goal: Discover hidden structure.
- Tasks: Clustering, Dimensionality Reduction, Density Estimation, Anomaly Detection.
- Typical algorithms: K-Means, GMM (Gaussian Mixture Model), PCA, DBSCAN, t-SNE, UMAP.

---

### 2.3 Reinforcement Learning

- Setup: Agent ↔ Environment loop.

```
Observe state s_t
    ↓
Select action a_t (policy)
    ↓
Receive reward r_t
    ↓
Transition to state s_{t+1}
    ↓
Update policy
```

- Goal: Maximize cumulative discounted reward `Σ γ^t r_t`.

---

## 3. Geometric Survey of the Data (EDA)

> Think of every sample as a point in an N-dimensional space. Before preprocessing, understand the **shape** of that space.

**Why this matters:** Preprocessing decisions (scaling method, imputation strategy, encoding) are determined by data geometry — not by defaults.

### ⚠️ Pitfall — Skipping EDA

```
Raw data
    ↓  (no exploration)
Wrong assumptions
    ↓
Wrong preprocessing
    ↓
Wrong algorithm choice
    ↓
Poor performance with no clear diagnosis
```

---

### 3.1 Lens 1 — Global Overview

Questions to answer before anything else:

| Question | Tool |
|----------|------|
| How many samples? How many features? | `df.shape` |
| What are the feature types? (numeric, categorical, bool, datetime) | `df.dtypes`, `df.info()` |
| Are there missing values? | `df.isnull().sum()` |
| What are the value ranges? | `df.describe()` |
| Any obvious anomalies or impossible values? | visual inspection |

---

### 3.2 Lens 2 — Univariate Distribution

Examine each feature independently. Every feature has its own probability distribution.

Tools:
- Histogram / KDE plot — detect skew, multimodality, gaps.
- Box plot — visualize spread and outliers.
- Summary statistics: mean, median, std, skewness, kurtosis.

**What to determine per feature:**

```
Is the distribution roughly symmetric (Gaussian)?
    YES → mean is representative; standardization safe
    NO  → distribution is skewed; use median; consider log transform
```

---

### 3.3 Lens 3 — Multivariate Relationships

Study interactions between features.

Tools:
- Correlation matrix (heatmap).
- Scatter matrix / pair plot.
- Mutual information.

**Correlation interpretation:**

| `r` value | Meaning | Suggested Action |
|-----------|---------|-----------------|
| `r ≈ ±1` | Strong co-linear relationship | Consider removing one, or combine |
| `0.3 < |r| < 0.7` | Moderate correlation | Consider basis change / interaction term |
| `r ≈ 0` | Independent | Keep both; check non-linear relationships separately |

> **Warning:** Correlation only captures linear relationships. Two features can have `r = 0` but strong non-linear dependence. Always supplement with scatter plots.

> **Warning:** Never remove a feature based on correlation alone. Consult domain knowledge and observed model behavior first.

---

### 3.4 Why Distribution Knowledge Drives Everything

Distribution is the input to every downstream decision:

| Decision | Requires Distribution Knowledge |
|----------|--------------------------------|
| Missing value imputation method | Yes — mean vs median vs KNN |
| Outlier detection threshold | Yes — IQR bounds depend on spread |
| Scaling method | Yes — Gaussian → standardize; bounded → min-max |
| Feature engineering targets | Yes — skewed features benefit from log transform |
| Model family selection | Yes — linear vs non-linear separability |

---

## 4. Missing Values

> Handle missing values before any model training. The correct strategy depends on the data type and the distribution.

---

### 4.1 Numerical Features

#### Distribution is approximately Gaussian (symmetric)

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
```

**Why:** Mean is the maximum-likelihood estimate for a Gaussian distribution.

#### Distribution is skewed or has outliers

```python
imputer = SimpleImputer(strategy='median')
```

**Why:** Median is robust to extreme values. Mean is pulled toward outliers and destroys the true distribution center.

**Concrete example:**

```
Salary values: [20k, 22k, 25k, 28k, 5000k]
Mean ≈ 1019k  ← severely distorted
Median = 25k  ← representative
```

Filling a missing salary with `1019k` is incorrect and corrupts the feature distribution for every downstream step.

#### Multivariate missing pattern (missingness related to other features)

```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
```

**Why:** KNN uses the local neighborhood in feature space to estimate the missing value, preserving local structure better than any global statistic.

---

### 4.2 Categorical Features

Mean and median are undefined for categorical variables. Options:

| Strategy | When to Use |
|----------|-------------|
| **Mode** (`most_frequent`) | Distribution is clearly dominated by one category |
| **New category `"Unknown"`** | Missingness itself may be informative |
| **Domain-specific fill** | Business rules dictate the correct value |
| **KNN Imputer** (after encoding) | Missingness correlates with other features |

> **AI Agent Note:** Never apply `strategy='mean'` to a categorical column. Always check `dtype` first.

---

### 4.3 Decision Tree for Imputation Strategy

```
Feature is numerical?
    YES → Distribution is Gaussian?
              YES → Mean imputation
              NO  → Median imputation
          Missingness correlates with other features?
              YES → KNN Imputer (overrides mean/median)
    NO  (categorical) →
          Missingness itself is informative?
              YES → Add "Unknown" category
              NO  → Mode imputation or domain fill
```

---

## 5. Outlier Detection and Treatment

An **outlier** is a sample that deviates significantly from the expected distribution.

**Outliers may be:**
- Data entry or sensor errors → should be corrected or removed.
- Genuine rare events → **must be kept** (fraud, disease, faults are precisely the minority events a model needs to learn).

> **Rule:** Never blindly remove outliers. First determine: is this an error, or is this signal?

---

### 5.1 Why Outliers Break ML

#### They hijack the loss function

Linear Regression (OLS) minimizes `Σ (y_i - ŷ_i)²`. Squaring the error means one extreme point contributes an enormous loss:

```
Normal points: 5 points with error ≈ 1  →  loss += 5
Outlier:       1 point  with error = 50 →  loss += 2500
```

The optimizer pulls the entire model toward one outlier. The fitted line no longer represents the bulk of the data.

#### They corrupt scaling

Min-Max scaling is defined as:

```
x' = (x - x_min) / (x_max - x_min)
```

One outlier extends `x_max` or `x_min` dramatically, compressing all normal values into a tiny range near 0 or 1. The feature effectively loses information.

#### They distort imputation

If a feature has outliers and you use mean imputation for missing values, the imputed values will reflect the outlier rather than the true distribution center. This is the reason the distribution check in Step 4 must come **before** imputation.

---

### 5.2 Detection Methods

#### Box Plot (visual, fast)

```
|----[Q1----median----Q3----|    ●
           IQR                outlier
```

Use during EDA for quick visual identification.

#### IQR Rule (numerical)

```
Q1  = 25th percentile
Q3  = 75th percentile
IQR = Q3 - Q1

Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```

Points outside `[Lower, Upper]` are flagged as potential outliers.

```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df[col] < lower) | (df[col] > upper)]
```

#### Other Methods (beyond current course scope)

| Method | Characteristic |
|--------|---------------|
| Z-score | Assumes Gaussian; flags points > 3σ from mean |
| Isolation Forest | Tree-based; works in high dimensions |
| Local Outlier Factor (LOF) | Density-based; detects local anomalies |

---

### 5.3 Treatment Strategies

| Strategy | When to Use | Code Pattern |
|----------|------------|-------------|
| **Remove** | Confirmed data errors | `df = df[df[col].between(lower, upper)]` |
| **Cap (Winsorize)** | Extreme but plausible values | `df[col] = df[col].clip(lower, upper)` |
| **Log transform** | Right-skewed distribution | `df[col] = np.log1p(df[col])` |
| **Square root transform** | Moderate skew | `df[col] = np.sqrt(df[col])` |
| **Box-Cox** | Strictly positive; flexible | `scipy.stats.boxcox(df[col])` |
| **Yeo-Johnson** | Zero or negative values allowed | `sklearn PowerTransformer(method='yeo-johnson')` |

**Geometric interpretation of transforms:** Log and Box-Cox compress the right tail, pulling extreme values closer to the bulk of the distribution. This changes the shape of the feature space from a long ellipsoid into something closer to spherical — the same goal as scaling.

---

## 6. Feature Scaling

> Most ML algorithms operate using **distance** or **gradient** in feature space. Features on different scales distort both.

---

### 6.1 Why Scaling is Necessary

Consider:

```
Feature A: Age       → range [0, 100]
Feature B: Salary    → range [0, 100,000,000]
```

Euclidean distance between two samples:

```
d = sqrt((ΔAge)² + (ΔSalary)²)
           ≈ negligible     ≈ dominates
```

Feature A contributes essentially nothing to any distance computation. The algorithm is operating in a space dominated by Feature B.

**Geometric view:**

```
Unscaled space:                Scaled space:
    ●                              ●  ●
              ●           →     ●     ●
●                              ●  ●
long ellipsoid                 approximate sphere
```

---

### 6.2 Consequences Per ML Paradigm

| Paradigm | Algorithm | Consequence Without Scaling |
|----------|-----------|----------------------------|
| Supervised | KNN | Distance dominated by large-scale features |
| Supervised | SVM (RBF) | Kernel distances distorted |
| Supervised | Linear/Logistic Regression | Gradient descent converges slowly |
| Supervised | Neural Network (MLP) | Slow convergence; unstable training |
| Unsupervised | PCA | Variance dominated by large-scale features; principal components biased |
| Unsupervised | K-Means | Euclidean distances distorted; wrong cluster assignments |
| Unsupervised | GMM | Covariance estimation distorted; numerically unstable |
| RL | Any | Large state dimensions dominate policy gradient; slow/unstable convergence |

> **Note:** Tree-based algorithms (Decision Tree, Random Forest, XGBoost) are **invariant to feature scaling** because they split on thresholds, not distances. You may skip scaling for these.

---

### 6.3 Scaling Methods

#### Standardization (Z-score normalization)

```
x' = (x - μ) / σ
```

Result: `mean = 0`, `std = 1`. Does **not** bound values to a fixed range.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Use when:**
- Distribution is approximately Gaussian.
- No fixed known minimum/maximum.
- Algorithm: Linear/Logistic Regression, SVM, MLP, PCA, K-Means, GMM.
- Default choice when uncertain.

#### Min-Max Scaling

```
x' = (x - x_min) / (x_max - x_min)    →    x' ∈ [0, 1]
```

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Use when:**
- Feature has a known, meaningful minimum and maximum.
- Neural network uses bounded activation (sigmoid, tanh).
- Image pixel values (0–255 → 0–1).

**⚠️ Sensitive to outliers.** One extreme value compresses all other values. If outliers exist, treat them (Step 5) before applying Min-Max.

#### Robust Scaling (referenced by instructor, not detailed in class)

```
x' = (x - median) / IQR
```

Uses median and IQR instead of mean and std. Inherently outlier-resistant. Appropriate when outliers cannot be removed.

---

### 6.4 Scaling Selection Decision Tree

```
Features have known, fixed bounds AND no significant outliers?
    YES → Min-Max Scaling
    NO  →
        Significant outliers that cannot be removed?
            YES → Robust Scaling
            NO  → Standardization (default)

Additionally:
    Neural network with sigmoid/tanh activations → Min-Max preferred
    PCA, SVM, Linear models, GMM → Standardization
    Tree-based models → No scaling required
```

> **AI Agent Rule:** Always fit the scaler **only** on training data. Apply (transform only) to validation and test data. Never fit on the full dataset — that causes data leakage (Step 10).

```python
# CORRECT
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# WRONG — leaks test statistics into training
scaler.fit(X_all)
```

---

## 7. Encoding Categorical Variables

> ML models operate on vectors in ℝⁿ. Categorical text labels must be converted to numbers — but the **method** of conversion matters, because it encodes geometric assumptions.

---

### 7.1 Why Encoding Method Matters

Encoding is not just type conversion. It defines the geometric relationship between categories in the feature space. A wrong encoding introduces artificial ordering or artificial distances that the model will learn as if they were real.

---

### 7.2 Label Encoding

```
Red   → 0
Blue  → 1
Green → 2
```

**Problem:** Implies `Green > Blue > Red` and `|Green - Blue| = |Blue - Red|`. If these categories are independent (no natural order), the model learns a false ordinal structure.

**Use only when:** The categories genuinely have a natural order AND the model is tree-based (tree splits ignore the numerical magnitude).

---

### 7.3 Ordinal Encoding

Categories with a true natural ranking:

```
Small  → 0
Medium → 1
Large  → 2
```

This is correct when the order is real. Logically equivalent to Label Encoding but semantically explicit.

```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
```

---

### 7.4 One-Hot Encoding

Each category becomes a separate binary dimension:

```
Country  →  Vietnam  Japan  USA
Vietnam      1        0      0
Japan        0        1      0
USA          0        0      1
```

**Advantages:** No false ordering. Geometrically, each category is equidistant from all others.  
**Disadvantage:** Dimensionality grows linearly with the number of categories.

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
```

**Drop first category** to avoid the dummy variable trap (perfect multicollinearity) in linear models:

```python
OneHotEncoder(drop='first')
```

---

### 7.5 Soft / Probabilistic Encoding (Instructor's Idea)

Instead of hard one-hot `[1, 0, 0, 0, 0]`, use a soft representation:

```
[0.9, 0.1, 0.1, 0.1, 0.1]  ← Vietnam with soft mass on neighbors
```

**Motivation:** Reduces the effective contribution of the added dimensions while preserving category identity. Related to label smoothing and embedding techniques in deep learning. Useful when one-hot creates severe sparsity.

**Note:** This is not a standard sklearn encoder. Implement as a custom transform or use learned embeddings in a neural network.

---

### 7.6 High-Cardinality Encodings

When a categorical feature has many unique values (e.g., ZIP code, product ID), one-hot encoding is impractical.

#### Frequency / Count Encoding

```
Replace each category with its frequency in the training set.
Vietnam → 0.45  (appears in 45% of rows)
Japan   → 0.30
USA     → 0.25
```

```python
freq_map = df['country'].value_counts(normalize=True)
df['country_freq'] = df['country'].map(freq_map)
```

#### Target Encoding

Replace each category with the mean target value for that category.

```
Vietnam → mean(y | country == Vietnam)
```

**⚠️ Danger: Data leakage.** Must be computed inside cross-validation folds only, never on the full dataset. See Step 10 (Data Leakage) — placeholder.

---

### 7.7 Curse of Dimensionality

One-hot encoding with many categories produces a high-dimensional, sparse feature space.

**Consequences:**
- Distance-based algorithms require exponentially more data to populate the space.
- Memory and compute increase.
- Models need more data to generalize.

**Geometric intuition:** In high dimensions, all points become approximately equidistant. The concept of "nearest neighbor" loses meaning.

### Impact by ML Paradigm

| Paradigm | Impact of High Dimensionality |
|----------|------------------------------|
| Supervised (distance-based) | KNN degrades; SVM kernel bandwidth becomes harder to tune |
| Supervised (linear) | Regularization more critical; risk of overfitting |
| Supervised (tree) | Less affected; trees split sequentially and ignore irrelevant features |
| Unsupervised | PCA becomes necessary; K-Means and GMM cluster quality degrades |
| RL | Observation space explosion; policy learning slows dramatically |

---

### 7.8 Encoding Selection Decision Tree

```
Does the feature have a natural, meaningful order?
    YES → Ordinal Encoding
    NO  →
        Number of unique categories is small (≤ ~10–15)?
            YES → One-Hot Encoding
                  (if linear model: drop='first' to avoid multicollinearity)
            NO  →
                Target variable is available and leakage is controlled?
                    YES → Target Encoding (inside CV folds only)
                    NO  → Frequency Encoding or Count Encoding
```

**Core principle (instructor):** Minimize the number of dimensions while preserving meaningful information. Never create artificial relationships between independent categories.

---

## 8. Class Imbalance

> A model trained on imbalanced data is not learning the task — it is learning the class distribution.

---

### 8.1 The Problem

Real-world datasets are often dominated by one class:

```
Healthy patients:  99%
Cancer patients:    1%
```

A model that predicts "Healthy" for every sample achieves **99% accuracy** while being completely useless for the actual task (detecting cancer).

**Why this happens:**  
Cross-entropy loss (and most loss functions) minimize total error. With 99% healthy samples, always predicting "Healthy" minimizes loss. The model never needs to learn the minority class.

---

### 8.2 Correct Evaluation Requires Non-Accuracy Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Precision** | Of all predicted positives, how many are actually positive? |
| **Recall** | Of all actual positives, how many were caught? |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Discrimination ability across all thresholds |

High accuracy with low recall on the minority class is a **failure mode**, not a success.

> **AI Agent Note:** For imbalanced datasets, always report Precision, Recall, and F1 per class — not just overall accuracy.

---

### 8.3 Solutions

#### Oversampling

Randomly duplicate minority class samples.

```python
# Manual approach
minority = df[df['label'] == 1]
df_balanced = pd.concat([df, minority.sample(n=desired_count, replace=True)])
```

**Downside:** Model may overfit to the exact duplicated samples.

#### SMOTE (Synthetic Minority Over-sampling Technique)

Generate synthetic minority samples by interpolating between existing minority samples in feature space.

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
```

**Advantage over oversampling:** Creates new examples rather than duplicates; reduces overfitting.  
**Constraint:** Apply SMOTE only to training data, never to validation or test.

#### Class Weighting

Tell the loss function to penalize errors on minority classes more heavily.

```python
# Sklearn models
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')

# PyTorch
weight = torch.tensor([1.0, 99.0])  # [majority_weight, minority_weight]
criterion = nn.CrossEntropyLoss(weight=weight)
```

**Advantage:** No resampling needed; works directly on the original dataset.

#### Cost-Sensitive Learning

Assign asymmetric misclassification costs. A false negative (missed cancer) may cost far more than a false positive.

Implement by modifying the loss function or using algorithms that accept a cost matrix.

#### Undersampling

Remove majority class samples until classes are balanced.

**Downside:** Loses potentially useful information. Use only when dataset is very large and information loss is acceptable.

---

### 8.4 RL Analogues (Instructor Mention)

In RL, rare but important transitions (analogous to minority class samples) can be lost in uniform experience replay.

| RL Technique | Analogy |
|-------------|---------|
| **Prioritized Experience Replay (PER)** | Oversample high-error transitions |
| **ε-greedy exploration** | Force visits to under-explored states |
| **Entropy-based exploration** | Encourage diversity in actions |
| **UCB (Upper Confidence Bound)** | Prioritize uncertain state-action pairs |

These are beyond the current course scope but the structural parallel to imbalanced data is exact.

---

## 9. Feature Engineering

> Feature Engineering is the highest-leverage step in classical ML. Better features beat better algorithms.

---

### 9.1 Definition

Feature Engineering is the process of transforming raw data into new representations using domain knowledge, so that the learning algorithm can find patterns more easily.

```
Raw data (pull everything from the source)
    ↓
Domain knowledge + exploratory analysis
    ↓
New features with more predictive signal
    ↓
Better learned representations
    ↓
Higher model performance
```

The information content does not change — only the **representation** changes.

---

### 9.2 ML vs. DL

| Setting | Feature Engineering |
|---------|---------------------|
| **Classical ML** | Manual. Practitioner designs features from domain knowledge. |
| **Deep Learning** | Automatic. The network learns hierarchical features from raw data. |

Even in DL, low-level preprocessing (scaling, encoding, handling missing values) still applies. Feature engineering in DL shifts to architecture design and data augmentation.

---

### 9.3 Geometric Interpretation

Each feature is one coordinate axis in feature space.

- **Before feature engineering:** Classes may be entangled in the original coordinate system. No linear boundary can separate them.
- **After feature engineering:** A change of basis (new coordinates) can make classes linearly separable or much easier to cluster.

```
Original space (x₁, x₂):           Engineered space (x₁², x₁·x₂, x₂²):
      class A and B mixed                   class A and B separated
```

A better representation allows even a simple model (linear classifier) to solve a problem that required a complex model in the original space.

---

### 9.4 Common Feature Engineering Operations

| Operation | Example |
|-----------|---------|
| Binning / discretization | Age → `[0–18, 18–35, 35–60, 60+]` |
| Date decomposition | Datetime → Day of week, Month, Season, Hour |
| Interaction terms | `Height × Weight → BMI` |
| Ratio features | `Income / Debt → Debt-to-income ratio` |
| Spatial features | `(lat, lon) → Distance from city center` |
| Rolling statistics | `Transaction history → Avg spend per 30 days` |
| Log transform | Compress skewed distributions |
| Polynomial expansion | `x → x, x², x³` for non-linear relationships |
| Domain-specific aggregations | User behavioral features, product co-occurrence, etc. |

---

### 9.5 What Makes a Feature Good

A useful engineered feature:
- Increases class separability or predictive signal.
- Reduces ambiguity in the input representation.
- Does **not** introduce information that would only be available at inference time (→ data leakage, Step 10).
- Does **not** artificially correlate with the target through the training set statistics (→ target leakage).
- Is interpretable enough to validate with domain knowledge.

---

### 9.6 Feature Selection After Engineering

Creating many candidate features does not mean keeping all of them. After engineering:

1. **Correlation filter:** Remove features with `|r| > 0.95` with another feature (near-duplicate).
2. **Variance threshold:** Remove near-constant features.
3. **Model-based importance:** Use tree feature importances or L1 regularization to identify low-signal features.
4. **Domain sanity check:** Reject features that cannot be computed at inference time.

---

### 9.7 Engineering Workflow

```
1. Collect raw data (pull all available fields from the source)
2. Run EDA (Step 3) — understand what the data represents physically
3. Hypothesize features based on domain knowledge
4. Create candidate features
5. Evaluate signal (correlation with target, model performance delta)
6. Remove redundant or harmful features
7. Repeat before model selection
```

---

## Appendix: Common API Patterns for the Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OrdinalEncoder, OneHotEncoder, PowerTransformer
)
from imblearn.over_sampling import SMOTE

# 1. Load and inspect
df = pd.read_csv('data.csv')
print(df.shape, df.dtypes, df.isnull().sum(), df.describe())

# 2. Split features by type
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

# 3. Outlier detection (IQR)
def iqr_bounds(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# 4. Impute
num_imputer  = SimpleImputer(strategy='median')    # or 'mean' for Gaussian
cat_imputer  = SimpleImputer(strategy='most_frequent')
knn_imputer  = KNNImputer(n_neighbors=5)           # for multivariate patterns

# 5. Scale (fit on train only)
scaler = StandardScaler()     # default
# scaler = MinMaxScaler()     # for bounded features / bounded activations
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)       # transform only

# 6. Encode
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
ord_enc = OrdinalEncoder(categories=[['Small','Medium','Large']])

# 7. Imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)   # TRAIN ONLY

# 8. Transform (outlier-robust)
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X_train[skewed_cols])
```

---

*This document will be extended as subsequent lecture content (Steps 10–21) is covered.*
