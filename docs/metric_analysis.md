# 📊 GMM Metric Analysis: Covariance Collapse & BIC Miscalibration

This document details why the **Log-Likelihood (LL)** and **Bayesian Information Criterion (BIC)** charts in `exp1_vary_k.png` produce anomalous results for $K \ge 3$.

---

## 🔍 Executive Summary

While the **Silhouette Score** correctly identifies $K=2$ as the optimal clustering, both **Log-Likelihood** and **BIC** suggest that $K=4$ or $K=5$ is superior. This is caused by a GMM singularity issue:
1. **Rounding artifact in raw data**: 53 out of 298 observations (17.79%) have the exact eruption duration of `4.000` minutes.
2. **Component covariance collapse**: For $K \ge 3$, one GMM component converges exactly on this "stripe" of points. The variance in that dimension drops to the regularization limit $\epsilon = 10^{-6}$.
3. **Exploded likelihood & BIC miscalibration**: The tiny variance inflates the probability density function (PDF), spiking the total log-likelihood from $\approx -442$ to $\approx -187$. This massive likelihood jump overwhelms the complexity penalty, dropping the BIC score to $\approx 505$ and indicating that $K=4$ is optimal.

---

## 🛠️ The Mechanics of the Collapse

### 1. Data Rounding / Discretization
Historically, the Old Faithful dataset contains human-recorded observations. Park rangers frequently rounded eruption durations to the nearest half-minute or integer minute.
In our raw dataset:
* **`4.000` min**: 53 points (17.79% of the dataset)
* **`2.000` min**: 22 points (7.38% of the dataset)

### 2. EM Algorithm Behavior for $K=4$
When we fit a GMM with $K=4$, the learned parameters for the components are:

```
Component 1: Weight = 0.290, Mean = [-1.34, -1.28], Covariance = [[0.013, 0.012], [0.012, 0.142]]
Component 2: Weight = 0.104, Mean = [-0.58, -0.35], Covariance = [[0.514, 0.377], [0.377, 0.529]]
Component 3: Weight = 0.429, Mean = [0.852, 0.714], Covariance = [[0.083, 0.036], [0.036, 0.222]]
Component 4: Weight = 0.177, Mean = [0.470, 0.575], Covariance = [[1.000e-06, 0.000], [0.000, 0.232]]
```

> [!WARNING]
> Look at **Component 4** (Weight $0.177$, which corresponds exactly to the 53 points at $4.0$ minutes):
> * The variance in the first feature is exactly `1.000e-06` (the `REG_COVAR` value $\epsilon$).
> * The off-diagonal covariance is effectively zero.

### 3. Log-Likelihood Explosion
The Gaussian probability density function is defined as:
$$ \mathcal{N}(x|\mu,\Sigma) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right) $$

For Component 4:
* Determinant $|\Sigma_4| \approx 10^{-6} \times 0.232 = 2.32 \times 10^{-7}$.
* Square root of determinant $|\Sigma_4|^{1/2} \approx 0.000481$.
* Normalization constant:
  $$ \frac{1}{2\pi \times 0.000481} \approx 331 $$

For any of the 53 points with eruption time exactly equal to the mean (raw `4.0` min, standardized `0.470`), the exponent is $0$, resulting in a PDF value of $\approx 331$.
In contrast, points in a normal, non-collapsed component with variance $\approx 0.2$ have a PDF value of only $\approx 0.3$. 

The log-likelihood contribution of one point on this line is:
$$ \ln(p(x_n)) \approx \ln(\pi_4 \times 331) = \ln(0.177 \times 331) = \ln(58.6) \approx 4.07 $$
This positive contribution counteracts the typical negative log-likelihood of all other points, inflating the overall Log-Likelihood from **$-442.07$** to **$-187.27$**.

---

## 📉 Impact on BIC (Bayesian Information Criterion)

The BIC formula is:
$$ \text{BIC} = -2 \ln(\widehat{L}) + k_p \ln(N) $$

Let's look at the trade-off between $K=2$ and $K=4$ under different regularization settings:

### Scenario A: Original Regularization (`reg_covar = 1e-6`)
* **$K=2$ ($k_p = 11$)**:
  $$ \text{BIC} = -2(-442.07) + 11 \ln(298) = 884.14 + 62.67 = 946.81 $$
* **$K=4$ ($k_p = 23$)**:
  $$ \text{BIC} = -2(-187.27) + 23 \ln(298) = 374.54 + 131.03 = 505.57 $$

Here, the increase in parameter penalty ($+68.36$) is completely overwhelmed by the massive drop in the likelihood term ($-509.6$). As a result, **BIC incorrectly flags $K=4$ as the optimal model.**

### Scenario B: High Regularization (`reg_covar = 1e-1`)
If we restrict the minimum covariance element from shrinking below $0.1$:
* **$K=2$ ($k_p = 11$)**:
  $$ \text{BIC} = -2(-494.56) + 11 \ln(298) = 989.12 + 62.67 = 1051.79 $$
* **$K=3$ ($k_p = 17$)**:
  $$ \text{BIC} = -2(-494.56) + 17 \ln(298) = 989.12 + 96.85 = 1085.97 $$
* **$K=4$ ($k_p = 23$)**:
  $$ \text{BIC} = -2(-494.56) + 23 \ln(298) = 989.12 + 131.03 = 1120.15 $$

By preventing the covariance from collapsing, the likelihood term is identical for all $K$, and the parameter penalty successfully selects **$K=2$ as the best model**.

---

## 💡 Key Takeaways
1. **Discretization is a GMM Hazard**: Datasets with heavy rounding/ties can cause GMMs to overfit "lines" of identical data values, leading to local singularities.
2. **BIC is Sensitive to Scale**: Because the log-likelihood term is $O(N)$ whereas the parameter penalty is $O(\ln N)$, any inflation in likelihood (such as component collapse) easily overrides the parameter penalty.
3. **Silhouette is More Robust**: Silhouette Score only uses cluster assignments and distances, making it immune to GMM singularity artifacts.
