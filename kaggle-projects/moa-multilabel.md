# Mechanisms of Action (MoA) 

The MoA competition was one of the most complex multi-label challenges I’ve worked on.
The dataset includes **hundreds of biological features**, **206 target mechanisms**, heavy **class imbalance**, and strong **drug-dependent correlations** — making validation, modeling, and ensembling the key components of the solution.

This project taught me a lot about high-dimensional tabular ML, leakage-safe validation, and multi-model stacking.

---

## 1. Data Understanding

### 1.1 Feature Groups

* **g-features (gene expression):** large correlated clusters
* **c-features (cell viability):** moderate correlations
* Both follow near-normal distributions → scaling required.

### 1.2 Experimental Metadata

* `cp_time`: {24h, 48h, 72h} (48h dominates)
* `cp_dose`: two discrete dose levels
* `cp_type`: control samples (`ctl_vehicle`) contain only zero-targets → removed.

### 1.3 Target Imbalance

* Frequent MoA labels appear thousands of times.
* Rare ones appear <20 times.
* Macro metrics become highly unstable.

### 1.4 Drug Dependency (Important)

Samples from the same `drug_id` share underlying target structure → must **never** mix train/val across drugs.

---

## 2. Feature Pipeline

### 2.1 Remove Controls

Control samples add noise and always have 0 targets → excluded.

### 2.2 Scaling

Used either:

* **QuantileTransformer**
* or **StandardScaler**

Both improved stability of neural networks.

### 2.3 Encode Metadata

One-hot encoding for `cp_time` and `cp_dose`.

### 2.4 PCA Reduction

Strong correlations justified PCA:

* **600 PCs** for g-features
* **60 PCs** for c-features

Final input dimension: **662 features**.

This drastically reduced noise and improved convergence.

---

## 3. Validation Strategy (Core of the Solution)

The dataset is vulnerable to leakage from two sources:

1. **Same drug in train and validation** → artificially high scores.
2. **Rare labels unevenly split across folds** → unstable macro/F1.

To fix this:

### ✔ `GroupKFold(by drug_id)`

Keeps each drug strictly in one fold.

### ✔ Combined with `MultilabelStratifiedKFold`

Balances rare labels inside each drug-group split.

This two-level scheme produced realistic and stable CV.

---

## 4. Models

## 4.1 MLP with FeatureGate (Primary Model)

### Architecture

* Dense 1024 → BatchNorm → FeatureGate → ReLU → Dropout
* Dense 512  → BatchNorm → FeatureGate → ReLU → Dropout
* Dense 256  → BatchNorm → ReLU → Dropout
* Output: 206 sigmoids

### Training

* Loss: **BCEWithLogitsLoss**
* Optimizer: **Adam**
* Batch size: 256
* Epochs: 50
* Weight decay: 1e-5

### Results

| Split   | LogLoss     |
| ------- | ----------- |
| Private | **0.01704** |
| Public  | **0.01947** |

Very small gap → excellent generalization.

---

## 4.2 XGBoost

```
n_estimators=1000
max_depth=8
learning_rate=0.01
subsample=0.8
colsample_bytree=0.8
tree_method='gpu_hist'
eval_metric='logloss'
```

| Split   | LogLoss |
| ------- | ------- |
| Private | 0.0208  |
| Public  | 0.0221  |

Strong on mid-frequency labels, weaker on rare ones.

---

# 5. Ensembling & Stacking Strategy

This solution used a **2-level ensemble**, not a simple weighted average.

---

## 5.1 Level-1: Base OOF Predictions

For each fold, I saved OOF predictions from:

* MLP
* XGBoost

OOF predictions ensured leak-free stacking.

---

## 5.2 Per-Class α-Blending

Instead of one global α, I optimized **α per target**:

```
final_prob[class] = α[class] * mlp[class] + (1 − α[class]) * xgb[class]
```

This allowed:

* MLP → dominate common labels
* XGB → stabilize rare labels

Blending reduced fold variance and smoothed noise.

---

## 5.3 Level-2 Meta-Model (Stacking)

The concatenated OOF vectors
`[OOF_MLP, OOF_XGB]`
were fed into **206 logistic regressions** (one per MoA target).

Why it works:

* MLP learns complex non-linear patterns
* tree models catch interactions & splits
* meta-model learns optimal weighting per class

Stacking gave strong and stable gains.

---

## 5.4 Threshold Optimization (Per-Label)

Since the competition metric uses binary predictions, I applied:

```
threshold ∈ [0.05 ... 0.95]
```

optimized per class using coordinate descent on OOF.
This improved macro-F1 and especially rare labels.

---

# 6. Final Results

| Model                   | Private     | Public      |
| ----------------------- | ----------- | ----------- |
| MLP                     | 0.01704     | 0.01947     |
| XGBoost                 | 0.0208      | 0.0221      |
| **Blend (α per class)** | **0.01841** | **0.02060** |
| **Final Stacking**      | **0.01713** | **0.01930** |

---

# 7. Lessons Learned

* Proper **validation** matters more than architecture depth.
* Rare labels → require sampling, per-class thresholds, and blending.
* PCA helped reduce noise in a high-dimensional biological setting.
* OOF predictions are the foundation of all reliable stacking.
* Multi-model ensembles consistently outperform any single learner.
