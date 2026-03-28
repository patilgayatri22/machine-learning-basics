# 🎯 Datasets, Generalization & Overfitting

> **Crash Course** | Estimated Time: 50–65 minutes  
> **Prerequisites:** Basic ML concepts, Python, NumPy  
> **Series:** Machine Learning from Scratch

---

## 🎯 What You'll Learn

- How to properly split and manage datasets
- What generalization means and why it matters
- The bias-variance tradeoff
- How to detect, prevent, and fix overfitting
- Regularization techniques
- Cross-validation strategies

---

## 1. The Core Problem: Learning vs. Memorizing

Imagine a student preparing for an exam. Two approaches:

1. **Understanding the concepts** → can answer any question on the topic ✅
2. **Memorizing the exact practice questions** → fails on new questions ❌

Machine learning models face the same dilemma. A model that just memorizes training data is **useless in the real world**.

> **Generalization** = the model's ability to perform well on **new, unseen data**.

---

## 2. Understanding Your Dataset

### 2.1 The Three Splits

| Split | Size (typical) | Purpose |
|-------|---------------|---------|
| **Training set** | 60–80% | Model learns from this |
| **Validation set** | 10–20% | Tune hyperparameters, catch overfitting |
| **Test set** | 10–20% | Final unbiased evaluation — **touch only once** |

```python
from sklearn.model_selection import train_test_split

# First split: separate out the test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: training vs validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 × 0.8 = 0.2
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

### 2.2 Stratified Splitting (for classification)

When classes are imbalanced, a random split might give the test set very few examples of a rare class.

```python
# Ensure class proportions are maintained
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,       # ← preserves class ratios
    random_state=42
)

# Verify
import pandas as pd
print("Train class distribution:\n", pd.Series(y_train).value_counts(normalize=True))
print("Test class distribution:\n", pd.Series(y_test).value_counts(normalize=True))
```

### 2.3 Data Leakage — The Silent Killer ☠️

Data leakage happens when **information from outside the training set** influences the model. This causes unrealistically good validation scores that collapse in production.

**Common sources of leakage:**

```python
# ❌ WRONG: Fit scaler on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # test data influenced training stats!
X_train, X_test = train_test_split(X_scaled, ...)

# ✅ CORRECT: Fit scaler only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # learn stats from train only
X_test = scaler.transform(X_test)           # apply same transform to test
```

> 💡 **Pipelines automatically prevent this.** Always use `sklearn.pipeline.Pipeline`.

---

## 3. Overfitting, Underfitting & the Sweet Spot

### 3.1 The Three Regimes

```
                    Model Complexity →
         Low                               High
          |                                  |
 Training |  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ → keeps improving
    Error |
          |
Validation|  \          ___________~~~~~ → gets worse (overfitting)
    Error |   \________/
          |
          |  Underfitting | Just Right | Overfitting
```

| State | Training Error | Validation Error | Diagnosis |
|-------|---------------|-----------------|-----------|
| **Underfitting** | High | High | Model too simple |
| **Just right** | Low | Low | ✅ |
| **Overfitting** | Very low | High | Model too complex |

### 3.2 Visualizing with Learning Curves

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=10)

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_mean, label="Training score", color="blue")
plt.plot(train_sizes, val_mean, label="Validation score", color="orange")
plt.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color="blue")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curves")
plt.legend()
plt.grid(True)
plt.show()
```

**Interpreting learning curves:**
- **Large gap** between train and val → overfitting → add data, regularize, simplify
- **Both curves high & converging** → underfitting → more complex model, more features
- **Both curves converging at good score** → healthy generalization ✅

---

## 4. The Bias-Variance Tradeoff

The **total error** of any model can be decomposed into:

```
Total Error = Bias² + Variance + Irreducible Noise
```

| Component | Meaning | Caused by |
|-----------|---------|-----------|
| **Bias** | How wrong the model is on average | Oversimplified model |
| **Variance** | How much the model fluctuates across datasets | Overly complex model |
| **Irreducible noise** | Inherent randomness in data | Nothing you can do |

```python
# High Bias Example: Underfitting
from sklearn.linear_model import LinearRegression
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
print("Train score:", model_simple.score(X_train, y_train))   # e.g., 0.55
print("Val score:", model_simple.score(X_val, y_val))         # e.g., 0.54 — low & similar

# High Variance Example: Overfitting
from sklearn.tree import DecisionTreeClassifier
model_complex = DecisionTreeClassifier(max_depth=None)  # unlimited depth
model_complex.fit(X_train, y_train)
print("Train score:", model_complex.score(X_train, y_train))  # e.g., 1.00
print("Val score:", model_complex.score(X_val, y_val))        # e.g., 0.72 — big gap!
```

---

## 5. Fixing Overfitting

### 5.1 Get More Training Data

The most effective fix. More data makes it harder to memorize patterns.

```python
# Simulate effect of more data
from sklearn.model_selection import learning_curve
# See section 3.2 — if val score improves as training size grows → get more data
```

### 5.2 Regularization

Regularization **penalizes model complexity**, forcing the model to stay simple.

#### L1 Regularization (Lasso) — drives some weights to zero (feature selection)

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)   # alpha controls penalty strength
model.fit(X_train, y_train)
print("Features selected:", np.sum(model.coef_ != 0), "of", X_train.shape[1])
```

#### L2 Regularization (Ridge) — shrinks all weights toward zero

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

#### ElasticNet — combines L1 and L2

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio: 0=Ridge, 1=Lasso
model.fit(X_train, y_train)
```

> 💡 **Choosing alpha:** Use cross-validation to find the best value (see Section 6).

### 5.3 Dropout (Neural Networks)

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),      # randomly zero 30% of neurons during training
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

### 5.4 Early Stopping (Neural Networks / Gradient Boosting)

```python
# Keras
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,          # stop if no improvement for 5 epochs
    restore_best_weights=True
)

model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=200, callbacks=[callback])

# XGBoost
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=10)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("Best iteration:", xgb_model.best_iteration)
```

### 5.5 Reduce Model Complexity

```python
# Decision Tree: limit depth
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)

# Random Forest: fewer/shallower trees
model = RandomForestClassifier(n_estimators=100, max_depth=8, max_features="sqrt")
```

### 5.6 Data Augmentation (when more data isn't available)

```python
# Image augmentation (Keras)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)
```

---

## 6. Cross-Validation

Instead of a single train/val split, **K-Fold Cross-Validation** trains and evaluates K times:

```python
from sklearn.model_selection import cross_val_score, KFold

model = Ridge(alpha=1.0)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

### 6.1 Finding the Best Regularization Strength

```python
from sklearn.linear_model import RidgeCV

# Automatically tries multiple alpha values with cross-validation
model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
model.fit(X_train, y_train)
print("Best alpha:", model.alpha_)
```

### 6.2 Stratified K-Fold (for classification)

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
print(f"F1 Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 7. Complete Overfitting Detection Workflow

```python
def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)

    gap = train_score - val_score

    print(f"Train: {train_score:.4f}")
    print(f"Val:   {val_score:.4f}")
    print(f"Test:  {test_score:.4f}")
    print(f"Gap:   {gap:.4f}")

    if gap > 0.1:
        print("⚠️  Likely OVERFITTING — reduce complexity or regularize")
    elif val_score < 0.6:
        print("⚠️  Likely UNDERFITTING — increase model complexity")
    else:
        print("✅  Model looks healthy")
```

---

## 8. Summary Table

| Problem | Symptom | Fix |
|---------|---------|-----|
| Overfitting | Train ↑↑, Val ↓ | More data, regularization, simpler model, dropout, early stopping |
| Underfitting | Train ↓, Val ↓ | More features, more complex model, less regularization |
| Data leakage | Val ↑↑ (suspiciously good) | Fit preprocessors only on training data; use Pipelines |
| High variance | Scores vary a lot across CV folds | More data, ensemble methods |

---

## 9. 🧪 Practice Exercise

**Dataset:** [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

**Tasks:**
1. Split into train/val/test (60/20/20)
2. Train a `DecisionTreeRegressor` with no constraints — observe overfitting
3. Plot learning curves
4. Apply Ridge and Lasso regularization — compare results
5. Use 5-fold CV to find the best `alpha` for Ridge
6. Report final test score — explain your findings

---

## Key Takeaways

- ✅ Always split data **before** any preprocessing
- ✅ The **test set is sacred** — touch it only once at the very end
- ✅ A model that memorizes training data is **not intelligent**
- ✅ Use **learning curves** to diagnose bias vs variance
- ✅ **Regularization** is your primary tool against overfitting
- ✅ **Cross-validation** gives more reliable estimates than a single split

---

## 📚 Further Reading

- [Google's Machine Learning Crash Course — Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting)
- [Understanding the Bias-Variance Tradeoff — Scott Fortmann-Roe](http://scott.fortmann-roe.com/docs/BiasVariance.html)
- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/model_selection.html)

---

*← Previous: [01 - Numerical & Categorical Data](./01_ML_Data_Numerical_Categorical.md)*  
*Next → [03 - Advanced ML Models](./03_Advanced_ML_Models.md)*
