# 🚀 Advanced ML Models

> **Crash Course** | Estimated Time: 90–120 minutes  
> **Prerequisites:** Supervised learning basics, scikit-learn, numpy  
> **Series:** Machine Learning from Scratch

---

## 🎯 What You'll Learn

- Ensemble methods: Bagging, Boosting, Stacking
- Random Forests in depth
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Support Vector Machines
- Neural Networks fundamentals
- How to choose the right model

---

## 1. The Limits of Single Models

A single decision tree is easy to understand but fragile. A single linear model is stable but limited. **Advanced models** overcome these limitations through:

- **Ensembles**: combining many models to reduce error
- **Kernel methods**: projecting data into higher dimensions
- **Neural networks**: learning hierarchical representations

---

## 2. Ensemble Methods

### 2.1 Bagging (Bootstrap Aggregating)

Train **multiple independent models** on random subsets of data, then average (regression) or vote (classification).

```
Original Data → [Bootstrap Sample 1] → [Model 1] ─┐
              → [Bootstrap Sample 2] → [Model 2] ──┼→ [Aggregate] → Prediction
              → [Bootstrap Sample 3] → [Model 3] ─┘
```

**Effect:** Reduces **variance** — less overfitting than a single model.

### 2.2 Random Forest — Bagging Done Right

Random Forest adds one twist to bagging: at each split, only a **random subset of features** is considered. This decorrelates the trees.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=None,         # trees grow fully — bagging prevents overfitting
    max_features="sqrt",    # √(n_features) features per split
    min_samples_leaf=1,
    n_jobs=-1,              # use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
```

#### Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(rf.feature_importances_,
                        index=[f"Feature {i}" for i in range(20)])
importances.sort_values().tail(10).plot(kind="barh")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()
```

#### Key Hyperparameters for Random Forest

| Parameter | Effect | Tuning tip |
|-----------|--------|-----------|
| `n_estimators` | More trees = better + slower | Start 100, increase until improvement stops |
| `max_depth` | Controls tree depth | None (full) is usually fine with enough trees |
| `max_features` | Features per split | `sqrt` for classification, `0.33` for regression |
| `min_samples_leaf` | Minimum leaf size | Increase to reduce overfitting |

---

## 3. Gradient Boosting

Instead of training models independently (bagging), **boosting trains models sequentially**, each correcting the errors of the previous.

```
Model 1 → Residuals → Model 2 (fit on residuals) → Residuals → Model 3 → ...
Final prediction = weighted sum of all models
```

**Effect:** Reduces both **bias and variance** — often achieves best performance on tabular data.

### 3.1 XGBoost — The Competition Winner

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,      # also called eta — shrinks each tree's contribution
    subsample=0.8,           # fraction of samples per tree (like bagging)
    colsample_bytree=0.8,    # fraction of features per tree
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    early_stopping_rounds=20,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

preds = xgb_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"Best iteration: {xgb_model.best_iteration}")
```

### 3.2 LightGBM — Faster, Better for Large Data

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    num_leaves=31,           # controls complexity (default=31)
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
)
```

**Why LightGBM is faster:**
- Grows trees **leaf-wise** instead of level-wise
- Uses **histogram-based** feature binning
- Native support for categorical features

### 3.3 CatBoost — Best for Categorical Features

```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    cat_features=["city", "style"],   # no encoding needed!
    eval_metric="Accuracy",
    random_seed=42,
    verbose=50
)

cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
```

### 3.4 Gradient Boosting Comparison

| Model | Speed | Categorical | Memory | Best Use Case |
|-------|-------|-------------|--------|---------------|
| XGBoost | Fast | Manual encoding | Moderate | General tabular |
| LightGBM | Fastest | Good support | Low | Large datasets |
| CatBoost | Moderate | Native support | Moderate | Many categoricals |
| sklearn GBM | Slow | Manual encoding | Low | Small datasets |

---

## 4. Stacking (Stacked Generalization)

Train several **diverse base models**, then train a **meta-model** that learns how to combine their predictions.

```
                  ┌─ Random Forest ─┐
Training Data ────┼─ XGBoost       ─┼─→ [Meta-model (LogReg)] → Final Prediction
                  └─ SVM           ─┘
```

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

base_models = [
    ("rf", RandomForestClassifier(n_estimators=100)),
    ("xgb", xgb.XGBClassifier(n_estimators=100, verbosity=0)),
    ("svc", SVC(probability=True))
]

stacked = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5                   # base models trained with cross-val to avoid leakage
)

stacked.fit(X_train, y_train)
print(f"Stacked Accuracy: {stacked.score(X_test, y_test):.4f}")
```

---

## 5. Support Vector Machines (SVM)

SVMs find the **maximum-margin hyperplane** that separates classes. The **kernel trick** maps data into higher dimensions where linear separation becomes possible.

```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# IMPORTANT: SVMs need scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification
svm_clf = SVC(
    kernel="rbf",     # radial basis function — best for nonlinear data
    C=10.0,           # regularization: larger C → less regularization
    gamma="scale",    # controls influence radius of each training point
    probability=True  # enables predict_proba
)
svm_clf.fit(X_train_scaled, y_train)
print(f"SVM Accuracy: {svm_clf.score(X_test_scaled, y_test):.4f}")
```

#### Kernel Types

| Kernel | Formula | Use Case |
|--------|---------|---------|
| `linear` | x · x' | Linearly separable data, high-dimensional (NLP) |
| `rbf` | exp(-γ‖x-x'‖²) | General purpose, nonlinear data |
| `poly` | (γx·x' + r)^d | Polynomial relationships |
| `sigmoid` | tanh(γx·x' + r) | Neural network-like behavior |

#### SVM Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01],
    "kernel": ["rbf", "poly"]
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
```

---

## 6. Neural Networks Fundamentals

### 6.1 Architecture

```
Input Layer    Hidden Layers       Output Layer
[x1]  ─────→  [Neuron] ─┐
[x2]  ─────→  [Neuron]  ├─→ [Neuron] ─→ [y_hat]
[x3]  ─────→  [Neuron] ─┘
```

Each neuron: `output = activation(weights · inputs + bias)`

### 6.2 Building with Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, regularizers

# Functional API (more flexible)
inputs = tf.keras.Input(shape=(X_train.shape[1],))
x = layers.Dense(256, activation="relu",
                 kernel_regularizer=regularizers.l2(0.01))(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
```

### 6.3 Training with Callbacks

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
```

### 6.4 Activation Functions

| Function | Formula | Use Case |
|----------|---------|---------|
| ReLU | max(0, x) | Hidden layers (default choice) |
| LeakyReLU | max(αx, x) | When ReLU neurons die |
| Sigmoid | 1/(1+e^-x) | Binary output layer |
| Softmax | e^xi / Σe^xj | Multi-class output layer |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | RNNs, zero-centered |

---

## 7. Choosing the Right Model

```
Is your data tabular?
    ├── YES → Is your dataset large (>100K rows)?
    │             ├── YES → LightGBM or XGBoost
    │             └── NO  → Try Random Forest first, then XGBoost
    └── NO  → Is it images/audio/video?
                  ├── YES → Convolutional Neural Network (CNN)
                  └── NO  → Is it text/sequences?
                                ├── YES → Transformer / LSTM / BERT
                                └── NO  → Depends on structure
```

### Quick Decision Table

| Algorithm | Tabular | Interpretable | Scales | Handles Categoricals | Train Speed |
|-----------|---------|---------------|--------|---------------------|------------|
| Linear/Logistic Reg | ✅ | ✅✅ | ✅ | Manual | Fast |
| Decision Tree | ✅ | ✅✅ | ❌ | Manual | Fast |
| Random Forest | ✅ | ⚠️ | ✅ | Manual | Medium |
| XGBoost | ✅ | ⚠️ | ✅ | Manual | Fast |
| LightGBM | ✅ | ⚠️ | ✅✅ | Good | Very Fast |
| CatBoost | ✅ | ⚠️ | ✅ | Native | Medium |
| SVM | ✅ | ❌ | ❌ | Manual | Slow |
| Neural Network | ✅/✅✅ | ❌ | ✅✅ | Embedding | Slow |

---

## 8. Hyperparameter Tuning at Scale

```python
# Randomized Search — much faster than Grid Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 12),
    "learning_rate": uniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(verbosity=0),
    param_distributions,
    n_iter=50,         # try 50 random combinations
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Best params:", random_search.best_params_)
```

---

## 9. 🧪 Practice Exercise

**Dataset:** [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

**Tasks:**
1. Train a Random Forest, XGBoost, and SVM
2. Compare their accuracy and ROC-AUC scores on the test set
3. Build a Stacking ensemble using all three as base models
4. Use RandomizedSearchCV to tune the best-performing model
5. Plot feature importances for the best tree-based model
6. Does the ensemble beat the best individual model?

---

## Key Takeaways

- ✅ **Random Forest** = great starting point for tabular data, robust and fast
- ✅ **Gradient Boosting** (XGBoost/LightGBM) = usually best on tabular data
- ✅ **Stacking** = squeezes extra performance from diverse models
- ✅ **SVM** = powerful but needs scaling and careful tuning
- ✅ **Neural Networks** = best for unstructured data (images, text, audio)
- ✅ When in doubt: **start simple, then add complexity**

---

## 📚 Further Reading

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow — Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [StatQuest with Josh Starmer (YouTube)](https://www.youtube.com/c/joshstarmer)

---

*← Previous: [02 - Generalization & Overfitting](./02_Datasets_Generalization_Overfitting.md)*  
*Next → [04 - Production ML Systems](./04_Real_World_ML_Production.md)*
