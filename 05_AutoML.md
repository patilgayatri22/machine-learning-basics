# 🤖 AutoML: Automated Machine Learning

> **Crash Course** | Estimated Time: 60–75 minutes  
> **Prerequisites:** ML fundamentals, Python, scikit-learn basics  
> **Series:** Machine Learning from Scratch

---

## 🎯 What You'll Learn

- What AutoML is and when to use it
- The components AutoML automates
- Major AutoML frameworks and tools
- How to use AutoML in practice (AutoSklearn, H2O, FLAML, AutoGluon)
- Limitations and pitfalls of AutoML
- When to go manual vs. automated

---

## 1. What Is AutoML?

**AutoML (Automated Machine Learning)** automates the end-to-end process of applying machine learning to real-world problems. Instead of manually:

- Choosing which algorithms to try
- Engineering features
- Tuning hyperparameters
- Selecting the best model

...you let the AutoML system do it for you.

```
Manual ML:   [You] → Feature Engineering → Algorithm Selection → Hyperparameter Tuning → [Model]

AutoML:      [You] → [AutoML System does everything] → [Optimized Model]
```

> **Who benefits?**
> - Domain experts without deep ML knowledge
> - ML engineers who want faster baselines
> - Teams with limited time for experimentation
> - Anyone solving a well-defined tabular prediction task

---

## 2. What AutoML Automates

| Component | What It Does | Example |
|-----------|-------------|---------|
| **Data preprocessing** | Imputation, scaling, encoding | Auto-detects categorical columns |
| **Feature engineering** | Creates new features automatically | Polynomial features, date decomposition |
| **Algorithm selection** | Tries multiple models | RF, XGBoost, SVM, Linear, etc. |
| **Hyperparameter optimization** | Finds best settings | Bayesian optimization, random search |
| **Ensemble construction** | Combines best models | Stacking, voting |
| **Model evaluation** | Cross-validates and selects best | Optimizes specified metric |

---

## 3. The AutoML Landscape

| Tool | Best For | Open Source | Notes |
|------|----------|-------------|-------|
| **Auto-sklearn** | Tabular, research | ✅ | sklearn-compatible, powerful |
| **H2O AutoML** | Large datasets, enterprise | ✅ | Fast, distributed |
| **FLAML** | Speed, resource-aware | ✅ | Microsoft, very fast |
| **AutoGluon** | Tabular + text + images | ✅ | Amazon, state-of-the-art |
| **TPOT** | Genetic programming | ✅ | Exports clean sklearn code |
| **Google AutoML** | Cloud-native | ❌ | No-code, production-ready |
| **Azure AutoML** | Azure ecosystem | ❌ | Enterprise MLOps integration |
| **DataRobot** | Enterprise | ❌ | No-code platform |

---

## 4. AutoML in Practice

### 4.1 FLAML — Fast and Resource-Aware

FLAML is an excellent starting point: fast, lightweight, and easy to use.

```python
from flaml import AutoML
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AutoML
automl = AutoML()

automl.fit(
    X_train, y_train,
    task="classification",    # "classification", "regression", "ts_forecast"
    time_budget=60,           # seconds — total search budget
    metric="accuracy",        # or "f1", "roc_auc", "r2", "rmse"
    n_jobs=-1,
    verbose=1
)

# Results
print("Best model:", automl.best_estimator)
print("Best config:", automl.best_config)
print("CV accuracy:", 1 - automl.best_loss)

# Predict
y_pred = automl.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### 4.2 AutoGluon — State-of-the-Art AutoML

AutoGluon often achieves **top results** with minimal code:

```python
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# Load data as pandas DataFrame
train_data = TabularDataset("train.csv")
test_data = TabularDataset("test.csv")

# Train — specify target column only
predictor = TabularPredictor(
    label="target",                  # target column name
    eval_metric="accuracy",
    path="autogluon-models/"        # where to save models
).fit(
    train_data,
    time_limit=300,                  # 5 minutes
    presets="best_quality"           # or "good_quality", "medium_quality" for speed
)

# Evaluate
performance = predictor.evaluate(test_data)
print(performance)

# Predict
predictions = predictor.predict(test_data.drop("target", axis=1))

# Leaderboard of all models tried
leaderboard = predictor.leaderboard(test_data, silent=True)
print(leaderboard[["model", "score_test", "score_val", "fit_time"]].head(10))
```

**AutoGluon presets:**

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|---------|
| `medium_quality` | Fastest | Good | Quick experiments |
| `good_quality` | Fast | Better | Balanced |
| `high_quality` | Slow | High | Competitions |
| `best_quality` | Slowest | Best | Production |

### 4.3 H2O AutoML — Great for Large Datasets

```python
import h2o
from h2o.automl import H2OAutoML

# Start H2O cluster
h2o.init()

# Load data
df = h2o.import_file("data.csv")

# Define features and target
target = "loan_status"
features = [c for c in df.columns if c != target]

# Split
train, valid, test = df.split_frame(ratios=[0.7, 0.15])

# Train
aml = H2OAutoML(
    max_runtime_secs=300,          # 5 minutes
    max_models=20,                 # try up to 20 models
    seed=42,
    sort_metric="AUC"
)

aml.train(x=features, y=target, training_frame=train, validation_frame=valid)

# Leaderboard
print(aml.leaderboard.head(10))

# Best model
best_model = aml.leader
perf = best_model.model_performance(test)
print("AUC:", perf.auc())

# Save best model
model_path = h2o.save_model(model=best_model, path="h2o_models/", force=True)
```

### 4.4 TPOT — Exports Clean Code

TPOT is unique: it generates actual scikit-learn pipeline code you can inspect and own.

```python
from tpot import TPOTClassifier

tpot = TPOTClassifier(
    generations=5,         # number of evolutionary generations
    population_size=20,    # number of pipelines per generation
    cv=5,
    random_state=42,
    verbosity=2,
    n_jobs=-1,
    max_time_mins=5
)

tpot.fit(X_train, y_train)
print(f"Test Accuracy: {tpot.score(X_test, y_test):.4f}")

# Export the best pipeline as Python code
tpot.export("best_pipeline.py")
```

```python
# Contents of best_pipeline.py (example output)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

exported_pipeline = make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=200,
        subsample=0.8
    )
)
```

---

## 5. Neural Architecture Search (NAS)

For deep learning, AutoML extends to automatically finding optimal neural network architectures.

```python
import keras_tuner as kt
import tensorflow as tf

def build_model(hp):
    model = tf.keras.Sequential()

    # Tune number of layers
    for i in range(hp.Int("num_layers", 1, 4)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
            activation=hp.Choice("activation", ["relu", "tanh", "swish"])
        ))
        model.add(tf.keras.layers.Dropout(
            hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
        ))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=30,
    directory="nas_results",
    project_name="binary_classification"
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val),
             epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

best_model = tuner.get_best_models(1)[0]
best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best architecture:", best_hp.values)
```

---

## 6. Hyperparameter Optimization Deep Dive

AutoML relies on smart search strategies — understanding them helps you apply them manually too.

### 6.1 Bayesian Optimization

Uses a probabilistic model to choose the next hyperparameter set based on prior results.

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

search_spaces = {
    "n_estimators": Integer(50, 500),
    "max_depth": Integer(3, 15),
    "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0)
}

import xgboost as xgb
opt = BayesSearchCV(
    xgb.XGBClassifier(verbosity=0),
    search_spaces,
    n_iter=40,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42
)

opt.fit(X_train, y_train)
print("Best params:", opt.best_params_)
print("Best CV score:", opt.best_score_)
```

### 6.2 Optuna — Modern HPO Framework

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best trial:", study.best_trial.params)
print("Best score:", study.best_trial.value)

# Visualization
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

---

## 7. AutoML Limitations & When to Go Manual

### Limitations

| Limitation | Description |
|-----------|-------------|
| **Black box** | Hard to understand what and why it chose |
| **Compute cost** | Can be slow and expensive |
| **Data quality** | Garbage in, garbage out — AutoML can't fix bad data |
| **Domain knowledge** | Can't apply domain-specific feature engineering |
| **Novel problems** | Struggles with unusual tasks (custom loss functions, multi-output) |
| **Deployment** | May produce large, complex pipelines hard to serve |

### When to Use AutoML

✅ Use AutoML when:
- You need a **fast baseline** to compare against
- The task is **standard** (binary classification, regression)
- Domain expertise is limited
- You're doing **rapid prototyping**
- The dataset is **well-structured tabular data**

❌ Skip AutoML when:
- You need **full interpretability** and control
- Data is **unstructured** (images, text) — use DL directly
- You have **strong domain knowledge** to apply manually
- The problem has **custom constraints** or a non-standard objective
- **Compute budget is very tight**

---

## 8. AutoML Evaluation Workflow

```python
from flaml import AutoML
from sklearn.metrics import classification_report
import pandas as pd

def run_automl_experiment(X_train, X_test, y_train, y_test,
                          time_budget=120, task="classification"):
    automl = AutoML()
    automl.fit(X_train, y_train, task=task,
               time_budget=time_budget, metric="roc_auc")

    # Get predictions
    y_pred = automl.predict(X_test)
    y_prob = automl.predict_proba(X_test)[:, 1]

    # Report
    print("=" * 50)
    print("AutoML Results")
    print("=" * 50)
    print(f"Best model: {automl.best_estimator}")
    print(f"Best config: {automl.best_config}")
    print()
    print(classification_report(y_test, y_pred))

    return automl, y_pred, y_prob

automl, y_pred, y_prob = run_automl_experiment(
    X_train, X_test, y_train, y_test, time_budget=120
)
```

---

## 9. 🧪 Practice Exercise

**Dataset:** [Kaggle Titanic](https://www.kaggle.com/c/titanic) or any Kaggle tabular competition

**Tasks:**
1. Run FLAML with a 60-second budget — note best model and score
2. Run AutoGluon with `medium_quality` preset — compare
3. Run TPOT and export the generated pipeline code — inspect it
4. Compare all AutoML results to a manually tuned XGBoost model
5. Which approach gives the best score? Which is fastest? Which is most explainable?

---

## Key Takeaways

- ✅ AutoML is a **time-saver, not a replacement** for ML knowledge
- ✅ Use it to **establish baselines quickly**, then improve manually
- ✅ **AutoGluon** and **FLAML** are the best open-source options today
- ✅ TPOT is unique for generating **interpretable, exportable code**
- ✅ Bayesian optimization and **Optuna** are powerful for manual HPO
- ✅ AutoML does not fix **bad data** or **bad problem framing**

---

## 📚 Further Reading

- [AutoML.org](https://www.automl.org/)
- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [FLAML GitHub](https://github.com/microsoft/FLAML)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [AutoML: Methods, Systems, Challenges (Book)](https://www.automl.org/book/)

---

*← Previous: [04 - Production ML Systems](./04_Real_World_ML_Production.md)*  
*Next → [06 - ML Fairness](./06_ML_Fairness.md)*
