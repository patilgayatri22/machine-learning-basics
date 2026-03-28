# 🏭 Real-World ML: Production ML Systems

> **Crash Course** | Estimated Time: 75–90 minutes  
> **Prerequisites:** ML fundamentals, Python, basic software engineering  
> **Series:** Machine Learning from Scratch

---

## 🎯 What You'll Learn

- The full ML production lifecycle
- How to build ML pipelines and APIs
- Model versioning and experiment tracking
- Monitoring and detecting model drift
- Serving models at scale
- MLOps fundamentals

---

## 1. Research vs. Production: A Critical Gap

Most ML tutorials end at `model.fit()`. Production ML is an entirely different discipline.

| Aspect | Research/Notebook | Production |
|--------|-------------------|-----------|
| Data | Static CSV files | Streaming, changing data |
| Scale | Hundreds of rows | Millions of requests/day |
| Error handling | Crashes are fine | Must be fault-tolerant |
| Versioning | Git + notebooks | Models, data, code all versioned |
| Monitoring | None | Drift detection, alerting |
| Iteration | Weeks | Continuous deployment |

> 💡 **Industry stat:** Only ~22% of ML projects ever make it to production. Most fail not due to modeling, but due to engineering and operational challenges.

---

## 2. The ML Production Lifecycle

```
[Business Problem]
      ↓
[Data Collection & Validation]
      ↓
[Feature Engineering & Pipelines]
      ↓
[Model Training & Experimentation]
      ↓
[Model Evaluation & Validation]
      ↓
[Model Packaging & Versioning]
      ↓
[Deployment (API / Batch / Streaming)]
      ↓
[Monitoring & Observability]
      ↓
[Retraining & Continuous Improvement]
      ↑_________________________________|
```

---

## 3. Experiment Tracking with MLflow

Never lose track of what you trained, with what parameters, and what results.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_experiment("housing-price-classifier")

with mlflow.start_run(run_name="rf-baseline"):

    # Log parameters
    params = {"n_estimators": 200, "max_depth": 8, "random_state": 42}
    mlflow.log_params(params)

    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))

    # Log the model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Run complete. View at: mlflow ui")
```

```bash
# Launch the MLflow UI
mlflow ui
# Visit http://127.0.0.1:5000
```

---

## 4. Building Reproducible ML Pipelines

### 4.1 The Problem with Ad-Hoc Preprocessing

```python
# ❌ This is not reproducible in production
df["age_scaled"] = (df["age"] - 35.2) / 12.1   # hardcoded stats!
```

### 4.2 Production-Ready Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

numerical_features = ["age", "income", "credit_score"]
categorical_features = ["occupation", "city"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200))
])

# Train
full_pipeline.fit(X_train, y_train)

# Save the ENTIRE pipeline (preprocessing + model)
joblib.dump(full_pipeline, "model_pipeline_v1.pkl")
print("Pipeline saved!")

# Load and predict in production
pipeline = joblib.load("model_pipeline_v1.pkl")
predictions = pipeline.predict(new_data)   # new_data: raw features, no preprocessing needed
```

---

## 5. Serving Models as REST APIs

### 5.1 FastAPI Model Serving

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="ML Model API", version="1.0")

# Load pipeline at startup
pipeline = joblib.load("model_pipeline_v1.pkl")

class PredictionRequest(BaseModel):
    age: float
    income: float
    credit_score: float
    occupation: str
    city: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert to DataFrame (pipeline expects this format)
        input_data = pd.DataFrame([request.dict()])

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0].max()

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}
```

```bash
# Run the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Test it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 75000, "credit_score": 720, "occupation": "engineer", "city": "San Francisco"}'
```

### 5.2 Containerizing with Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_pipeline_v1.pkl .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ml-api:v1 .
docker run -p 8000:8000 ml-api:v1
```

---

## 6. Data Validation

Never trust incoming data. Validate it before inference.

```python
# Using Great Expectations
import great_expectations as ge
import pandas as pd

df = ge.from_pandas(pd.read_csv("training_data.csv"))

# Define expectations
df.expect_column_values_to_not_be_null("age")
df.expect_column_values_to_be_between("age", min_value=0, max_value=120)
df.expect_column_values_to_be_between("income", min_value=0, max_value=10_000_000)
df.expect_column_values_to_be_in_set("occupation", ["engineer", "doctor", "teacher", "other"])

# Run validation
results = df.validate()
print(f"Validation passed: {results['success']}")

if not results["success"]:
    failed = [r for r in results["results"] if not r["success"]]
    for f in failed:
        print(f"FAILED: {f['expectation_config']['expectation_type']}")
```

---

## 7. Model Monitoring & Drift Detection

Models degrade over time as the real world changes. You must monitor them.

### 7.1 Types of Drift

| Type | Definition | Example |
|------|-----------|---------|
| **Data drift** | Input distribution changes | Average age of users changes |
| **Label drift** | Target distribution changes | Fraud rate suddenly spikes |
| **Concept drift** | Relationship between X and y changes | Post-COVID behavior patterns |

### 7.2 Detecting Data Drift

```python
from scipy import stats
import numpy as np

def detect_drift(reference_data: np.array, current_data: np.array,
                 threshold: float = 0.05) -> dict:
    """
    Kolmogorov-Smirnov test for distribution drift.
    p-value < threshold indicates drift.
    """
    ks_stat, p_value = stats.ks_2samp(reference_data, current_data)

    return {
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "drift_detected": p_value < threshold
    }

# Example usage
reference_ages = X_train["age"].values         # baseline distribution
current_ages = new_data["age"].values          # today's incoming data

result = detect_drift(reference_ages, current_ages)
print(result)
# {'ks_statistic': 0.15, 'p_value': 0.003, 'drift_detected': True}

if result["drift_detected"]:
    print("⚠️ Data drift detected in 'age' feature — consider retraining!")
```

### 7.3 Monitoring Prediction Quality

```python
import pandas as pd
from datetime import datetime

class ModelMonitor:
    def __init__(self):
        self.predictions_log = []

    def log_prediction(self, features, prediction, probability, actual=None):
        self.predictions_log.append({
            "timestamp": datetime.now(),
            "prediction": prediction,
            "probability": probability,
            "actual": actual  # filled in later when ground truth arrives
        })

    def get_metrics_window(self, hours=24):
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        recent = [p for p in self.predictions_log if p["timestamp"] > cutoff]

        labeled = [p for p in recent if p["actual"] is not None]
        if not labeled:
            return {"warning": "No labeled data in window"}

        from sklearn.metrics import accuracy_score
        y_true = [p["actual"] for p in labeled]
        y_pred = [p["prediction"] for p in labeled]

        return {
            "window_hours": hours,
            "total_predictions": len(recent),
            "labeled_predictions": len(labeled),
            "accuracy": accuracy_score(y_true, y_pred),
            "avg_confidence": sum(p["probability"] for p in recent) / len(recent)
        }

monitor = ModelMonitor()
```

---

## 8. Batch vs. Online vs. Streaming Inference

| Mode | Description | Latency | Use Case |
|------|-------------|---------|---------|
| **Batch** | Score large datasets periodically | Hours | Daily recommendations, risk scoring |
| **Online (REST)** | Score one record at a time | <100ms | Real-time fraud detection, search ranking |
| **Streaming** | Score records as they arrive in a stream | Seconds | Event-driven applications |

```python
# Batch inference example
def batch_predict(model_path: str, input_csv: str, output_csv: str):
    import joblib
    import pandas as pd

    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)

    df["prediction"] = model.predict(df)
    df["probability"] = model.predict_proba(df).max(axis=1)
    df["scored_at"] = pd.Timestamp.now()

    df.to_csv(output_csv, index=False)
    print(f"Scored {len(df)} records → {output_csv}")

batch_predict("model_pipeline_v1.pkl", "new_customers.csv", "scored_customers.csv")
```

---

## 9. Model Versioning & Registry

```python
# Using MLflow Model Registry
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
result = mlflow.register_model(
    model_uri="runs:/your-run-id/model",
    name="fraud-detector"
)
print(f"Model version: {result.version}")

# Transition to production
client.transition_model_version_stage(
    name="fraud-detector",
    version=result.version,
    stage="Production",
    archive_existing_versions=True
)

# Load production model anywhere
model = mlflow.pyfunc.load_model("models:/fraud-detector/Production")
```

---

## 10. Retraining Strategy

```python
# Automated retraining trigger
def should_retrain(current_accuracy: float, baseline_accuracy: float,
                   drift_detected: bool, days_since_training: int) -> bool:
    """
    Retrain when:
    - Accuracy drops > 5% from baseline
    - Data drift is detected
    - Model is older than 30 days
    """
    accuracy_dropped = current_accuracy < (baseline_accuracy - 0.05)
    stale = days_since_training > 30

    reasons = []
    if accuracy_dropped:
        reasons.append(f"Accuracy dropped: {current_accuracy:.3f} vs baseline {baseline_accuracy:.3f}")
    if drift_detected:
        reasons.append("Data drift detected")
    if stale:
        reasons.append(f"Model is {days_since_training} days old")

    if reasons:
        print("🔄 Retraining triggered:", "; ".join(reasons))
        return True
    return False
```

---

## 11. 🧪 Practice Exercise

**Project:** Build a mini end-to-end ML system for predicting customer churn

**Tasks:**
1. Train a model and log it with MLflow (parameters + metrics + artifact)
2. Wrap it in a FastAPI endpoint with input validation via Pydantic
3. Add a `/health` and `/metrics` endpoint
4. Write a drift detection function that alerts when mean feature values shift >10%
5. Create a Dockerfile for the API
6. Bonus: Add a retraining script that runs automatically if accuracy drops

---

## Key Takeaways

- ✅ Production ML is **80% engineering, 20% modeling**
- ✅ Always save the **full pipeline** (preprocessing + model), never just the model
- ✅ **Track every experiment** — you'll forget what worked
- ✅ **Monitor everything**: data, predictions, performance
- ✅ Have a **retraining strategy** before you deploy
- ✅ Use **Docker** to ensure reproducibility across environments

---

## 📚 Further Reading

- [Designing Machine Learning Systems — Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Made With ML — MLOps Course](https://madewithml.com/)
- [Google's ML Engineering for Production Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)

---

*← Previous: [03 - Advanced ML Models](./03_Advanced_ML_Models.md)*  
*Next → [05 - AutoML](./05_AutoML.md)*
