# ⚖️ ML Fairness: Building Equitable Machine Learning Systems

> **Crash Course** | Estimated Time: 60–75 minutes  
> **Prerequisites:** ML fundamentals, basic statistics  
> **Series:** Machine Learning from Scratch

---

## 🎯 What You'll Learn

- Why ML systems can be unfair and the real-world consequences
- Types of bias in ML and their sources
- Mathematical definitions of fairness (and why they conflict)
- How to measure bias in your models
- Techniques to mitigate bias
- Fairness tools and frameworks
- Ethical responsibilities as an ML practitioner

---

## 1. Why ML Fairness Matters

Machine learning models are increasingly used in **high-stakes decisions**:

- 🏦 Loan approvals
- 👮 Criminal risk scoring (e.g., COMPAS recidivism)
- 🏥 Medical diagnosis and treatment recommendations
- 👔 Hiring and resume screening
- 🏠 Housing eligibility
- 👁️ Facial recognition and surveillance

When these systems are **biased**, they can cause real harm — denying opportunities, reinforcing discrimination, or misidentifying people based on protected characteristics like race, gender, or age.

> **Famous case:** The COMPAS recidivism algorithm was found to be nearly twice as likely to falsely flag Black defendants as future criminals compared to white defendants (ProPublica, 2016).

---

## 2. Types of Bias in ML

### 2.1 Sources of Bias

| Bias Type | Definition | Example |
|-----------|-----------|---------|
| **Historical bias** | Training data reflects past discrimination | Credit scores based on historically redlined neighborhoods |
| **Representation bias** | Certain groups underrepresented in data | Face recognition trained mostly on light-skinned faces |
| **Measurement bias** | Proxy variables that correlate with protected attributes | Using zip code as a proxy for race |
| **Aggregation bias** | One model for all groups, but groups differ | A diabetes model trained on males applied to females |
| **Evaluation bias** | Benchmarks not representative of all groups | NLP benchmarks lacking non-English dialects |
| **Deployment bias** | Model used differently than intended | Hiring tool used in different contexts than trained on |

### 2.2 The Feedback Loop Problem

```
Biased Training Data
        ↓
  Biased Model
        ↓
  Biased Decisions
        ↓
More Biased Data (future training)
        ↓
  More Biased Model  ← This is the loop
```

Example: A biased hiring algorithm rejects qualified candidates from underrepresented groups → fewer from those groups are hired → future training data shows them as "less qualified."

---

## 3. Mathematical Fairness Definitions

There is no single definition of fairness. Different stakeholders want different things — and they **mathematically conflict with each other**.

### 3.1 Key Definitions

Let:
- `Y` = true outcome (0/1)
- `Ŷ` = model prediction
- `A` = protected attribute (e.g., race, gender)

#### Demographic Parity (Statistical Parity)
The model predicts positive outcomes at the **same rate** across groups.

```
P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
```

```python
def demographic_parity(y_pred, sensitive_feature):
    groups = {}
    for group in sensitive_feature.unique():
        mask = sensitive_feature == group
        groups[group] = y_pred[mask].mean()
    return groups

result = demographic_parity(predictions, df["gender"])
print("Positive prediction rate:", result)
# {'M': 0.62, 'F': 0.41}  ← significant disparity
```

#### Equal Opportunity
Equal **true positive rates** (sensitivity/recall) across groups — same chance of being correctly identified as positive.

```
P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)
```

#### Equalized Odds
Both **true positive rates AND false positive rates** are equal across groups.

```
P(Ŷ=1 | Y=y, A=0) = P(Ŷ=1 | Y=y, A=1)  for y ∈ {0,1}
```

#### Predictive Parity (Calibration)
Equal **precision** across groups — if the model says "high risk", it should be right equally often for all groups.

```
P(Y=1 | Ŷ=1, A=0) = P(Y=1 | Ŷ=1, A=1)
```

### 3.2 The Impossibility Theorem

> **Chouldechova (2017) & Kleinberg (2016) proved:** It is mathematically impossible to simultaneously satisfy demographic parity, equal opportunity, and predictive parity when base rates differ between groups.

This means **every fairness intervention involves a tradeoff**. You must choose which definition of fairness matters most for your specific context.

---

## 4. Measuring Bias with Python

### 4.1 Using Fairlearn

```python
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    equalized_odds_difference
)
from sklearn.metrics import accuracy_score, precision_score

# Assume: y_true, y_pred, sensitive_features (e.g., gender series)
metrics = {
    "accuracy": accuracy_score,
    "selection_rate": selection_rate,
    "true_positive_rate": true_positive_rate,
    "false_positive_rate": false_positive_rate,
    "precision": precision_score
}

frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df_test["gender"]
)

print("By group:")
print(frame.by_group)
print()
print("Overall:")
print(frame.overall)
print()
print("Difference between groups:")
print(frame.difference())
```

**Interpreting output:**
- Small differences (< 0.05) → relatively fair
- Moderate (0.05–0.10) → warrants investigation
- Large (> 0.10) → significant disparity

### 4.2 Equalized Odds Difference

```python
from fairlearn.metrics import equalized_odds_difference

eod = equalized_odds_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df_test["race"]
)
print(f"Equalized Odds Difference: {eod:.4f}")
# 0.0 = perfect, 1.0 = maximum disparity
```

### 4.3 Visualizing Fairness

```python
from fairlearn.visualization import plot_model_comparison

# Compare across sensitive groups
frame.by_group.plot(kind="bar", figsize=(12, 6))
plt.title("Model Performance by Gender")
plt.ylabel("Metric Value")
plt.xticks(rotation=0)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
```

---

## 5. Bias Mitigation Techniques

Mitigation can happen at three stages:

### 5.1 Pre-processing (Fix the Data)

#### Reweighting
Give more weight to underrepresented or disadvantaged groups during training.

```python
from fairlearn.preprocessing import CorrelationRemover

# Remove features correlated with the sensitive attribute
cr = CorrelationRemover(sensitive_feature_ids=["gender"])
X_transformed = cr.fit_transform(X_train)

# Or manually compute sample weights
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)
# Adjust further based on group membership
```

#### Disparate Impact Remover

```python
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset

# Convert to AIF360 format
dataset = BinaryLabelDataset(
    df=df_train,
    label_names=["outcome"],
    protected_attribute_names=["race"]
)

di_remover = DisparateImpactRemover(repair_level=0.8)
dataset_fixed = di_remover.fit_transform(dataset)
```

### 5.2 In-processing (Constrain the Model)

Train the model with **fairness constraints** built in.

```python
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression

base_estimator = LogisticRegression()
constraint = EqualizedOdds()   # or DemographicParity(), TruePositiveRateParity()

mitigated_model = ExponentiatedGradient(
    estimator=base_estimator,
    constraints=constraint,
    eps=0.01   # maximum allowed constraint violation
)

mitigated_model.fit(X_train, y_train, sensitive_features=df_train["gender"])

y_pred_fair = mitigated_model.predict(X_test)
```

### 5.3 Post-processing (Adjust Predictions)

Change decision thresholds differently per group after training.

```python
from fairlearn.postprocessing import ThresholdOptimizer

postprocess_est = ThresholdOptimizer(
    estimator=base_estimator,
    constraints="equalized_odds",     # fairness criterion
    predict_method="predict_proba",
    objective="accuracy_score"
)

postprocess_est.fit(X_train, y_train, sensitive_features=df_train["gender"])
y_pred_fair = postprocess_est.predict(X_test, sensitive_features=df_test["gender"])

print("Fair model EOD:", equalized_odds_difference(y_test, y_pred_fair,
                                                    sensitive_features=df_test["gender"]))
```

### 5.4 The Fairness-Accuracy Tradeoff

```python
from fairlearn.reductions import GridSearch, EqualizedOdds

# Get the Pareto frontier of fairness vs. accuracy
grid_search = GridSearch(
    LogisticRegression(),
    constraints=EqualizedOdds(),
    grid_size=20
)
grid_search.fit(X_train, y_train, sensitive_features=df_train["gender"])

# Each point in predictors is a different tradeoff
for i, predictor in enumerate(grid_search.predictors_):
    y_pred_i = predictor.predict(X_test)
    acc = accuracy_score(y_test, y_pred_i)
    eod = equalized_odds_difference(y_test, y_pred_i, sensitive_features=df_test["gender"])
    print(f"Model {i}: Accuracy={acc:.3f}, EOD={eod:.3f}")
```

---

## 6. Intersectionality

Fairness analysis must consider **combinations** of protected attributes, not just one at a time.

```python
# Don't just check gender OR race — check gender AND race together
frame_intersectional = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df_test[["gender", "race"]]  # ← multiple attributes
)

print(frame_intersectional.by_group)
# Shows metrics for: (M, White), (M, Black), (F, White), (F, Black), etc.
```

---

## 7. Model Cards and Transparency

A **Model Card** documents a model's intended use, performance across groups, and limitations.

```markdown
# Model Card: Loan Approval Model v2.1

## Model Details
- **Type:** XGBoost Classifier
- **Version:** 2.1
- **Date:** 2025-03-15
- **Contact:** mlteam@company.com

## Intended Use
- **Primary use:** Loan eligibility screening for amounts under $50,000
- **Out-of-scope:** Not for use in criminal justice, hiring, or healthcare

## Training Data
- **Source:** Internal loan applications 2018-2024
- **Size:** 245,000 records
- **Protected attributes present:** Race (inferred via ZIP), Gender, Age

## Performance
| Group          | Accuracy | TPR   | FPR   |
|----------------|----------|-------|-------|
| Overall        | 0.887    | 0.842 | 0.103 |
| Male           | 0.891    | 0.855 | 0.097 |
| Female         | 0.882    | 0.823 | 0.112 |
| White          | 0.901    | 0.866 | 0.088 |
| Black          | 0.851    | 0.798 | 0.131 |
| Hispanic       | 0.863    | 0.811 | 0.121 |

## Fairness Metrics
- Equalized Odds Difference (Gender): 0.032
- Equalized Odds Difference (Race): 0.068 ⚠️

## Known Limitations
- ZIP code used as feature may proxy for race
- Underrepresentation of certain demographic groups in training data
- Model not validated outside the US

## Mitigation Efforts
- Applied ExponentiatedGradient with EqualizedOdds constraint
- Removed direct race/gender features; reviewed proxy features
- Regular audits scheduled quarterly
```

---

## 8. Practical Fairness Audit Checklist

Before deploying any model that affects people:

```
□ Identified all sensitive/protected attributes relevant to the domain
□ Checked representation of each group in training data
□ Measured performance separately for each demographic group
□ Computed Equalized Odds Difference and Demographic Parity Difference
□ Investigated intersectional fairness (group combinations)
□ Documented known proxy variables (zip code, name, etc.)
□ Applied and evaluated at least one mitigation technique
□ Documented the fairness-accuracy tradeoff chosen and rationale
□ Created a model card with per-group metrics
□ Set up ongoing monitoring for demographic performance drift
□ Defined a human review process for disputed decisions
```

---

## 9. 🧪 Practice Exercise

**Dataset:** [Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult) (predict income >$50K)

**Tasks:**
1. Train a baseline LogisticRegression model
2. Use `MetricFrame` to evaluate fairness by gender and race
3. Compute Equalized Odds Difference across groups
4. Apply `ExponentiatedGradient` with `EqualizedOdds` constraint
5. Plot the accuracy vs. fairness tradeoff for 10 different constraint levels
6. Write a one-page Model Card for the final model
7. Discuss: Which fairness definition is most appropriate for this use case?

---

## Key Takeaways

- ✅ ML models inherit and can amplify **historical biases** in data
- ✅ There is **no single correct definition** of fairness — context matters
- ✅ Always **measure bias** before and after mitigation
- ✅ Fairness often trades off against accuracy — **be transparent about the choice**
- ✅ Check **intersectional fairness**, not just individual attributes
- ✅ Document everything in a **Model Card**
- ✅ Fairness is not a one-time fix — **monitor continuously**

---

## 📚 Further Reading

- [Fairlearn Documentation](https://fairlearn.org/)
- [AIF360 by IBM](https://aif360.mybluemix.net/)
- [Google's Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/)
- [The Alignment Problem — Brian Christian](https://brianchristian.org/the-alignment-problem/)
- [ProPublica COMPAS Analysis](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- [Fairness and Machine Learning — Barocas, Hardt, Narayanan (free book)](https://fairmlbook.org/)

---

*← Previous: [05 - AutoML](./05_AutoML.md)*  
*Back to: [README — ML Course Index](./README.md)*
