# 📊 ML Data: Numerical & Categorical Data

> **Crash Course** | Estimated Time: 45–60 minutes  
> **Prerequisites:** Basic Python, no ML experience required  
> **Series:** Machine Learning from Scratch

---

## 🎯 What You'll Learn

- The difference between numerical and categorical data
- How to identify and handle each type
- Feature engineering techniques for both types
- Encoding strategies for categorical variables
- Scaling strategies for numerical variables
- Hands-on examples with Python & pandas

---

## 1. Why Data Types Matter in ML

Machine learning algorithms are fundamentally mathematical — they perform operations like addition, multiplication, and distance calculations. This means:

- **Numbers** can be fed directly (after preprocessing)
- **Categories** (like "red", "blue", "New York") must be **converted** to numbers first

Getting this wrong is one of the most common reasons ML models fail in practice.

---

## 2. Numerical Data

### 2.1 What Is Numerical Data?

Numerical data represents **measurable quantities** — values where arithmetic makes sense.

| Type | Description | Examples |
|------|-------------|---------|
| **Continuous** | Any value in a range | Height (1.72m), Temperature (36.6°C), Price ($49.99) |
| **Discrete** | Countable whole numbers | Number of rooms (3), Page views (1402) |

### 2.2 Key Properties to Check

```python
import pandas as pd
import numpy as np

df = pd.read_csv("housing.csv")

# Quick statistical summary
print(df[["price", "sqft", "bedrooms"]].describe())

# Check for missing values
print(df.isnull().sum())

# Check skewness
print(df[["price", "sqft"]].skew())
```

**Output interpretation:**
- `mean` vs `median` far apart → likely skewed or outliers
- `skew > 1` or `skew < -1` → consider log transform
- `std` very large relative to `mean` → consider scaling

### 2.3 Handling Outliers

```python
import matplotlib.pyplot as plt

# Visualize with boxplot
df["price"].plot(kind="box")
plt.title("Price Distribution")
plt.show()

# IQR method to remove outliers
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[
    (df["price"] >= Q1 - 1.5 * IQR) &
    (df["price"] <= Q3 + 1.5 * IQR)
]
print(f"Removed {len(df) - len(df_clean)} outliers")
```

### 2.4 Scaling Numerical Features

Most ML models are sensitive to the **scale** of features. Two main strategies:

#### Min-Max Normalization (range: 0 to 1)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[["price_scaled", "sqft_scaled"]] = scaler.fit_transform(df[["price", "sqft"]])
```

**Use when:** You know the bounds of your data; neural networks and image data.

#### Standardization (Z-score: mean=0, std=1)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[["price_std", "sqft_std"]] = scaler.fit_transform(df[["price", "sqft"]])
```

**Use when:** Distribution is roughly Gaussian; algorithms like SVM, PCA, logistic regression.

#### Log Transform (for skewed data)

```python
df["price_log"] = np.log1p(df["price"])  # log1p handles zero values safely
```

**Use when:** Feature is heavily right-skewed (e.g., income, price, population).

### 2.5 Handling Missing Numerical Values

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")  # or "mean", "most_frequent"
df[["price", "sqft"]] = imputer.fit_transform(df[["price", "sqft"]])
```

> 💡 **Rule of thumb:** Use median for skewed data, mean for roughly normal data.

---

## 3. Categorical Data

### 3.1 What Is Categorical Data?

Categorical data represents **labels or groups** — values where arithmetic does NOT make sense.

| Type | Description | Examples |
|------|-------------|---------|
| **Nominal** | No inherent order | Color (red/blue/green), City, Gender |
| **Ordinal** | Has a meaningful order | Rating (Low/Med/High), Education level |
| **Binary** | Only two values | Yes/No, True/False, Spam/Not-spam |

### 3.2 Identifying Categorical Columns

```python
# Automatic detection
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
print("Categorical columns:", categorical_cols.tolist())

# Check cardinality (number of unique values)
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values — {df[col].unique()[:5]}")
```

### 3.3 Encoding Strategies

#### Label Encoding (for ordinal data)

```python
from sklearn.preprocessing import LabelEncoder

# Maps categories to integers: Low→0, Med→1, High→2
le = LabelEncoder()
df["rating_encoded"] = le.fit_transform(df["rating"])

# Better: manual mapping to preserve order
order_map = {"Low": 0, "Medium": 1, "High": 2}
df["rating_encoded"] = df["rating"].map(order_map)
```

⚠️ **Warning:** Don't use Label Encoding on **nominal** data — it implies a false order (e.g., "red"=0, "blue"=1, "green"=2 implies green > blue > red).

#### One-Hot Encoding (for nominal data)

```python
# pandas get_dummies
df = pd.get_dummies(df, columns=["color"], prefix="color", drop_first=True)

# sklearn approach (better for pipelines)
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded = ohe.fit_transform(df[["city"]])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(["city"]))
df = pd.concat([df, encoded_df], axis=1)
```

> 💡 Use `drop="first"` (or `drop_first=True`) to avoid **multicollinearity** (the dummy variable trap).

#### Target Encoding (for high-cardinality nominal data)

```python
# Replace each category with the mean of the target variable
target_mean = df.groupby("city")["price"].mean()
df["city_encoded"] = df["city"].map(target_mean)
```

**Use when:** Column has many unique values (50+), like city names or zip codes. One-hot encoding would create too many columns.

#### Frequency Encoding

```python
freq_map = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq_map)
```

### 3.4 Handling Missing Categorical Values

```python
# Option 1: Fill with most frequent value
df["city"].fillna(df["city"].mode()[0], inplace=True)

# Option 2: Create a new "Unknown" category
df["city"].fillna("Unknown", inplace=True)

# Option 3: Use sklearn imputer
imputer = SimpleImputer(strategy="most_frequent")
df[["city", "color"]] = imputer.fit_transform(df[["city", "color"]])
```

---

## 4. Building a Full Preprocessing Pipeline

Here's how to combine everything cleanly using `sklearn.pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define column types
numerical_features = ["sqft", "bedrooms", "age"]
categorical_features = ["city", "style", "condition"]

# Numerical pipeline
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

# Combine both
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Plug into full ML pipeline
from sklearn.linear_model import LinearRegression

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train
full_pipeline.fit(X_train, y_train)

# Predict
predictions = full_pipeline.predict(X_test)
```

---

## 5. Quick Reference Cheat Sheet

| Situation | Technique |
|-----------|-----------|
| Continuous feature, normal distribution | StandardScaler |
| Continuous feature, skewed | Log transform + StandardScaler |
| Feature with known bounds | MinMaxScaler |
| Nominal categorical, low cardinality (<15) | OneHotEncoder |
| Ordinal categorical | LabelEncoder with manual map |
| Nominal categorical, high cardinality (50+) | TargetEncoder or FrequencyEncoder |
| Missing numerical values | SimpleImputer (median/mean) |
| Missing categorical values | SimpleImputer (most_frequent) or fill "Unknown" |

---

## 6. 🧪 Practice Exercise

**Dataset:** [Titanic dataset](https://www.kaggle.com/c/titanic/data)

**Tasks:**
1. Identify all numerical and categorical columns
2. Check for missing values — handle them appropriately
3. Apply the correct encoding to `Sex`, `Embarked`, and `Pclass`
4. Scale `Age` and `Fare` using StandardScaler
5. Build a preprocessing pipeline and train a Logistic Regression classifier
6. Report accuracy on the test set

---

## 7. Key Takeaways

- ✅ Always identify your data types **before** modeling
- ✅ Numerical data needs **scaling** and **outlier handling**
- ✅ Categorical data needs **encoding** — choose based on ordinality and cardinality
- ✅ Use `sklearn` **Pipelines** to avoid data leakage and keep code clean
- ✅ Preprocessing must be **fit on training data only**, then applied to test data

---

## 📚 Further Reading

- [Scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas Data Types Documentation](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes)
- [Feature Engineering for Machine Learning (Book) — Alice Zheng](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

---

*Next in series → [02 - Datasets, Generalization & Overfitting](./02_Datasets_Generalization_Overfitting.md)*
