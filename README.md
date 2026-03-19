# 🛡️ FraudGuard — Financial Fraud Detection System

A production-grade fraud detection dashboard combining **Random Forest** (supervised)  
and **Isolation Forest** (anomaly detection) with a real-time Streamlit UI.

---

## Features
| Requirement | Implementation |
|---|---|
| Load CSV dataset | File uploader + synthetic fallback (5,000 rows) |
| Handle missing values | `DataFrame.fillna(median)` |
| Normalize features | `StandardScaler` on all numeric columns |
| Handle class imbalance | **SMOTE** oversampling on training set |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` (150 trees) |
| Isolation Forest | `sklearn.ensemble.IsolationForest` (anomaly score) |
| Fraud risk score 0–100 | 70% RF probability + 30% ISO anomaly score |
| Transaction table | Sortable / filterable dataframe with color-coded risk |
| Highlight suspicious | Risk badge: CRITICAL / HIGH / MEDIUM / LOW |
| Fraud vs normal charts | Donut, histogram, scatter, bar, time-series |
| "Why flagged?" section | Top-5 features weighted by `feature_importances_` × |value| |
| Model metrics | Confusion matrix, ROC curve, Classification report |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Using Your Own Dataset

Upload any CSV with:
- Numeric feature columns (V1–V28 or any names)
- Optional `Class` column: `0` = normal, `1` = fraud
- Optional `Amount` and `Time` columns

The app auto-detects columns and excludes `Class` from features.

---

## Architecture

```
CSV / Synthetic Data
        │
        ▼
  Preprocessing
  ┌─────────────────┐
  │ Fill NaN        │
  │ StandardScaler  │
  │ SMOTE (train)   │
  └────────┬────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
Random Forest  Isolation Forest
(supervised)   (anomaly detection)
     │            │
     │  70%  30%  │
     └──── blend ─┘
              │
         Risk Score
          0 – 100
              │
     Streamlit Dashboard
```

---

## Risk Levels

| Score | Level    | Color  |
|-------|----------|--------|
| 75–100 | CRITICAL | 🔴 Red |
| 50–74  | HIGH     | 🟠 Orange |
| 25–49  | MEDIUM   | 🟡 Yellow |
| 0–24   | LOW      | 🟢 Green |
