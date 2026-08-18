"""
Train a loan default risk prediction model using the real
Kaggle "Credit Risk Dataset" (credit_risk_dataset.csv).

Run this once locally / on Streamlit Cloud build step is NOT needed --
we train here and save the model as a .pkl file that app.py loads.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("credit_risk_dataset.csv")

# ---------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------
# Remove obviously bad rows (data entry errors: age > 100, emp length > 60)
df = df[(df["person_age"] <= 100) & (df["person_age"] >= 18)]
df = df[(df["person_emp_length"].isna()) | (df["person_emp_length"] <= 60)]

# Fill missing numeric values with median
df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

# ---------------------------------------------------------
# 3. Encode categorical columns
# ---------------------------------------------------------
categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------------------------------------------------
# 4. Features / target
# ---------------------------------------------------------
feature_cols = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

X = df[feature_cols]
y = df["loan_status"]  # 1 = default, 0 = no default

# ---------------------------------------------------------
# 5. Train / test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 6. Train model
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 8. Save model + encoders + feature order
# ---------------------------------------------------------
joblib.dump(model, "loan_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

print("\nSaved: loan_model.pkl, encoders.pkl, feature_cols.pkl")
