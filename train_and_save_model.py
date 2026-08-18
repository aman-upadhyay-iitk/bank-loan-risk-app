"""
train_and_save_model.py
------------------------
Trains the loan default prediction model and saves it as loan_model.pkl
for the Streamlit app.

Run:
    python3 train_and_save_model.py
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("bank_loans.csv")

df["annual_installment"] = df["loan_amount"] / df["tenure_years"]
df["debt_to_income"] = df["annual_installment"] / df["income"]

features_num = ["age", "income", "credit_score", "loan_amount", "tenure_years",
                 "existing_loans", "debt_to_income"]
features_cat = ["employment_type", "education", "marital_status", "region"]

X = df[features_num + features_cat]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat)
], remainder="passthrough")

model = LogisticRegression(max_iter=1000, class_weight="balanced")
pipe = Pipeline([("prep", preprocessor), ("clf", model)])
pipe.fit(X_train, y_train)

with open("loan_model.pkl", "wb") as f:
    pickle.dump(pipe, f)

acc = pipe.score(X_test, y_test)
print(f"Model trained and saved -> loan_model.pkl (test accuracy: {acc:.3f})")
