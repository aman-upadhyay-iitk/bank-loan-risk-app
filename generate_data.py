"""
generate_data.py
-----------------
Generates a realistic synthetic bank loan dataset (mirrors the schema of
Kaggle's popular "Loan Prediction Dataset" / UCI "Bank Marketing" style
data): customer demographics, income, credit history, loan details, and
a default flag with a genuine underlying risk pattern (not random) so
the analysis and ML model actually learn something meaningful.

NOTE: This is synthetic data for you to test the full pipeline right
now. For your real resume/portfolio project, download a real dataset:
- Kaggle "Loan-Approval-Prediction-Dataset" or
- UCI "Bank Marketing Dataset" (https://archive.ics.uci.edu/dataset/222/bank+marketing)
Save it as bank_loans.csv with matching-ish columns and re-run analysis.py.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

age = np.random.randint(21, 65, n)
income = np.random.lognormal(mean=13.0, sigma=0.45, size=n).astype(int)  # annual income (INR)
loan_amount = np.random.randint(50000, 2000000, n)  # INR
credit_score = np.clip(np.random.normal(650, 90, n), 300, 900).astype(int)
tenure_years = np.random.choice([1, 2, 3, 5, 10, 15, 20], n)
employment_type = np.random.choice(["Salaried", "Self-Employed", "Business"], n, p=[0.55, 0.25, 0.20])
existing_loans = np.random.poisson(0.7, n)
marital_status = np.random.choice(["Single", "Married"], n, p=[0.4, 0.6])
region = np.random.choice(["North", "South", "East", "West", "Central"], n)
education = np.random.choice(["Graduate", "Not Graduate"], n, p=[0.7, 0.3])

# debt-to-income ratio drives real risk
dti = (loan_amount / tenure_years) / (income + 1)

# genuine risk score combining real factors (not random)
risk_score = (
    -3.4
    - 0.006 * (credit_score - 650)
    + 3.0 * dti
    + 0.25 * existing_loans
    + np.where(employment_type == "Self-Employed", 0.3, 0)
    + np.where(education == "Not Graduate", 0.25, 0)
)
prob_default = 1 / (1 + np.exp(-risk_score))
default = (np.random.rand(n) < prob_default).astype(int)

df = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
    "age": age,
    "income": income,
    "employment_type": employment_type,
    "education": education,
    "marital_status": marital_status,
    "region": region,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "tenure_years": tenure_years,
    "existing_loans": existing_loans,
    "default": default
})

df.to_csv("bank_loans.csv", index=False)
print(f"Generated {len(df)} rows -> bank_loans.csv")
print(f"Overall default rate: {df['default'].mean()*100:.2f}%")
