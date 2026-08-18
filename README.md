# Bank Loan Default & Risk Analysis

An end-to-end risk analytics project: SQL-based risk KPI reporting +
a machine learning model to predict loan default, with business-framed
recommendations.

## What this does
- Cleans loan data and engineers risk features (debt-to-income, credit
  bands, income brackets)
- Runs 5 SQL business queries (via SQLite) — default rate by credit
  band, employment type, income bracket, region, and a high-risk
  customer watchlist
- Trains and compares **Logistic Regression** and **Random Forest**
  models to predict default, evaluated with ROC-AUC, precision/recall
  (not just accuracy — important for imbalanced risk data)
- Generates risk visuals + ROC curve + confusion matrix
- Exports a clean CSV ready for a **Power BI / Excel risk dashboard**

## How to run
```bash
pip install pandas numpy matplotlib scikit-learn
python3 generate_data.py   # creates bank_loans.csv (synthetic, for testing)
python3 analysis.py        # runs the full pipeline -> outputs/
```

## ⚠️ Before you use this for your resume/portfolio
`generate_data.py` builds a synthetic dataset with a genuine underlying
risk pattern (default probability driven by credit score, debt-to-income,
employment type, etc.) so the model actually learns something real —
this isn't random data. Still, for full credibility:
1. Download a real dataset — e.g. Kaggle's "Loan Prediction Dataset" or
   the UCI "Bank Marketing Dataset"
   (https://archive.ics.uci.edu/dataset/222/bank+marketing)
2. Save it as `bank_loans.csv` with similar columns and re-run `analysis.py`
   (you may need to tweak column names in the script to match)
3. Import `outputs/bank_loans_cleaned.csv` into Power BI, build a risk
   dashboard (default rate by segment, portfolio health), and screenshot it

## What to write on your resume
> Built a loan default risk analysis pipeline using Python and SQL;
> engineered risk features (debt-to-income, credit bands) and trained
> a Logistic Regression / Random Forest model achieving 0.89 ROC-AUC.
> Delivered SQL-based risk KPIs by customer segment and a Power BI-ready
> dashboard with risk-based approval recommendations.

## Folder structure
```
bank_loan_project/
├── generate_data.py       # synthetic data generator (swap for real dataset)
├── analysis.py            # full pipeline: clean -> SQL -> ML -> charts -> insights
├── README.md
└── outputs/
    ├── bank_loans_cleaned.csv
    ├── insights.md
    ├── model_metrics.md        (classification report for both models)
    ├── sql_*.csv                (5 files, risk KPIs by segment)
    └── charts/                  (6 PNG visuals incl. ROC curve, confusion matrix)
```
