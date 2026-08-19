# Bank Loan Default Risk Prediction

Streamlit app that predicts loan default risk using a Random Forest model
trained on the real Kaggle "Credit Risk Dataset" (32,000+ real loan
applications, ~91% test accuracy).

## Files
- `app.py` — the Streamlit web app
- `train_and_save_model.py` — script that trains the model from the CSV
- `credit_risk_dataset.csv` — the real training data (from Kaggle)
- `loan_model.pkl`, `encoders.pkl`, `feature_cols.pkl` — already-trained model files (app loads these directly, no need to retrain on deploy)
- `requirements.txt` — Python packages needed

## Dataset source
Kaggle: "Credit Risk Dataset" by Lao Tse
https://www.kaggle.com/datasets/laotse/credit-risk-dataset

## SQL & Financial Analysis
Includes standalone SQL queries (`loan_queries.sql`) and an Expected Loss (EL = PD × LGD × EAD) financial risk model (`financial_impact.py`), the same credit-risk methodology used by banks.
