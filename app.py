"""
app.py
------
Streamlit app: Bank Loan Default Risk Prediction — live demo.

Run locally:
    pip install streamlit pandas scikit-learn
    streamlit run app.py
"""

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Risk Prediction Demo", page_icon="🏦", layout="centered")

st.title("🏦 Bank Loan Default Risk Prediction")
st.write(
    "Enter a loan applicant's details below to predict their risk of "
    "default, using a Logistic Regression model trained on customer "
    "income, credit score, and loan data."
)

@st.cache_resource
def load_model():
    with open("loan_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run `python3 train_and_save_model.py` first.")
    st.stop()

st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 21, 65, 30)
    income = st.number_input("Annual Income (₹)", 100000, 3000000, 500000, step=10000)
    credit_score = st.slider("Credit Score", 300, 900, 650)
    loan_amount = st.number_input("Loan Amount (₹)", 50000, 2000000, 500000, step=10000)

with col2:
    tenure_years = st.selectbox("Tenure (years)", [1, 2, 3, 5, 10, 15, 20], index=3)
    existing_loans = st.slider("Existing Loans", 0, 5, 0)
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])

if st.button("Predict Default Risk", type="primary"):
    annual_installment = loan_amount / tenure_years
    debt_to_income = annual_installment / income

    input_df = pd.DataFrame([{
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "tenure_years": tenure_years,
        "existing_loans": existing_loans,
        "debt_to_income": debt_to_income,
        "employment_type": employment_type,
        "education": education,
        "marital_status": marital_status,
        "region": region
    }])

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Result")
    st.metric("Default Probability", f"{prob*100:.1f}%")
    st.caption(f"Debt-to-Income Ratio: {debt_to_income:.2f}")

    if pred == 1:
        st.error("⚠️ High risk — recommend tighter approval terms or risk-based pricing.")
    else:
        st.success("✅ Low risk — favorable candidate for standard approval terms.")

    st.progress(min(int(prob * 100), 100))

    with st.expander("Why this prediction? (key risk factors)"):
        notes = []
        if credit_score < 600:
            notes.append("Credit score below 600 is a strong default risk indicator.")
        if debt_to_income > 0.4:
            notes.append("Debt-to-income ratio above 0.4 indicates high repayment burden.")
        if employment_type == "Self-Employed":
            notes.append("Self-employed applicants show elevated risk in this dataset.")
        if education == "Not Graduate":
            notes.append("Non-graduate applicants show slightly higher default rates.")
        if existing_loans >= 2:
            notes.append("Multiple existing loans increase repayment risk.")
        if not notes:
            notes.append("This applicant's profile matches historically low-risk patterns.")
        for n in notes:
            st.write(f"- {n}")

st.divider()
st.caption(
    "Model: Logistic Regression trained on bank loan customer data. "
    "Built as a portfolio project — see the GitHub repo for the full "
    "analysis pipeline (SQL risk KPIs, model comparison)."
)
