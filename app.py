import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Loan Default Risk Prediction", page_icon="🏦", layout="wide")

# ---------------------------------------------------------
# Load model, encoders, feature order
# ---------------------------------------------------------
model = joblib.load("loan_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_cols = joblib.load("feature_cols.pkl")

st.title("🏦 Bank Loan Default Risk Prediction")
st.write(
    "Enter a loan applicant's details below to predict their risk of default, "
    "using a Random Forest model trained on **real Kaggle credit risk data** "
    "(32,000+ loan applications)."
)

st.header("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 30)
    income = st.number_input("Annual Income (₹)", min_value=0, value=500000, step=10000)
    home_ownership = st.selectbox("Home Ownership", encoders["person_home_ownership"].classes_)
    emp_length = st.slider("Employment Length (years)", 0, 40, 5)
    loan_intent = st.selectbox("Loan Purpose", encoders["loan_intent"].classes_)
    loan_grade = st.selectbox("Loan Grade (A = best, G = worst)", sorted(encoders["loan_grade"].classes_))

with col2:
    loan_amnt = st.number_input("Loan Amount (₹)", min_value=0, value=500000, step=10000)
    int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, step=0.1)
    prior_default = st.selectbox("Prior Default on File?", encoders["cb_person_default_on_file"].classes_)
    cred_hist_length = st.slider("Credit History Length (years)", 0, 30, 5)

# loan_percent_income is derived, not asked directly
loan_percent_income = round(loan_amnt / income, 2) if income > 0 else 0
st.caption(f"Loan as % of income (auto-calculated): **{loan_percent_income*100:.1f}%**")

st.markdown("---")

if st.button("Predict Default Risk", type="primary"):
    input_dict = {
        "person_age": age,
        "person_income": income,
        "person_home_ownership": encoders["person_home_ownership"].transform([home_ownership])[0],
        "person_emp_length": emp_length,
        "loan_intent": encoders["loan_intent"].transform([loan_intent])[0],
        "loan_grade": encoders["loan_grade"].transform([loan_grade])[0],
        "loan_amnt": loan_amnt,
        "loan_int_rate": int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": encoders["cb_person_default_on_file"].transform([prior_default])[0],
        "cb_person_cred_hist_length": cred_hist_length,
    }

    input_df = pd.DataFrame([input_dict])[feature_cols]

    proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    risk_pct = proba * 100

    if prediction == 1:
        st.error(f"⚠️ High Default Risk — {risk_pct:.1f}% probability of default")
    else:
        st.success(f"✅ Low Default Risk — {risk_pct:.1f}% probability of default")

    st.progress(min(int(risk_pct), 100))

    with st.expander("Why this prediction? (Key risk factors)"):
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(5)
        st.write("The model weighs these factors most heavily overall:")
        st.bar_chart(feat_imp.set_index("Feature"))

st.markdown("---")
st.caption("Model trained on the Kaggle 'Credit Risk Dataset' (Lao Tse) — for educational/demo purposes only, not real financial advice.")
