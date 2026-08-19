"""
app.py  (with SQL Insights + Expected Loss tab)
-------------------------------------------------
Streamlit app: Bank Loan Default Risk Prediction — live demo.
Includes a live SQL Insights tab (runs loan_queries.sql live via SQLite)
and Expected Loss (PD x LGD x EAD) financial metrics.

Run locally:
    pip install streamlit pandas scikit-learn
    streamlit run app.py
"""

import pickle
import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Risk Prediction Demo", page_icon="🏦", layout="wide")

st.title("🏦 Bank Loan Default Risk Prediction")
st.write(
    "Live demo with an ML default-risk predictor, live SQL risk KPIs, and "
    "Expected Loss (EL = PD × LGD × EAD) — the credit-risk formula banks actually use."
)

tab1, tab2, tab3 = st.tabs(["🔮 Risk Predictor", "🗄️ SQL Insights", "💰 Expected Loss"])

# ============================================================
# TAB 1: PREDICTOR
# ============================================================
with tab1:
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
            "age": age, "income": income, "credit_score": credit_score,
            "loan_amount": loan_amount, "tenure_years": tenure_years,
            "existing_loans": existing_loans, "debt_to_income": debt_to_income,
            "employment_type": employment_type, "education": education,
            "marital_status": marital_status, "region": region
        }])

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Default Probability", f"{prob*100:.1f}%")
        LGD = 0.60
        expected_loss = prob * LGD * loan_amount
        c2.metric("Expected Loss (this loan)", f"₹{expected_loss:,.0f}")
        st.caption(f"Debt-to-Income Ratio: {debt_to_income:.2f}  |  EL = PD × LGD(60%) × Loan Amount")

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
    st.caption("Model: Logistic Regression trained on bank loan customer data.")

# ============================================================
# TAB 2: SQL INSIGHTS
# ============================================================
with tab2:
    st.subheader("Live SQL Risk KPI Queries")
    st.write(
        "These queries run **live** against the loan portfolio using SQL (SQLite) — "
        "same queries are also in `loan_queries.sql` for use in any SQL tool."
    )

    @st.cache_data
    def load_data():
        return pd.read_csv("bank_loans.csv")

    df = load_data()
    conn = sqlite3.connect(":memory:")
    df.to_sql("loans", conn, index=False, if_exists="replace")

    queries = {
        "Overall portfolio default rate": """
            SELECT ROUND(100.0 * SUM("default") / COUNT(*), 2) AS default_rate_pct,
                   COUNT(*) AS total_customers
            FROM loans;
        """,
        "Default rate by credit band": """
            SELECT
                CASE WHEN credit_score < 600 THEN 'Poor (<600)'
                     WHEN credit_score < 700 THEN 'Fair (600-700)'
                     WHEN credit_score < 800 THEN 'Good (700-800)'
                     ELSE 'Excellent (800+)' END AS credit_band,
                COUNT(*) AS customers,
                ROUND(100.0 * SUM("default") / COUNT(*), 2) AS default_rate_pct
            FROM loans GROUP BY credit_band ORDER BY default_rate_pct DESC;
        """,
        "Default rate by employment type": """
            SELECT employment_type, COUNT(*) AS customers,
                   ROUND(100.0 * SUM("default") / COUNT(*), 2) AS default_rate_pct
            FROM loans GROUP BY employment_type ORDER BY default_rate_pct DESC;
        """,
        "Default rate by region": """
            SELECT region, COUNT(*) AS customers,
                   ROUND(100.0 * SUM("default") / COUNT(*), 2) AS default_rate_pct
            FROM loans GROUP BY region ORDER BY default_rate_pct DESC;
        """,
        "Portfolio value at risk": """
            SELECT ROUND(SUM(CASE WHEN "default"=1 THEN loan_amount ELSE 0 END), 2) AS loan_value_in_default,
                   ROUND(SUM(loan_amount), 2) AS total_portfolio_value
            FROM loans;
        """,
        "Top 25 high-risk watchlist": """
            SELECT customer_id, income, credit_score, loan_amount
            FROM loans
            WHERE credit_score < 600 AND (loan_amount * 1.0 / tenure_years) / income > 0.4
            ORDER BY loan_amount DESC LIMIT 25;
        """,
    }

    query_choice = st.selectbox("Choose a business question", list(queries.keys()))
    result = pd.read_sql_query(queries[query_choice], conn)

    st.code(queries[query_choice].strip(), language="sql")
    st.dataframe(result, use_container_width=True)

    if result.shape[1] == 3 and result.shape[0] > 1:
        chart_col = result.columns[0]
        value_col = result.columns[-1]
        st.bar_chart(result.set_index(chart_col)[value_col])

# ============================================================
# TAB 3: EXPECTED LOSS (finance/credit-risk concept)
# ============================================================
with tab3:
    st.subheader("Expected Loss Model (EL = PD × LGD × EAD)")
    st.write(
        "**EL** is the standard credit-risk formula banks use to provision for losses. "
        "PD = observed default rate, LGD = Loss Given Default (assumed 60%), "
        "EAD = Exposure at Default (loan amount)."
    )

    df2 = load_data()
    LGD = 0.60
    df2["credit_band"] = pd.cut(df2["credit_score"], bins=[0, 600, 700, 800, 900],
                                 labels=["Poor(<600)", "Fair(600-700)", "Good(700-800)", "Excellent(800+)"])

    band = df2.groupby("credit_band", observed=True).apply(
        lambda g: pd.Series({
            "Customers": len(g),
            "PD (%)": round(g["default"].mean() * 100, 2),
            "EAD (₹)": g["loan_amount"].sum(),
            "Expected Loss (₹)": round(g["default"].mean() * LGD * g["loan_amount"].sum(), 0),
        })
    ).reset_index()
    band["Suggested Min Interest Rate (%)"] = round(
        (band["PD (%)"] / 100 * LGD + 0.05) * 100, 2
    )

    portfolio_pd = df2["default"].mean()
    portfolio_ead = df2["loan_amount"].sum()
    portfolio_el = portfolio_pd * LGD * portfolio_ead

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio PD", f"{portfolio_pd*100:.2f}%")
    c2.metric("Total Exposure (EAD)", f"₹{portfolio_ead:,.0f}")
    c3.metric("Portfolio Expected Loss", f"₹{portfolio_el:,.0f}")

    st.write("**Expected Loss by Credit Band** (with risk-based pricing suggestion)")
    st.dataframe(band, use_container_width=True)
    st.bar_chart(band.set_index("credit_band")["Expected Loss (₹)"])

    st.caption(
        "Risk-based pricing: interest rates should scale with each band's expected "
        "loss rate plus a target margin — not a flat rate for every applicant."
    )
