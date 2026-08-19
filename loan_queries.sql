-- ============================================================
-- loan_queries.sql
-- Standalone SQL risk-KPI queries for the Bank Loan Default
-- Analysis. Import bank_loans.csv (or your real dataset) into
-- any SQL engine as a table named `loans` and run these directly.
-- ============================================================

-- 1. Overall portfolio default rate
SELECT
    ROUND(100.0 * SUM("default") / COUNT(*), 2) AS default_rate_pct,
    COUNT(*) AS total_customers
FROM loans;

-- 2. Default rate by credit score band
SELECT
    CASE
        WHEN credit_score < 600 THEN 'Poor (<600)'
        WHEN credit_score < 700 THEN 'Fair (600-700)'
        WHEN credit_score < 800 THEN 'Good (700-800)'
        ELSE 'Excellent (800+)'
    END AS credit_band,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM("default") / COUNT(*), 2) AS default_rate_pct
FROM loans
GROUP BY credit_band
ORDER BY default_rate_pct DESC;

-- 3. Default rate by employment type
SELECT
    employment_type,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM("default") / COUNT(*), 2) AS default_rate_pct
FROM loans
GROUP BY employment_type
ORDER BY default_rate_pct DESC;

-- 4. Default rate by region
SELECT
    region,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM("default") / COUNT(*), 2) AS default_rate_pct
FROM loans
GROUP BY region
ORDER BY default_rate_pct DESC;

-- 5. Portfolio value at risk (total loan amount held by defaulted accounts)
SELECT
    ROUND(SUM(CASE WHEN "default" = 1 THEN loan_amount ELSE 0 END), 2) AS loan_value_in_default,
    ROUND(SUM(loan_amount), 2) AS total_portfolio_value,
    ROUND(100.0 * SUM(CASE WHEN "default" = 1 THEN loan_amount ELSE 0 END) / SUM(loan_amount), 2) AS pct_portfolio_at_risk
FROM loans;

-- 6. High debt-to-income + poor credit watchlist
SELECT
    customer_id, income, credit_score, loan_amount,
    ROUND((loan_amount * 1.0 / tenure_years) / income, 3) AS debt_to_income
FROM loans
WHERE credit_score < 600 AND (loan_amount * 1.0 / tenure_years) / income > 0.4
ORDER BY debt_to_income DESC
LIMIT 25;

-- 7. Average loan size and default rate by income bracket
SELECT
    CASE
        WHEN income < 300000 THEN '<3L'
        WHEN income < 600000 THEN '3-6L'
        WHEN income < 1000000 THEN '6-10L'
        ELSE '10L+'
    END AS income_bracket,
    COUNT(*) AS customers,
    ROUND(AVG(loan_amount), 0) AS avg_loan_amount,
    ROUND(100.0 * SUM("default") / COUNT(*), 2) AS default_rate_pct
FROM loans
GROUP BY income_bracket
ORDER BY MIN(income);
