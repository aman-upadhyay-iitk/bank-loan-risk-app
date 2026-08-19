"""
financial_impact.py
--------------------
Finance/risk-analyst metrics: Expected Loss (EL = PD x LGD x EAD), a
core credit-risk concept used by real banks, plus portfolio-level
financial summary.

Run after generate_data.py:
    python3 financial_impact.py
Output:
    outputs/financial_summary.md
"""

import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("bank_loans.csv")

# ---------- Expected Loss (EL = PD x LGD x EAD) ----------
# PD  = Probability of Default -> use actual observed default rate per segment
# LGD = Loss Given Default -> assumption: bank recovers 40% via collateral/collections,
#       so LGD = 60% (industry-typical assumption for unsecured/semi-secured retail loans)
# EAD = Exposure at Default -> the loan amount outstanding

LGD = 0.60

df["EAD"] = df["loan_amount"]
portfolio_pd = df["default"].mean()
portfolio_ead = df["EAD"].sum()
portfolio_el = portfolio_pd * LGD * portfolio_ead

# EL by credit band
df["credit_band"] = pd.cut(df["credit_score"], bins=[0, 600, 700, 800, 900],
                            labels=["Poor(<600)", "Fair(600-700)", "Good(700-800)", "Excellent(800+)"])
band_summary = df.groupby("credit_band", observed=True).apply(
    lambda g: pd.Series({
        "customers": len(g),
        "PD": g["default"].mean(),
        "EAD": g["EAD"].sum(),
        "EL": g["default"].mean() * LGD * g["EAD"].sum()
    })
).reset_index()

# ---------- Risk-based pricing suggestion ----------
# Simple illustrative rule: charge enough interest to cover expected loss + margin
target_margin = 0.05  # 5% target margin over expected loss rate
band_summary["expected_loss_rate"] = band_summary["PD"] * LGD
band_summary["suggested_min_interest_rate_pct"] = (
    (band_summary["expected_loss_rate"] + target_margin) * 100
).round(2)

band_summary.to_csv("outputs/expected_loss_by_credit_band.csv", index=False)

summary = f"""# Bank Loan — Financial & Risk Impact Summary (Expected Loss Model)

**Expected Loss (EL)** is the standard credit-risk formula banks use to
provision for losses: **EL = PD × LGD × EAD**
- PD (Probability of Default): observed default rate
- LGD (Loss Given Default): assumed 60% (i.e. 40% recovery via collateral/collection)
- EAD (Exposure at Default): outstanding loan amount

## Portfolio-level
- Portfolio PD: **{portfolio_pd*100:.2f}%**
- Total EAD (portfolio exposure): **₹{portfolio_ead:,.0f}**
- **Portfolio Expected Loss: ₹{portfolio_el:,.0f}**
  (i.e. the bank should provision roughly this much for expected credit losses)

## Expected Loss by credit band
{band_summary[['credit_band','customers','PD','EL','suggested_min_interest_rate_pct']].to_string(index=False)}

## Risk-based pricing recommendation
Interest rates should scale with each band's expected loss rate plus a
{target_margin*100:.0f}% target margin (see `suggested_min_interest_rate_pct`
column above) — rather than a flat rate for every applicant. This is the
same logic banks use to price loans by risk tier.

## Files
- `expected_loss_by_credit_band.csv` — full breakdown, ready for Power BI/Excel
"""

with open("outputs/financial_summary.md", "w") as f:
    f.write(summary)

print(summary)
print("\nSaved -> outputs/financial_summary.md and outputs/expected_loss_by_credit_band.csv")
