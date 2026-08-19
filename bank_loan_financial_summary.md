# Bank Loan — Financial & Risk Impact Summary (Expected Loss Model)

**Expected Loss (EL)** is the standard credit-risk formula banks use to
provision for losses: **EL = PD × LGD × EAD**
- PD (Probability of Default): observed default rate
- LGD (Loss Given Default): assumed 60% (i.e. 40% recovery via collateral/collection)
- EAD (Exposure at Default): outstanding loan amount

## Portfolio-level
- Portfolio PD: **30.04%**
- Total EAD (portfolio exposure): **₹5,096,178,397**
- **Portfolio Expected Loss: ₹918,535,194**
  (i.e. the bank should provision roughly this much for expected credit losses)

## Expected Loss by credit band
    credit_band  customers       PD           EL  suggested_min_interest_rate_pct
     Poor(<600)     1460.0 0.369178 3.270532e+08                            27.15
  Fair(600-700)     2121.0 0.297030 3.910948e+08                            22.82
  Good(700-800)     1190.0 0.244538 1.765157e+08                            19.67
Excellent(800+)      229.0 0.183406 2.444652e+07                            16.00

## Risk-based pricing recommendation
Interest rates should scale with each band's expected loss rate plus a
5% target margin (see `suggested_min_interest_rate_pct`
column above) — rather than a flat rate for every applicant. This is the
same logic banks use to price loans by risk tier.

## Files
- `expected_loss_by_credit_band.csv` — full breakdown, ready for Power BI/Excel
