# Step 6 results: the four models

Made by `train_models.py` on 15 Oct 2026. Train = fiscal 2012-2020 (26,660 company-years, 40.3% beat the S&P 500), test = fiscal 2021-2024 (17,008, 33.9% beat). Settings were picked by fitting on 2012-2017 and scoring on 2018-2020; the test years were used once, at the end.

## How well they predict (test years 2021-2024)

| Model | Test AUC (95% range) | Same model, controls only | Gain from tax + filing + abatements | Top 10% picks that beat |
|---|---|---|---|---|
| **Random Forest** | **0.564** (0.555-0.572) | 0.557 | **+0.006** (+0.003 to +0.010) | 36.0% |
| AdaBoost | 0.545 (0.536-0.553) | 0.544 | +0.001 (-0.003 to +0.004) | 34.5% |
| Logit | 0.544 (0.535-0.553) | 0.540 | +0.004 (-0.000 to +0.008) | 36.9% |
| XGBoost | 0.542 (0.533-0.551) | 0.542 | -0.000 (-0.004 to +0.004) | 34.2% |

(Base rate: 33.9% of test stocks beat the index.) Ranges come from 500 bootstrap resamples of the test rows.

- All four are clearly better than a coin flip (0.50), and Random Forest lands in the 0.55-0.60 range set as "good" for this problem.
- Unlike the churn capstone, XGBoost is not the winner here. It had the best score on 2018-2020 (0.570) but the lowest on 2021-2024: with this little signal, the extra flexibility mostly fits noise.
- The honest headline: **a model with only the controls (size, profitability, industry, last year's return...) does almost as well.** Only Random Forest's gain from the tax and filing signals is clearly above zero, and it is small.
- AUC by test year (Random Forest): 2021 0.626, 2022 0.549, 2023 0.551, 2024 0.535. The signal is strongest in 2021 and fades.

## Which signals matter (permutation importance, test years)

Shuffling a feature (with its blank flag, or all categories of an industry/exchange/region) and measuring the drop in test AUC. Bigger drop = the model relies on it more. Files: `permutation_importance.csv`, `feature_importance.png`, `importance_by_group.png`.

Drop in test AUC when a whole group is shuffled:

| Group | Logit | Random Forest | AdaBoost | XGBoost |
|---|---|---|---|---|
| Tax | 0.038 | 0.020 | 0.015 | 0.028 |
| Filing | 0.004 | 0.003 | 0.007 | 0.009 |
| Abatement | 0.000 | 0.000 | 0.000 | 0.000 |
| Controls | -0.005 | 0.007 | 0.004 | -0.001 |

- **Tax signals are the group the models lean on most.** The single tax features that matter most across models are **cash taxes paid vs tax expense** (`cash_to_book_tax`), **change in ETR**, **cash ETR**, and whether pretax income is reported at all. The plain ETR adds little once these are in.
- **Filing:** how fast the 10-K comes out (`filing_lag_days`, `tenk_filing_lag_days`) matters more than the NT late-filing notices themselves. Logit says an NT 10-K lowers the odds of beating the index, as expected, but late filers are rare (about 9%).
- **Abatements: zero.** Only 348 of 43,668 company-years have one (NY only, from 2018), too few for any model to use. Report this as inconclusive, not as "abatements don't matter".
- **Why tax matters in importance but adds little over controls:** tax figures overlap heavily with size and profitability (loss-making small companies have blank or odd tax ratios). When tax is there the models use it; when it is taken away, size and ROA carry nearly the same information. That is the "correlation is not causation" point in practice.
- **Last year's return hurts in the test years** (negative importance in all four models): the training years taught "winners keep winning", and that pattern flipped in 2021-2024.

## Choices made in this step

- Blank ETR: trees use -1, logit uses the training median plus the loss / tiny-profit flags (Rosana's rule).
- Other blanks (ETR change, deferred tax ratios, prior return, lags...): training median plus a 0/1 `_blank` flag, the same for every model.
- Classes weighted equally in all models (40% / 34% beat), so the 0.5 cut-off gives sensible recall. AUC does not depend on this.
- Not used as features: `delisted_in_window` (only known after the fact) and data-coverage flags (`from_comparative`, `filing_row_missing`, `abate_ny_covered`, `abate_data_year`).

## Files

| File | What |
|---|---|
| `model_metrics.csv` | AUC (with range), accuracy, precision, recall, F1, top-10% beat rate, gain vs controls-only |
| `auc_by_test_year.csv` | Test AUC for each year 2021-2024 |
| `tuning_2018_2020.csv`, `best_params.json` | Every setting tried and the one chosen |
| `permutation_importance.csv` | Per feature and per group, all four models |
| `logit_coefficients.csv` | Logit coefficients (per standard deviation), for the direction of each effect |
| `feature_importance.png`, `importance_by_group.png`, `roc_curves.png` | Charts |
| `../data/test_predictions.csv` | Each test company-year with every model's probability, for step 7 (portfolio test) |

## Saved models

`save_models.py` writes the four trained models (full feature set, trained on 2012-2020) to `models/logit.pkl`, `models/rf.pkl`, `models/ada.pkl`, `models/xgb.pkl`. Each reproduces the test AUC above. Load one with `joblib.load("models/rf.pkl")`, and use `prepare()` from `save_models.py` to turn rows shaped like `modeling_table.csv` into the model's inputs.

# Step 7 results: portfolio test (light version)

Made by `portfolio_test.py`. Each test year, stocks are ranked by a model's probability of beating the S&P 500, and equal-weighted portfolios are held for the 12 months after each 10-K (the same window as the target). Excess return = portfolio minus the S&P 500 over the same days. 95% ranges come from resampling the stocks 2,000 times. No trading costs, no monthly returns (so no Sharpe ratio or Fama-French alpha: four years are too few for those).

Average per test year (Random Forest unless noted):

| Portfolio | Return | S&P 500 | Excess return (95% range) | Years it beat the S&P 500 | Stocks that beat |
|---|---|---|---|---|---|
| Top 10% | 6.7% | 14.6% | **-7.9%** (-10.2 to -5.5) | 0 of 4 | 36.2% |
| Top 20% | 7.4% | 14.5% | -7.1% (-8.7 to -5.4) | 0 of 4 | 36.4% |
| All stocks | 4.2% | 14.5% | -10.3% (-11.4 to -9.2) | 0 of 4 | 33.9% |
| Bottom 10% | -15.3% | 14.7% | **-30.0%** (-35.3 to -24.6) | 0 of 4 | 21.1% |
| Top 10%, controls-only model | 5.9% | 14.6% | -8.8% (-11.0 to -6.5) | 0 of 4 | 36.0% |

Excess return by year (Random Forest): top 10% -1.1%, -9.7%, -14.2%, -6.6%; all stocks -8.6%, -19.3%, -11.0%, -2.3% (fiscal 2021-2024). Chart: `portfolio_returns.png`. Other models' portfolios: `portfolio_summary.csv`, `portfolio_by_year.csv`.

What it means:
- **The top picks did not beat the S&P 500** in any test year. 2021-2024 was a period when a handful of mega-caps carried the index and the typical stock lagged it by about 10% a year.
- **The model is much better at spotting losers than winners.** The top 10% beat the average stock by about 2.4 points a year, but the bottom 10% trailed it by about 20 points. An investor could use it to avoid stocks more than to pick them.
- **Tax and filing signals add about 0.9 points a year** to the top 10% over the controls-only model (-7.9% vs -8.8%), well within the noise.
- The top 10% only beat the average stock in 2 of 4 years (2021 and 2022), so even the stock-picking edge is not steady.

# Do slow filers lose money? (`filing_delay_test.py`)

Rosana's hypothesis: the longer a company takes to file its 10-K (the annual report with its tax figures), the more likely it is to be losing money. Delay is measured against each company's own SEC deadline (60 days after year-end for large accelerated filers, 75 for accelerated, 90 for the rest; an NT 10-K buys 15 more days). Fiscal 2012-2024, 41,024 company-years. Chart: `filing_delay.png`.

| How the 10-K was filed | Company-years | Loss this year | Loss next year | Median ROA | Median return vs S&P 500 | Beat the S&P 500 |
|---|---|---|---|---|---|---|
| Early (8+ days before deadline) | 15,426 | 30.3% | 30.1% | 2.0% | -8.1% | 40.3% |
| Last week before deadline | 20,635 | 31.7% | 32.4% | 1.2% | -10.3% | 38.6% |
| Late, within NT extension | 3,498 | 66.6% | 67.2% | -23.6% | -33.3% | 27.7% |
| Very late (past extension) | 1,465 | 74.2% | 74.5% | -36.3% | -35.6% | 31.0% |

**It is not just that small companies file late.** Among the smallest fifth of companies, 75% of early filers lost money vs 88% of very late ones; among the largest fifth, 9% vs 39%.

Regressions holding size, industry and year constant (standard errors clustered by company; `filing_delay_regressions.csv`):

- **Losing money this year:** each extra 10 days relative to the deadline raises the odds of a pretax loss by 9% (p < 0.001).
- **Losing money next year**, even after holding this year's loss and ROA constant: +7% odds per 10 days (p < 0.001). Filing after the deadline raises the odds by 48% (odds ratio 1.48, 95% range 1.32-1.66). So slow filing warns of future losses beyond what the latest results already show.
- **Stock performance over the next 12 months**, also holding this year's loss and ROA constant: filing after the deadline goes with returns 9.6 points lower vs the S&P 500 (95% range -13.4 to -5.9, p < 0.001); odds of beating the index fall 4% per 10 days (p < 0.001).

Caveats: this is correlation, not cause (late filing is usually a symptom of accounting or business trouble, not its source). The "very late" group's average return (-4.4%) is much better than its median (-35.6%) because a few of these stocks soar; most of them fall. Delisted companies are mostly missing (survivorship bias), which probably makes late filers look *better* than they really did.

# Tax and filing red flags on their own (`results/tax_flags_check.py`)

A Random Forest using **only** tax, filing and abatement features (no size, profitability or industry) reaches test AUC 0.574. Its bottom 10% trailed the S&P 500 by 26.9% a year (95% range -31.8 to -21.8) in every test year, vs -10.3% for all stocks.

Average return vs the S&P 500 over the next 12 months, test years 2021-2024:

| Red flag | Stocks per year | Return vs S&P 500 | Beat the S&P 500 |
|---|---|---|---|
| All stocks | 4,252 | -10.3% | 33.9% |
| Any late-filing notice (NT 10-K or 10-Q) | 564 | -24.5% | 23.3% |
| Late 10-K (NT 10-K) | 389 | -22.2% | 23.2% |
| Highest 10% valuation allowance (tax assets the company doubts it can use) | 342 | -25.9% | 22.9% |
| Pretax loss | 1,538 | -15.2% | 28.8% |
| Profitable (positive pretax income) | 2,127 | -5.9% | 38.8% |

Each red flag trailed the S&P 500 in all four test years.
