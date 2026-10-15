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
