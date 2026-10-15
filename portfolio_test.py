"""Step 7 (light version): would the model's top picks have beaten the S&P 500?

Input:  data/test_predictions.csv (from train_models.py), data/modeling_table.csv,
        results/best_params.json
Output: results/portfolio_by_year.csv, results/portfolio_summary.csv,
        results/portfolio_returns.png

How it works
- Each test fiscal year (2021-2024) is one "cohort". Within it, stocks are
  ranked by a model's probability of beating the S&P 500.
- Portfolios are equal-weighted: top 10%, top 20%, bottom 10% and all stocks.
- Each stock is held for the same 12 months as the target: from the first
  trading day after its 10-K was filed. The S&P 500 (SPY) return is measured
  over exactly the same days, so "excess return" = stock minus SPY.
- 95% ranges come from resampling the stocks in each cohort 2,000 times.
- The controls-only Random Forest is refit (same settings train_models.py
  chose) so its top 10% can be compared: do the tax and filing signals add money?

This is the light version: no monthly returns, so no Sharpe ratio or
Fama-French alpha. With only four test years those would be too noisy anyway.
No trading costs; survivorship bias as described in the README.

Usage
    python portfolio_test.py
"""
import json

import numpy as np
import pandas as pd

import config
from train_models import (FEATURE_SETS, MODEL_NAMES, RESULTS_DIR, design_matrix, fit, log,
                          make_model)

SEED = 42
BOOT = 2000


def controls_only_rf_probs():
    df = pd.read_csv(config.DATA_DIR / "modeling_table.csv")
    train_rows = (df["split"] == "train").to_numpy()
    y = df["beat_sp500"].to_numpy()
    params = json.loads((RESULTS_DIR / "best_params.json").read_text())["controls_only/rf"]
    X, _ = design_matrix(df, FEATURE_SETS["controls_only"], "tree", train_rows)
    model = fit(make_model("rf", params, y[train_rows]), "rf",
                X[train_rows].reset_index(drop=True), y[train_rows])
    out = df.loc[~train_rows, ["cik", "fiscal_year"]].copy()
    out["prob_rf_controls"] = model.predict_proba(X[~train_rows])[:, 1]
    return out


def portfolios(pred, prob_col):
    """{portfolio name: boolean mask} for one ranking column."""
    rank = pred.groupby("fiscal_year")[prob_col].rank(pct=True, method="first")
    return {"Top 10%": rank > 0.9, "Top 20%": rank > 0.8, "Bottom 10%": rank <= 0.1}


def summarize(pred, mask, label, model_label, rng):
    """Per-year and pooled stats for one portfolio, with a bootstrap range on excess return."""
    rows, boots = [], []
    for year, part in pred[mask].groupby("fiscal_year"):
        rows.append({"model": model_label, "portfolio": label, "fiscal_year": year,
                     "n_stocks": len(part), "return": part["ret_12m"].mean(),
                     "sp500_return": part["sp500_ret_12m"].mean(),
                     "excess_return": part["excess_ret_12m"].mean(),
                     "median_excess": part["excess_ret_12m"].median(),
                     "share_beat": part["beat_sp500"].mean()})
        ex = part["excess_ret_12m"].to_numpy()
        boots.append(ex[rng.integers(0, len(ex), (BOOT, len(ex)))].mean(axis=1))
    by_year = pd.DataFrame(rows)
    avg_boot = np.mean(boots, axis=0)  # average of the four cohorts, per resample
    summary = {"model": model_label, "portfolio": label,
               "avg_stocks_per_year": by_year["n_stocks"].mean(),
               "avg_return": by_year["return"].mean(),
               "avg_sp500_return": by_year["sp500_return"].mean(),
               "avg_excess_return": by_year["excess_return"].mean(),
               "excess_low": np.percentile(avg_boot, 2.5),
               "excess_high": np.percentile(avg_boot, 97.5),
               "years_beating_sp500": int((by_year["excess_return"] > 0).sum()),
               "share_beat": by_year["share_beat"].mean()}
    return by_year, summary


def chart(by_year):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rf = by_year[by_year["model"] == "Random Forest"]
    table = rf.pivot(index="fiscal_year", columns="portfolio", values="excess_return")
    table = table[["Top 10%", "Top 20%", "All stocks", "Bottom 10%"]] * 100
    ax = table.plot.bar(figsize=(9, 5), rot=0,
                        color=["#2a6fdb", "#7fa7ea", "#9a9a9a", "#e07b39"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Return minus S&P 500 over the same 12 months (%)")
    ax.set_xlabel("Fiscal year of the 10-K (held 12 months after filing)")
    ax.set_title("Random Forest portfolios vs the S&P 500, test years")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "portfolio_returns.png", dpi=150)
    plt.close()


def main():
    rng = np.random.default_rng(SEED)
    pred = pd.read_csv(config.DATA_DIR / "test_predictions.csv")
    log("Refitting the controls-only Random Forest")
    pred = pred.merge(controls_only_rf_probs(), on=["cik", "fiscal_year"], how="left")

    by_year, summary = [], []
    by, s = summarize(pred, pd.Series(True, index=pred.index), "All stocks", "None", rng)
    by_year.append(by.assign(model="Random Forest"))  # shown next to the RF portfolios
    summary.append(s)
    rankings = {f"prob_{k}": v for k, v in MODEL_NAMES.items()}
    rankings["prob_rf_controls"] = "Random Forest, controls only"
    for col, model_label in rankings.items():
        for label, mask in portfolios(pred, col).items():
            by, s = summarize(pred, mask, label, model_label, rng)
            by_year.append(by)
            summary.append(s)

    by_year = pd.concat(by_year, ignore_index=True)
    summary = pd.DataFrame(summary)
    by_year.round(4).to_csv(RESULTS_DIR / "portfolio_by_year.csv", index=False)
    summary.round(4).to_csv(RESULTS_DIR / "portfolio_summary.csv", index=False)
    chart(by_year)

    pct = summary.copy()
    for c in ["avg_return", "avg_sp500_return", "avg_excess_return", "excess_low", "excess_high", "share_beat"]:
        pct[c] = (pct[c] * 100).round(1)
    log("Average per test year (%):\n" + pct.drop(columns="avg_stocks_per_year").to_string(index=False))
    rf = by_year[by_year["model"] == "Random Forest"].pivot(index="fiscal_year", columns="portfolio",
                                                            values="excess_return")
    log("Random Forest, excess return by year (%):\n" + (rf * 100).round(1).to_string())


if __name__ == "__main__":
    main()
