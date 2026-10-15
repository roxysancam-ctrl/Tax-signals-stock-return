"""Do companies that take longer to file their annual report lose more money?

Rosana's hypothesis: the longer a company takes to file its 10-K (the annual
report that carries its tax figures), the more likely it is to be losing money,
both in its own results and for shareholders.

Delay is measured against each company's own SEC deadline, because big
companies must file sooner: 60 days after year-end for large accelerated
filers, 75 for accelerated filers, 90 for everyone else. An NT 10-K notice buys
15 more days.

Outcomes
- loss_now:  pretax loss in the year the 10-K covers
- loss_next: pretax loss the following year (not yet known when filing)
- excess_ret_12m / beat_sp500: the stock's next 12 months vs the S&P 500

Tests
1. Groups by delay (early, last week, late with NT extension, very late).
2. Regressions holding size, industry and year constant, with standard
   errors clustered by company. For loss_next it also holds this year's loss
   and ROA constant: does slow filing warn of losses beyond what the latest
   results already show?

Input:  data/modeling_table.csv, data/tax_company_year.csv
Output: results/filing_delay_groups.csv, results/filing_delay_regressions.csv,
        results/filing_delay.png

Usage
    python filing_delay_test.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config
from train_models import RESULTS_DIR

DEADLINE = {2: 60, 1: 75, 0: 90}  # filer_status_est -> days after fiscal year-end
GROUPS = [(-999, -8, "Early (8+ days before deadline)"),
          (-7, 2, "Last week before deadline"),
          (3, 17, "Late, within NT extension"),
          (18, 999, "Very late (past extension)")]


def load():
    df = pd.read_csv(config.DATA_DIR / "modeling_table.csv")
    tax = pd.read_csv(config.DATA_DIR / "tax_company_year.csv",
                      usecols=["cik", "fiscal_year", "pretax_income"])
    nxt = tax.assign(fiscal_year=tax["fiscal_year"] - 1)
    nxt["loss_next"] = (nxt["pretax_income"] <= 0).astype(float).where(nxt["pretax_income"].notna())
    df = df.merge(nxt[["cik", "fiscal_year", "loss_next"]], on=["cik", "fiscal_year"], how="left")

    df = df[df["tenk_filing_lag_days"].notna() & df["filer_status_est"].notna()].copy()
    df["deadline"] = df["filer_status_est"].map(DEADLINE)
    df["days_vs_deadline"] = df["tenk_filing_lag_days"] - df["deadline"]
    df["delay_group"] = None
    for lo, hi, label in GROUPS:
        df.loc[df["days_vs_deadline"].between(lo, hi), "delay_group"] = label
    df["loss_now"] = df["pretax_loss"].astype(float).where(df["pretax_missing"] == 0)
    df["after_deadline"] = (df["days_vs_deadline"] > 2).astype(int)
    df["delay_10d"] = df["days_vs_deadline"].clip(-40, 80) / 10  # effect per 10 days
    return df


def group_table(df):
    g = df.groupby("delay_group", sort=False)
    out = pd.DataFrame({
        "company_years": g.size(),
        "share_loss_now": g["loss_now"].mean(),
        "share_loss_next_year": g["loss_next"].mean(),
        "median_roa": g["roa"].median(),
        "avg_excess_return": g["excess_ret_12m"].mean(),
        "median_excess_return": g["excess_ret_12m"].median(),
        "share_beat_sp500": g["beat_sp500"].mean(),
    }).reindex([label for _, _, label in GROUPS])
    # Same groups within the smallest and largest size quintiles: is it just size?
    for q in (1, 5):
        part = df[df["size_quintile"] == q].groupby("delay_group")["loss_now"].mean()
        out[f"share_loss_now_size_q{q}"] = part.reindex(out.index)
    return out


def regressions(df):
    controls = "log_total_assets + C(ff12_industry) + C(fiscal_year)"
    specs = [
        ("loss_now", "logit", f"loss_now ~ delay_10d + {controls}"),
        ("loss_next", "logit", f"loss_next ~ delay_10d + loss_now + roa + {controls}"),
        ("beat_sp500", "logit", f"beat_sp500 ~ delay_10d + loss_now + roa + {controls}"),
        ("excess_ret_12m", "ols", f"excess_ret_12m ~ delay_10d + loss_now + roa + {controls}"),
        ("loss_next", "logit", f"loss_next ~ after_deadline + loss_now + roa + {controls}"),
        ("excess_ret_12m", "ols", f"excess_ret_12m ~ after_deadline + loss_now + roa + {controls}"),
    ]
    rows = []
    for outcome, kind, formula in specs:
        cols = [c for c in ["loss_now", "loss_next", "beat_sp500", "excess_ret_12m", "delay_10d",
                            "after_deadline", "roa", "log_total_assets", "ff12_industry",
                            "fiscal_year", "cik"] if c in formula or c == "cik"]
        data = df[cols].dropna()
        fit_fn = smf.logit if kind == "logit" else smf.ols
        model = fit_fn(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data["cik"]},
                                          **({"disp": 0} if kind == "logit" else {}))
        var = "delay_10d" if "delay_10d" in formula else "after_deadline"
        coef, p = model.params[var], model.pvalues[var]
        lo, hi = model.conf_int().loc[var]
        row = {"outcome": outcome, "model": kind, "delay_measure": var, "n": int(model.nobs),
               "coef": coef, "ci_low": lo, "ci_high": hi, "p_value": p,
               "holds_constant": formula.split("~")[1].strip()}
        if kind == "logit":
            row["odds_ratio"] = np.exp(coef)
        rows.append(row)
    return pd.DataFrame(rows)


def chart(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = [-60, -15, -8, -4, -1, 2, 10, 17, 40, 200]
    labels = ["15+ early", "8-14 early", "4-7 early", "1-3 early", "on time",
              "3-10 late", "11-17 late", "18-40 late", "40+ late"]
    d = df.assign(bucket=pd.cut(df["days_vs_deadline"], bins, labels=labels))
    g = d.groupby("bucket", observed=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(labels, g["loss_now"].mean() * 100, marker="o", color="#e07b39", label="Loss this year")
    ax1.plot(labels, g["loss_next"].mean() * 100, marker="o", color="#b5462b", linestyle="--",
             label="Loss next year")
    ax1.plot(labels, (1 - g["beat_sp500"].mean()) * 100, marker="o", color="#2a6fdb",
             label="Stock trails S&P 500 next 12 months")
    ax1.set_ylabel("Share of company-years (%)")
    ax1.set_xlabel("Days the 10-K was filed before / after the company's SEC deadline")
    ax1.set_ylim(0, 100)
    ax1.set_title("Slower 10-K filers lose money more often (fiscal 2012-2024)")
    ax1.legend(loc="upper left")
    for x, n in zip(labels, g.size()):
        ax1.annotate(f"n={n:,}", (x, 3), ha="center", fontsize=8, color="#666")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "filing_delay.png", dpi=150)
    plt.close()


def main():
    df = load()
    groups = group_table(df)
    regs = regressions(df)
    groups.round(4).to_csv(RESULTS_DIR / "filing_delay_groups.csv")
    regs.round(5).to_csv(RESULTS_DIR / "filing_delay_regressions.csv", index=False)
    chart(df)
    pd.set_option("display.width", 200)
    print(f"{len(df):,} company-years with a 10-K filing date\n")
    print((groups * [1, 100, 100, 100, 100, 100, 100, 100, 100]).round(1).to_string())
    print()
    print(regs.drop(columns="holds_constant").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
