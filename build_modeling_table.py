"""Join everything into one modeling table: one row per company-year.

Inputs (all in data/, joined on cik + fiscal_year):
- tax_company_year.csv       tax features, profitability, filer status
- filing_company_year.csv    late-filing (NT) features
- returns_company_year.csv   the target (beat_sp500) and prior-year return
- controls_company_year.csv  dividend payer, years since first SEC filing
                             (run build_controls.py first)
- company_info.csv           SIC code, exchange, HQ state (one row per company)
- abatements_company_year.csv  OPTIONAL: joined automatically once it exists

Output: data/modeling_table.csv

Every feature is known by the 10-K filing date ('filed'), and the return
window starts the first trading day after it, so there is no look-ahead.

Usage
    python build_modeling_table.py
"""
import argparse

import numpy as np
import pandas as pd

import config

TRAIN_YEARS = range(2012, 2021)   # 2012-2020
TEST_YEARS = range(2021, 2025)    # 2021-2024

ETR_COLUMNS = ["etr", "cash_etr"]
TREE_FILL = -1  # sentinel for blank ETRs in RF / AdaBoost / XGBoost

# Ratios that get winsorized at the 1st / 99th percentile of the training years.
WINSORIZE = [
    "cash_to_book_tax", "deferred_share_of_tax", "etr_change",
    "net_deferred_tax_to_assets", "valuation_allowance_to_assets",
    "unrecognized_tax_benefits_to_assets", "roa", "pretax_roa",
    "prior_ret_12m", "filing_lag_days", "tenk_filing_lag_days", "nt_max_days_late",
]

# Fama-French 12 industries from 4-digit SIC codes (Ken French's definitions).
FF12 = {
    "NoDur": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    "Durbl": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
              (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    "Manuf": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
              (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
              (3830, 3839), (3860, 3899)],
    "Enrgy": [(1200, 1399), (2900, 2999)],
    "Chems": [(2800, 2829), (2840, 2899)],
    "BusEq": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    "Telcm": [(4800, 4899)],
    "Utils": [(4900, 4949)],
    "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
    "Hlth":  [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    "Money": [(6000, 6999)],
}

# US Census regions; anything else (Canadian provinces, other countries, PR) is "Foreign".
CENSUS_REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_TO_REGION = {s: region for region, states in CENSUS_REGIONS.items() for s in states.split()}

BLANK_CHECK_SIC = 6770  # SPACs: cash shells whose price sits near $10, not operating companies


def ff12_industry(sic):
    if pd.isna(sic):
        return "Unknown"
    sic = int(sic)
    for name, ranges in FF12.items():
        if any(lo <= sic <= hi for lo, hi in ranges):
            return name
    return "Other"


def primary_exchange(exchanges):
    """First listed exchange, grouped into NYSE / Nasdaq / OTC / Other."""
    if pd.isna(exchanges) or not exchanges:
        return "Other"
    first = exchanges.split(",")[0]
    return first if first in {"NYSE", "Nasdaq", "OTC"} else "Other"


def load_inputs():
    d = config.DATA_DIR
    tax = pd.read_csv(d / "tax_company_year.csv", parse_dates=["period_end", "filed"])
    filing = pd.read_csv(d / "filing_company_year.csv")
    returns = pd.read_csv(d / "returns_company_year.csv")
    controls = pd.read_csv(d / "controls_company_year.csv")
    info = pd.read_csv(d / "company_info.csv",
                       usecols=["cik", "name", "sic", "exchanges", "hq_state"])
    return tax, filing, returns, controls, info


def add_abatements(df):
    """Join the abatement features on cik + fiscal_year once that file exists.

    Expected file: data/abatements_company_year.csv with cik, fiscal_year and
    feature columns (for example abatement_any, abatement_value_to_revenue,
    abatement_deals). Company-years with no match get 0 (no known abatement).
    """
    path = config.DATA_DIR / "abatements_company_year.csv"
    if not path.exists():
        print("No abatements file yet; skipping (re-run this script once it exists).")
        return df
    abatements = pd.read_csv(path)
    features = [c for c in abatements.columns if c not in ("cik", "fiscal_year")]
    df = df.merge(abatements, on=["cik", "fiscal_year"], how="left")
    df[features] = df[features].fillna(0)
    print(f"Joined abatements: {features}")
    return df


def build():
    tax, filing, returns, controls, info = load_inputs()
    tax = tax.drop(columns=["ticker"])
    filing = filing.drop(columns=["tenk_period_end", "tenk_filed"])
    returns = returns[["cik", "fiscal_year", "symbol", "ret_start", "ret_end", "ret_12m",
                       "sp500_ret_12m", "excess_ret_12m", "beat_sp500", "prior_ret_12m",
                       "suspect_price", "delisted_in_window"]]

    df = (tax.merge(returns, on=["cik", "fiscal_year"], how="left")
             .merge(filing, on=["cik", "fiscal_year"], how="left")
             .merge(controls, on=["cik", "fiscal_year"], how="left")
             .merge(info, on="cik", how="left"))
    df = add_abatements(df)
    print(f"Joined: {len(df):,} company-years")

    # Late-filing counts: a company-year with no row in the filing file had no NT notice.
    df["filing_row_missing"] = df["nt_count"].isna().astype(int)
    for col in ["nt_10k", "nt_10q_count", "nt_count", "nt_never_filed", "nt_missed_extension"]:
        df[col] = df[col].fillna(0).astype(int)
    df["nt_max_days_late"] = df["nt_max_days_late"].where(df["nt_count"] > 0, 0)

    # ---------------------------------------------------------- controls
    df["ff12_industry"] = df["sic"].map(ff12_industry)
    df["exchange"] = df["exchanges"].map(primary_exchange)
    df["hq_region"] = df["hq_state"].map(STATE_TO_REGION).fillna("Foreign")
    df.loc[df["hq_state"].isna(), "hq_region"] = "Unknown"
    df["fye_month"] = df["period_end"].dt.month
    df["fye_december"] = (df["fye_month"] == 12).astype(int)
    df["log_total_assets"] = np.log(df["total_assets"].where(df["total_assets"] > 0))
    # Size: quintile of total assets within each fiscal year (1 = smallest, 5 = largest).
    df["size_quintile"] = (df.groupby("fiscal_year")["log_total_assets"]
                             .transform(lambda s: pd.qcut(s, 5, labels=False)) + 1).astype("Int64")

    # ---------------------------------------------------------- filters
    steps = [("no target (return window not over, or no prices)", df["beat_sp500"].isna()),
             ("broken prices (suspect_price = 1)", df["suspect_price"] == 1),
             ("blank-check companies / SPACs (SIC 6770)", df["sic"] == BLANK_CHECK_SIC),
             ("outside 2012-2024", ~df["fiscal_year"].isin(list(TRAIN_YEARS) + list(TEST_YEARS)))]
    for label, mask in steps:
        mask = mask.reindex(df.index)
        print(f"  dropping {int(mask.sum()):,} rows: {label}")
        df = df[~mask]
    df = df.copy()
    df["beat_sp500"] = df["beat_sp500"].astype(int)
    df["split"] = np.where(df["fiscal_year"].isin(TRAIN_YEARS), "train", "test")
    train = df["split"] == "train"

    # ---------------------------------------------- winsorize (train cutoffs)
    for col in WINSORIZE:
        lo, hi = df.loc[train, col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)
    df["log_total_assets"] = df["log_total_assets"].clip(*df.loc[train, "log_total_assets"].quantile([0.01, 0.99]))

    # ------------------------------------------ blank ETRs (Rosana's choice)
    # Trees: -1 marks loss / tiny-profit years as their own group.
    # Logit: training-year median, with pretax_loss / pretax_tiny as flags.
    df["pretax_loss"] = df["pretax_loss"].fillna(0).astype(int)
    df["pretax_missing"] = df["pretax_income"].isna().astype(int)
    for col in ETR_COLUMNS:
        df[f"{col}_tree"] = df[col].fillna(TREE_FILL)
        df[f"{col}_logit"] = df[col].fillna(df.loc[train, col].median())

    return df


ID_COLUMNS = ["cik", "name", "symbol", "fiscal_year", "split", "period_end", "filed",
              "ret_start", "ret_end", "sic"]
TARGET_COLUMNS = ["beat_sp500", "ret_12m", "sp500_ret_12m", "excess_ret_12m"]
FEATURE_COLUMNS = [
    # tax
    "etr", "cash_etr", "etr_tree", "cash_etr_tree", "etr_logit", "cash_etr_logit",
    "etr_usable", "pretax_loss", "pretax_tiny", "pretax_missing", "etr_change",
    "cash_to_book_tax", "deferred_share_of_tax", "net_deferred_tax_to_assets",
    "valuation_allowance_to_assets", "unrecognized_tax_benefits_to_assets",
    # filing behavior
    "nt_10k", "nt_10q_count", "nt_count", "nt_max_days_late", "nt_never_filed",
    "nt_missed_extension", "tenk_filing_lag_days", "filing_lag_days", "filing_row_missing",
    # controls
    "size_quintile", "log_total_assets", "filer_status_est", "ff12_industry", "exchange",
    "hq_region", "fye_month", "fye_december", "dividend_payer", "years_since_first_filing",
    "prior_ret_12m", "roa", "pretax_roa",
    # data-quality flags
    "from_comparative", "delisted_in_window",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(config.DATA_DIR / "modeling_table.csv"))
    args = parser.parse_args()

    df = build()
    abatement_cols = [c for c in df.columns if c.startswith("abatement")]
    df = df[ID_COLUMNS + TARGET_COLUMNS + FEATURE_COLUMNS + abatement_cols]
    df.to_csv(args.out, index=False)

    print(f"\nSaved {len(df):,} company-years for {df['cik'].nunique():,} companies to {args.out}")
    for split, part in df.groupby("split", sort=False):
        print(f"  {split}: {len(part):,} rows, {part['beat_sp500'].mean():.1%} beat the S&P 500")
    blanks = df[FEATURE_COLUMNS].isna().mean()
    blanks = blanks[blanks > 0].sort_values(ascending=False)
    print("\nShare of blanks left in features (fill these per model):")
    print(blanks.map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
