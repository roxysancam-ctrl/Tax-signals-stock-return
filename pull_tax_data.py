"""Pull tax figures for each company-year from SEC EDGAR (XBRL "company facts").

Output: data/tax_company_year.csv, one row per company per fiscal year.

How each number is chosen
- Only annual reports (10-K) are used, and only full-year periods (350-380 days).
- A 10-K also repeats the previous two years as comparatives, and later
  10-Ks sometimes restate old numbers. We keep the value from the FIRST
  10-K that reported each year, because that is what investors actually
  saw at the time. 'filed' is that 10-K's filing date: treat it as the
  date the row becomes known (never the fiscal year-end).
- Companies use different XBRL tags for the same idea, so each field has
  a list of tags, tried in order.

Usage
    python pull_tax_data.py --tickers AAPL MSFT TSLA     # a few companies
    python pull_tax_data.py --limit 50                   # quick test
    python pull_tax_data.py --all                        # everything (can take an hour)
"""
import argparse

import pandas as pd

import config
import sec_client
from companies import add_universe_arguments, fiscal_year, pick_companies

# Values that cover a whole year (income statement / cash flow).
# For each output column: XBRL tags to try, best first.
DURATION_TAGS = {
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "current_tax_expense": ["CurrentIncomeTaxExpenseBenefit"],
    "deferred_tax_expense": ["DeferredIncomeTaxExpenseBenefit", "DeferredIncomeTaxesAndTaxCredits"],
    "cash_taxes_paid": ["IncomeTaxesPaidNet", "IncomeTaxesPaid"],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
}

# Some companies (Apple, for example) only tag the federal / state / foreign
# parts of current and deferred tax (or the domestic / foreign parts of pretax
# income), never the total. Add the parts up then.
TAX_COMPONENTS = {
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign"],
    "current_tax_expense": ["CurrentFederalTaxExpenseBenefit",
                            "CurrentStateAndLocalTaxExpenseBenefit",
                            "CurrentForeignTaxExpenseBenefit"],
    "deferred_tax_expense": ["DeferredFederalIncomeTaxExpenseBenefit",
                             "DeferredStateAndLocalIncomeTaxExpenseBenefit",
                             "DeferredForeignIncomeTaxExpenseBenefit"],
}

# Values measured on one date (balance sheet), matched to the fiscal year-end.
INSTANT_TAGS = {
    "total_assets": ["Assets"],
    "net_deferred_tax_reported": ["DeferredTaxAssetsLiabilitiesNet"],
    "deferred_tax_assets": ["DeferredTaxAssetsNet"],
    # "DeferredIncomeTaxLiabilities" is the newer (2018+) name for the same total.
    "deferred_tax_liabilities": ["DeferredTaxLiabilities", "DeferredIncomeTaxLiabilities"],
    "valuation_allowance": ["DeferredTaxAssetsValuationAllowance"],
    "unrecognized_tax_benefits": ["UnrecognizedTaxBenefits"],
}

# These define which fiscal years a company has (a year needs at least one).
ANCHOR_COLUMNS = ["pretax_income", "tax_expense", "net_income", "revenue"]

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}


def _tag_facts(facts, tag, taxonomy="us-gaap", unit="USD"):
    """All 10-K facts for one tag as a DataFrame (empty if the company never used it)."""
    try:
        rows = facts["facts"][taxonomy][tag]["units"][unit]
    except KeyError:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df[df["form"].isin(ANNUAL_FORMS)].copy()


def first_reported_values(facts, tags, full_year):
    """One value per period-end date, as first reported, trying tags in order.

    full_year=True keeps 12-month periods; False keeps point-in-time values.
    Returns columns: end, val, filed, accn.
    """
    pieces = []
    for priority, tag in enumerate(tags):
        df = _tag_facts(facts, tag)
        if df.empty:
            continue
        has_start = "start" in df.columns
        if full_year:
            if not has_start:
                continue
            days = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days
            df = df[days.between(350, 380)]  # 52/53-week years are 364 or 371 days
        elif has_start:
            df = df[df["start"].isna()]
        df["priority"] = priority
        pieces.append(df)
    if not pieces:
        return pd.DataFrame({"end": pd.to_datetime([]), "val": pd.Series(dtype=float),
                             "filed": pd.to_datetime([]), "accn": pd.Series(dtype=str)})
    df = pd.concat(pieces, ignore_index=True)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    # Earliest filing wins; within the same filing, the preferred tag wins.
    df = df.sort_values(["end", "filed", "priority"]).drop_duplicates("end")
    return df[["end", "val", "filed", "accn"]]


def public_float_by_filing(facts):
    """Public float (market value held by outsiders) reported on each 10-K, keyed by accession."""
    df = _tag_facts(facts, "EntityPublicFloat", taxonomy="dei")
    if df.empty:
        return pd.Series(dtype=float)
    return df.drop_duplicates("accn").set_index("accn")["val"]


def company_years(facts):
    """Raw tax figures for one company: one row per fiscal year."""
    duration = {col: first_reported_values(facts, tags, full_year=True)
                for col, tags in DURATION_TAGS.items()}

    # The fiscal years this company has, and the 10-K that first reported each.
    anchors = pd.concat([duration[c] for c in ANCHOR_COLUMNS], ignore_index=True)
    if anchors.empty:
        return pd.DataFrame()
    years = (anchors.sort_values(["end", "filed"])
                    .drop_duplicates("end")[["end", "filed", "accn"]]
                    .rename(columns={"end": "period_end"}))

    for col, values in duration.items():
        years = years.merge(values[["end", "val"]].rename(columns={"end": "period_end", "val": col}),
                            on="period_end", how="left")
    for col, parts in TAX_COMPONENTS.items():
        summed = None
        for tag in parts:
            v = first_reported_values(facts, [tag], full_year=True).set_index("end")["val"]
            summed = v if summed is None else summed.add(v, fill_value=0)
        years[col] = years[col].fillna(years["period_end"].map(summed))
    # Last resort for pretax income: after-tax income from continuing operations + tax.
    after_tax = first_reported_values(facts, ["IncomeLossFromContinuingOperations"], full_year=True)
    years["pretax_income"] = years["pretax_income"].fillna(
        years["period_end"].map(after_tax.set_index("end")["val"]) + years["tax_expense"])
    # Still missing one half? Total tax = current + deferred.
    years["current_tax_expense"] = years["current_tax_expense"].fillna(
        years["tax_expense"] - years["deferred_tax_expense"])
    years["deferred_tax_expense"] = years["deferred_tax_expense"].fillna(
        years["tax_expense"] - years["current_tax_expense"])

    for col, tags in INSTANT_TAGS.items():
        values = first_reported_values(facts, tags, full_year=False)
        years = years.merge(values[["end", "val"]].rename(columns={"end": "period_end", "val": col}),
                            on="period_end", how="left")

    years["public_float"] = years["accn"].map(public_float_by_filing(facts))
    years["fiscal_year"] = fiscal_year(years["period_end"])
    # If a company changed its year-end, two periods can land in one fiscal year: keep the later.
    years = years.sort_values("period_end").drop_duplicates("fiscal_year", keep="last")
    return years


def add_tax_features(df):
    """Model features built from the raw figures. Ratios are left unwinsorized."""
    df = df.sort_values(["cik", "fiscal_year"]).copy()
    pretax = df["pretax_income"]

    # Rows whose 'first report' is a comparative in a much later 10-K
    # (usually the company's first XBRL year). The values are fine, but
    # 'filed' is later than when the market really knew them.
    df["filing_lag_days"] = (df["filed"] - df["period_end"]).dt.days
    df["from_comparative"] = (df["filing_lag_days"] > 400).astype(int)

    # Tax rates only mean something when pretax income is positive and not tiny.
    tiny = pretax.abs() < config.TINY_PRETAX_SHARE_OF_ASSETS * df["total_assets"]
    usable = (pretax > 0) & ~tiny
    df["pretax_loss"] = (pretax <= 0).astype("Int64").where(pretax.notna())
    df["pretax_tiny"] = tiny.astype(int)
    df["etr_usable"] = usable.astype(int)

    # Effective tax rate = tax expense / pretax income, bounded to [0, 1]
    # (the usual convention in tax-avoidance research).
    df["etr"] = (df["tax_expense"] / pretax).where(usable).clip(0, 1)
    df["cash_etr"] = (df["cash_taxes_paid"] / pretax).where(usable).clip(0, 1)
    # Cash taxes paid vs. tax expense: below 1 means the company pays less
    # cash than its income statement shows (taxes deferred to later).
    df["cash_to_book_tax"] = (df["cash_taxes_paid"] / df["tax_expense"]).where(df["tax_expense"] > 0)
    df["deferred_share_of_tax"] = (df["deferred_tax_expense"] / df["tax_expense"]).where(df["tax_expense"] > 0)

    # Year-over-year change in ETR, only between back-to-back fiscal years.
    prev = df.groupby("cik")[["fiscal_year", "etr"]].shift(1)
    back_to_back = prev["fiscal_year"] == df["fiscal_year"] - 1
    df["etr_change"] = (df["etr"] - prev["etr"]).where(back_to_back)

    # Deferred tax balance: use the reported net figure, else assets minus liabilities.
    df["net_deferred_tax"] = df["net_deferred_tax_reported"].fillna(
        df["deferred_tax_assets"] - df["deferred_tax_liabilities"])
    assets = df["total_assets"].where(df["total_assets"] > 0)
    df["net_deferred_tax_to_assets"] = df["net_deferred_tax"] / assets
    df["valuation_allowance_to_assets"] = df["valuation_allowance"] / assets
    df["unrecognized_tax_benefits_to_assets"] = df["unrecognized_tax_benefits"] / assets

    # Profitability controls (ROA also feeds the backup target).
    df["roa"] = df["net_income"] / assets
    df["pretax_roa"] = pretax / assets

    # Filer status estimated from public float (SEC thresholds):
    # 2 = large accelerated (>= $700M), 1 = accelerated ($75M-$700M), 0 = non-accelerated.
    # Approximate: since 2020 some small-revenue companies are non-accelerated regardless.
    float_ = df["public_float"]
    df["filer_status_est"] = pd.cut(float_, [-float("inf"), 75e6, 700e6, float("inf")],
                                    labels=[0, 1, 2], right=False).astype("Int64")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_universe_arguments(parser)
    parser.add_argument("--out", default=str(config.DATA_DIR / "tax_company_year.csv"))
    args = parser.parse_args()

    companies = pick_companies(args)
    print(f"Pulling XBRL tax data for {len(companies)} companies")

    frames, failed = [], []
    for i, row in enumerate(companies.itertuples(index=False), start=1):
        try:
            facts = sec_client.get(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{row.cik:010d}.json",
                f"companyfacts/CIK{row.cik:010d}.json",
            )
            if facts:
                years = company_years(facts)
                if not years.empty:
                    years.insert(0, "cik", row.cik)
                    years.insert(1, "ticker", row.ticker)
                    frames.append(years)
        except Exception as err:  # one odd company should not stop a long run
            failed.append((row.cik, repr(err)))
        if i % 100 == 0 or i == len(companies):
            print(f"  {i}/{len(companies)} done")

    if not frames:
        raise SystemExit("No data pulled.")
    df = add_tax_features(pd.concat(frames, ignore_index=True))
    df = df[df["fiscal_year"].between(config.START_YEAR, config.END_YEAR)]

    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} company-years for {df['cik'].nunique():,} companies to {args.out}")
    if failed:
        print(f"{len(failed)} companies failed, e.g. {failed[:3]}")


if __name__ == "__main__":
    main()
