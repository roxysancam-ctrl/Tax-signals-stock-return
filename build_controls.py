"""Control features for each company-year, from the cached SEC downloads.

Output: data/controls_company_year.csv, one row per company per fiscal year
(the same rows as tax_company_year.csv), with:
- dividend_payer: 1 if the 10-K for that year reports common dividends > 0
- first_sec_filing / years_since_first_filing: how long the company has
  filed with the SEC, a stand-in for "years public"

The other controls (sector, exchange, HQ region, size, fiscal year-end month,
filer status, profitability, prior-year return) come from files we already
have and are added in build_modeling_table.py.

This script only reads cache/ (it never downloads). Reading thousands of
small files from the shared project folder is very slow, so copy
cache/companyfacts and cache/submissions to a local disk first and point
--cache-dir at the copy.

Usage
    python build_controls.py --cache-dir /home/user/cache
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Common-stock dividends, in dollars or dollars per share. A company-year
# counts as a dividend payer if any of these is above zero.
DIVIDEND_TAGS = {
    "USD": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends",
            "DividendsCommonStockCash", "DividendsCommonStock", "DividendsCash", "Dividends",
            "PaymentsOfOrdinaryDividends"],
    "USD/shares": ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"],
}
ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}


def _read(path):
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def dividend_years(facts):
    """Fiscal-year period ends (as Timestamps) where the first 10-K that reported
    that year shows common dividends above zero."""
    pieces = []
    for unit, tags in DIVIDEND_TAGS.items():
        for tag in tags:
            try:
                rows = facts["facts"]["us-gaap"][tag]["units"][unit]
            except KeyError:
                continue
            df = pd.DataFrame(rows)
            if "start" not in df.columns:
                continue
            df = df[df["form"].isin(ANNUAL_FORMS)]
            days = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days
            pieces.append(df.loc[days.between(350, 380), ["end", "val", "filed"]])
    if not pieces:
        return set()
    df = pd.concat(pieces)
    # As first reported: keep only facts from the earliest 10-K for each period.
    df["first_filed"] = df.groupby("end")["filed"].transform("min")
    df = df[df["filed"] == df["first_filed"]]
    paid = df.groupby("end")["val"].max() > 0
    return set(pd.to_datetime(paid[paid].index))


def first_filing_date(profile):
    """Earliest filing on EDGAR for this company (EDGAR itself starts around 1993-1996)."""
    dates = list(profile["filings"]["recent"].get("filingDate", []))
    dates += [f["filingFrom"] for f in profile["filings"].get("files", [])]
    return min(dates) if dates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=str(config.CACHE_DIR),
                        help="folder holding companyfacts/ and submissions/ (a local copy is much faster)")
    parser.add_argument("--out", default=str(config.DATA_DIR / "controls_company_year.csv"))
    args = parser.parse_args()
    cache = Path(args.cache_dir)

    tax = pd.read_csv(config.DATA_DIR / "tax_company_year.csv",
                      usecols=["cik", "fiscal_year", "period_end", "filed"],
                      parse_dates=["period_end", "filed"])
    ciks = tax["cik"].unique()
    print(f"Building controls for {len(ciks):,} companies")

    payer_rows, first_filed, missing_facts, missing_subs = [], {}, 0, 0
    for i, cik in enumerate(ciks, start=1):
        facts = _read(cache / "companyfacts" / f"CIK{cik:010d}.json.gz")
        if facts:
            for end in dividend_years(facts):
                payer_rows.append((cik, end))
        else:
            missing_facts += 1
        profile = _read(cache / "submissions" / f"CIK{cik:010d}.json.gz")
        if profile and "filings" in profile:
            first_filed[cik] = first_filing_date(profile)
        else:
            missing_subs += 1
        if i % 500 == 0 or i == len(ciks):
            print(f"  {i}/{len(ciks)} done", flush=True)

    payers = pd.DataFrame(payer_rows, columns=["cik", "period_end"]).assign(dividend_payer=1)
    out = tax.merge(payers, on=["cik", "period_end"], how="left")
    out["dividend_payer"] = out["dividend_payer"].fillna(0).astype(int)
    out["first_sec_filing"] = pd.to_datetime(out["cik"].map(first_filed))
    out["years_since_first_filing"] = ((out["filed"] - out["first_sec_filing"]).dt.days / 365.25).round(2)
    out = out[["cik", "fiscal_year", "dividend_payer", "first_sec_filing", "years_since_first_filing"]]

    out.to_csv(args.out, index=False)
    print(f"Saved {len(out):,} rows to {args.out}")
    print(f"Dividend payers: {out['dividend_payer'].mean():.1%} of company-years")
    if missing_facts or missing_subs:
        print(f"Not in cache: {missing_facts} companyfacts files, {missing_subs} submissions files")


if __name__ == "__main__":
    main()
