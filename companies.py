"""Which companies to pull, plus per-company filing history from EDGAR.

Two ways to choose the company universe:
- "current" (default): every company on the SEC's current ticker list.
  Quick, but it only has companies listed TODAY, so it carries
  survivorship bias (companies that were delisted or went bust are missing).
- "full-index": every company that filed a 10-K in any year from
  START_YEAR on, taken from EDGAR's quarterly filing index. This includes
  dead companies, but also some non-stocks (funds, trusts) that you will
  drop later when they have no price data.
"""
import re

import numpy as np
import pandas as pd

import config
import sec_client

ANNUAL_REPORT_FORMS = ["10-K", "10-K405", "10-KT"]


def fiscal_year(period_end):
    """Fiscal year label for a period-end date (works on a Series or one date).

    Same rule as Compustat: a year ending June-December belongs to that
    calendar year; a year ending January-May belongs to the year before.
    So a fiscal year ending 2021-01-31 (a retailer) is fiscal 2020.
    """
    period_end = pd.to_datetime(period_end)
    if isinstance(period_end, pd.Series):
        return np.where(period_end.dt.month >= 6, period_end.dt.year, period_end.dt.year - 1)
    return period_end.year if period_end.month >= 6 else period_end.year - 1


# ---------------------------------------------------------------- universe

def current_listed_companies(one_row_per_company=True):
    """CIK, name, ticker and exchange for every company on the SEC ticker list."""
    data = sec_client.get(
        "https://www.sec.gov/files/company_tickers_exchange.json",
        "company_tickers_exchange.json",
    )
    df = pd.DataFrame(data["data"], columns=data["fields"])
    # A company with several share classes appears once per ticker; keep one row per CIK.
    if one_row_per_company:
        df = df.drop_duplicates("cik").reset_index(drop=True)
    return df[["cik", "name", "ticker", "exchange"]]


def _parse_form_idx(text):
    """Turn one EDGAR form.idx file (fixed-width text) into a list of dicts."""
    lines = text.splitlines()
    first_row = next(i for i, line in enumerate(lines) if line.startswith("---")) + 1
    rows = []
    for line in lines[first_row:]:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 5:
            continue
        rows.append({
            "form": parts[0],
            "name": " ".join(parts[1:-3]),
            "cik": int(parts[-3]),
            "date_filed": parts[-2],
        })
    return rows


def all_10k_filers(start_year=config.START_YEAR, end_year=config.END_YEAR + 1):
    """Every CIK that filed a 10-K between start_year and end_year (incl. dead companies)."""
    rows = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            text = sec_client.get(
                f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx",
                f"full-index/{year}_QTR{quarter}_form.idx",
                as_json=False,
            )
            if text:  # future quarters do not exist yet
                rows += [r for r in _parse_form_idx(text) if r["form"] in ANNUAL_REPORT_FORMS]
    df = pd.DataFrame(rows)
    df = df.sort_values("date_filed").groupby("cik", as_index=False).last()
    df["ticker"] = pd.NA
    df["exchange"] = pd.NA
    return df[["cik", "name", "ticker", "exchange"]]


def add_universe_arguments(parser):
    """Command-line options shared by the pull scripts."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", nargs="+", help="only these tickers, e.g. --tickers AAPL MSFT")
    group.add_argument("--limit", type=int, help="the first N companies of the universe (a test run)")
    group.add_argument("--all", action="store_true", help="every company in the universe")
    parser.add_argument("--universe", choices=["current", "full-index"], default="current",
                        help="current = today's listed companies (default); "
                             "full-index = every 10-K filer since START_YEAR")


def pick_companies(args):
    """Return the DataFrame of companies (cik, name, ticker, exchange) the user asked for."""
    if args.tickers:
        listed = current_listed_companies(one_row_per_company=False)
        wanted = [t.upper() for t in args.tickers]
        picked = listed[listed["ticker"].isin(wanted)]
        missing = set(wanted) - set(picked["ticker"])
        if missing:
            print(f"Not on the SEC ticker list (skipped): {sorted(missing)}")
        return picked.drop_duplicates("cik")
    companies = current_listed_companies() if args.universe == "current" else all_10k_filers()
    return companies.head(args.limit) if args.limit else companies


# ------------------------------------------------------ filing history

def load_submissions(cik):
    """A company's EDGAR profile and its full list of filings back to START_YEAR.

    Returns (profile_dict, filings_dataframe), or (None, None) if EDGAR has nothing.
    """
    profile = sec_client.get(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
        f"submissions/CIK{cik:010d}.json",
    )
    if profile is None:
        return None, None

    # "recent" holds the latest ~1,000 filings; older ones are in extra pages.
    pages = [pd.DataFrame(profile["filings"]["recent"])]
    for extra in profile["filings"].get("files", []):
        if int(extra["filingTo"][:4]) >= config.START_YEAR:
            more = sec_client.get(f"https://data.sec.gov/submissions/{extra['name']}",
                                  f"submissions/{extra['name']}")
            if more:
                pages.append(pd.DataFrame(more))

    filings = pd.concat(pages, ignore_index=True)
    filings = filings[["accessionNumber", "form", "filingDate", "reportDate"]].copy()
    filings["filingDate"] = pd.to_datetime(filings["filingDate"])
    filings["reportDate"] = pd.to_datetime(filings["reportDate"].replace("", None))
    return profile, filings


def company_profile_row(cik, profile):
    """Descriptive fields (controls) from the EDGAR profile, as one dict.

    Note: 'filer_category' is the company's status TODAY, not in each past year.
    """
    business = (profile.get("addresses") or {}).get("business") or {}
    return {
        "cik": cik,
        "name": profile.get("name"),
        "tickers": ",".join(t for t in profile.get("tickers", []) if t),
        "exchanges": ",".join(e for e in profile.get("exchanges", []) if e),
        "sic": profile.get("sic"),
        "sic_description": profile.get("sicDescription"),
        "state_of_incorporation": profile.get("stateOfIncorporation"),
        "hq_state": business.get("stateOrCountry"),
        "fiscal_year_end_mmdd": profile.get("fiscalYearEnd"),
        "filer_category": profile.get("category"),
        "entity_type": profile.get("entityType"),
    }
