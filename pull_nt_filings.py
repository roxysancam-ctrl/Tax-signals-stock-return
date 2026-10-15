"""Pull late-filing notices (NT 10-K / NT 10-Q) and 10-K filing dates from EDGAR.

A company that cannot file its 10-K or 10-Q on time must file an
"NT" notice (Form 12b-25) by the next business day. The notice gives an
extra 15 days for a 10-K and 5 days for a 10-Q.

Outputs
- data/nt_filings.csv           one row per NT notice, matched to the report it delayed
- data/filing_company_year.csv  one row per company per fiscal year (model features)
- data/company_info.csv         descriptive controls: SIC, state, exchange, filer category

A fiscal year's filing features are all known by the time that year's
10-K is filed ('tenk_filed'), so use that date as the "as of" date.

Usage
    python pull_nt_filings.py --tickers AAPL MSFT TSLA
    python pull_nt_filings.py --limit 50
    python pull_nt_filings.py --all
"""
import argparse

import numpy as np
import pandas as pd

import config
from companies import add_universe_arguments, fiscal_year, load_submissions, company_profile_row, pick_companies

# NT form -> the kind of report it delays
NT_FORMS = {"NT 10-K": "10-K", "NT 10-K/A": "10-K", "NT 10-Q": "10-Q", "NT 10-Q/A": "10-Q"}
# Original (not amended) reports that an NT notice can be matched to
REPORT_FORMS = {"10-K": ["10-K", "10-K405", "10-KT"], "10-Q": ["10-Q"]}
EXTENSION_DAYS = {"10-K": 15, "10-Q": 5}
# If EDGAR has no period date for an NT notice and the report never came,
# guess the period end as this many days before the notice.
TYPICAL_DEADLINE_DAYS = {"10-K": 90, "10-Q": 45}


def match_nt_to_reports(filings):
    """For each NT notice, find the report it delayed and how late that report was."""
    nts = filings[filings["form"].isin(NT_FORMS)].copy()
    rows = []
    for nt in nts.itertuples(index=False):
        kind = NT_FORMS[nt.form]
        reports = filings[filings["form"].isin(REPORT_FORMS[kind])
                          & (filings["filingDate"] >= nt.filingDate - pd.Timedelta(days=5))]
        if pd.notna(nt.reportDate):
            # Best case: EDGAR tells us which period the notice was for.
            match = reports[reports["reportDate"] == nt.reportDate]
        else:
            # Otherwise take the first report of that kind filed after the notice.
            match = reports[(reports["filingDate"] >= nt.filingDate)
                            & (reports["filingDate"] <= nt.filingDate + pd.Timedelta(days=365))]
        match = match.sort_values("filingDate").head(1)

        period_end, period_estimated = nt.reportDate, 0
        if pd.isna(period_end) and not match.empty:
            period_end = match["reportDate"].iloc[0]
        if pd.isna(period_end):
            period_end = nt.filingDate - pd.Timedelta(days=TYPICAL_DEADLINE_DAYS[kind])
            period_estimated = 1

        report_filed = match["filingDate"].iloc[0] if not match.empty else pd.NaT
        # The notice is due one business day after the deadline, so the gap
        # between notice and report is roughly the number of days late.
        days_late = (report_filed - nt.filingDate).days if pd.notna(report_filed) else np.nan
        # A few companies file the report a day or so before the notice; count that as 0.
        days_late = max(days_late, 0) if pd.notna(days_late) else np.nan
        rows.append({
            "nt_form": nt.form,
            "report_kind": kind,
            "nt_filed": nt.filingDate,
            "period_end": period_end,
            "period_estimated": period_estimated,
            "report_filed": report_filed,
            "days_late": days_late,
            "never_filed": int(pd.isna(report_filed)),
            "missed_extension": int(pd.isna(report_filed) or days_late > EXTENSION_DAYS[kind]),
        })
    return pd.DataFrame(rows)


def annual_report_dates(filings):
    """Each original 10-K: its period end, filing date and filing lag."""
    tenk = filings[filings["form"].isin(REPORT_FORMS["10-K"]) & filings["reportDate"].notna()].copy()
    tenk = tenk.sort_values("filingDate").drop_duplicates("reportDate")
    tenk["fiscal_year"] = fiscal_year(tenk["reportDate"])
    tenk = tenk.drop_duplicates("fiscal_year", keep="last")
    return pd.DataFrame({
        "fiscal_year": tenk["fiscal_year"],
        "tenk_period_end": tenk["reportDate"],
        "tenk_filed": tenk["filingDate"],
        "tenk_filing_lag_days": (tenk["filingDate"] - tenk["reportDate"]).dt.days,
    })


def fiscal_year_of_quarter(quarter_end, tenk_period_ends, fye_mmdd):
    """Fiscal year a 10-Q quarter belongs to: that of the next 10-K period end."""
    later = tenk_period_ends[(tenk_period_ends >= quarter_end)
                             & (tenk_period_ends <= quarter_end + pd.Timedelta(days=366))]
    if len(later):
        return fiscal_year(later.min())
    # No 10-K followed (e.g. the company stopped filing): use its stated year-end month.
    fye_month = int(fye_mmdd[:2]) if fye_mmdd else 12
    year = quarter_end.year if quarter_end.month <= fye_month else quarter_end.year + 1
    return fiscal_year(pd.Timestamp(year, fye_month, 1))


def company_year_features(nt, tenk, fye_mmdd):
    """Combine NT notices and 10-K dates into one row per fiscal year."""
    if not nt.empty:
        nt = nt.copy()
        nt["fiscal_year"] = [
            fiscal_year(p) if kind == "10-K" else fiscal_year_of_quarter(p, tenk["tenk_period_end"], fye_mmdd)
            for p, kind in zip(nt["period_end"], nt["report_kind"])
        ]
        per_year = nt.groupby("fiscal_year").agg(
            nt_10k=("report_kind", lambda k: int((k == "10-K").any())),
            nt_10q_count=("report_kind", lambda k: int((k == "10-Q").sum())),
            nt_count=("report_kind", "size"),
            nt_max_days_late=("days_late", "max"),
            nt_never_filed=("never_filed", "max"),
            nt_missed_extension=("missed_extension", "max"),
        ).reset_index()
        years = tenk.merge(per_year, on="fiscal_year", how="outer")
    else:
        years = tenk.copy()
    for col in ["nt_10k", "nt_10q_count", "nt_count", "nt_never_filed", "nt_missed_extension"]:
        years[col] = years[col].fillna(0).astype(int) if col in years else 0
    if "nt_max_days_late" not in years:
        years["nt_max_days_late"] = np.nan
    return years.sort_values("fiscal_year")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_universe_arguments(parser)
    args = parser.parse_args()

    companies = pick_companies(args)
    print(f"Pulling filing histories for {len(companies)} companies")

    profiles, notices, years, failed = [], [], [], []
    for i, row in enumerate(companies.itertuples(index=False), start=1):
        try:
            profile, filings = load_submissions(row.cik)
            if profile is not None:
                profiles.append(company_profile_row(row.cik, profile))
                nt = match_nt_to_reports(filings)
                tenk = annual_report_dates(filings)
                year_rows = company_year_features(nt, tenk, profile.get("fiscalYearEnd"))
                year_rows.insert(0, "cik", row.cik)
                years.append(year_rows)
                if not nt.empty:
                    nt.insert(0, "cik", row.cik)
                    notices.append(nt)
        except Exception as err:  # one odd company should not stop a long run
            failed.append((row.cik, repr(err)))
        if i % 100 == 0 or i == len(companies):
            print(f"  {i}/{len(companies)} done")

    config.DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame(profiles).to_csv(config.DATA_DIR / "company_info.csv", index=False)

    nt_all = pd.concat(notices, ignore_index=True) if notices else pd.DataFrame()
    nt_all.to_csv(config.DATA_DIR / "nt_filings.csv", index=False)

    year_all = pd.concat(years, ignore_index=True)
    year_all = year_all[year_all["fiscal_year"].between(config.START_YEAR, config.END_YEAR)]
    year_all.to_csv(config.DATA_DIR / "filing_company_year.csv", index=False)

    print(f"Saved {len(profiles):,} company profiles, {len(nt_all):,} NT notices, "
          f"{len(year_all):,} company-years to {config.DATA_DIR}")
    if failed:
        print(f"{len(failed)} companies failed, e.g. {failed[:3]}")


if __name__ == "__main__":
    main()
