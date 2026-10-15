"""Tax abatement features: did the company get a tax break, how big, how many deals?

Output: data/abatements_company_year.csv, one row per company per fiscal year
(the same rows as tax_company_year.csv; join on cik + fiscal_year), plus
data/abatement_name_matches.csv, every recipient name and the company it was
matched to, so the matches can be checked by eye.

Sources
- NY State IDA project data (free): https://data.ny.gov/d/9rtk-3fkw. One row
  per project per reporting year, with the tax exemptions it got that year.
- Good Jobs First Subsidy Tracker (optional): a CSV exported from
  https://subsidytracker.goodjobsfirst.org/ passed with --gjf-file. Its
  "Parent Company" column already rolls subsidiaries up to the parent.

Features
- abate_any         1 if the company had any abatement that year
- abate_value       total dollars of exemptions that year
- abate_to_revenue  abate_value / revenue (from tax_company_year.csv)
- abate_deals       number of distinct projects / awards that year
- abate_ny_covered  1 if the company is headquartered in NY. NY IDA data only
                    sees NY deals, so a 0 for an out-of-state company means
                    "unknown", not "none". Use this to check the skew.

No look-ahead: IDA reports for year Y are filed about 90 days after the year
ends and reach data.ny.gov later still, so by default the features for fiscal
year Y use abatements reported for year Y-1 (--lag-years 1). Most abatements
(PILOTs, multi-year exemptions) run for many years, so little is lost.

Name matching
Recipient names are cleaned (upper case, punctuation and suffixes such as INC,
CORP, LLC, HOLDINGS removed) and matched exactly to cleaned EDGAR names first.
Names left over are matched fuzzily (difflib, standard library) against
companies sharing the first word, accepted at a similarity of 0.95 or more.
Many IDA recipients are property LLCs ("123 Main Street Realty LLC") owned by
private developers, so expect a low match rate.

Usage
    python pull_abatements.py                         # NY IDA only
    python pull_abatements.py --gjf-file gjf.csv      # plus Good Jobs First
    python pull_abatements.py --lag-years 0           # same-year abatements
"""
import argparse
import difflib
import gzip
import json
import re
import time

import numpy as np
import pandas as pd
import requests

import config

NY_IDA_URL = "https://data.ny.gov/resource/9rtk-3fkw.json"
NY_IDA_CACHE = config.CACHE_DIR / "ny_ida_projects.json.gz"
PAGE_SIZE = 50000
FUZZY_CUTOFF = 0.95

# Large IDA recipients that are subsidiaries of a listed parent under a different
# name, found by reading the biggest unmatched recipients. Pattern -> parent cik.
ALIASES = {
    r"\bNBC\s*UNIVERSAL": 1166691,                  # Comcast
    r"METROPOLITAN LIFE INSURANCE": 1099219,         # MetLife
    r"GLOBAL\s*FOUNDRIES": 1709048,                  # GlobalFoundries
}

# Legal forms and tags: always removed.
LEGAL_WORDS = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LLC", "LLP",
    "LP", "LTD", "LIMITED", "PLC", "SA", "NV", "AG", "THE", "DE", "NY", "ADR",
    "PROJECT", "FACILITY", "EXPANSION", "II", "III", "IV",
}
# Generic words: removed only for the looser second pass, which needs two words
# left so that "Murphy Co" can't match "Murphy USA" or "Albany Enterprises"
# match "Albany International".
GENERIC_WORDS = {"HOLDINGS", "HOLDING", "GROUP", "TRUST", "USA", "US",
                 "INTERNATIONAL", "INTL", "ENTERPRISES", "COMPANIES"}


def clean_name(name, loose=False):
    """Upper case, drop /DE/-style tags, punctuation and legal suffixes."""
    if not isinstance(name, str):
        return ""
    s = name.upper()
    s = re.sub(r"\s/[A-Z]{2,4}/?", " ", s)    # DANAHER CORP /DE/, COSTCO /NEW
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    drop = LEGAL_WORDS | GENERIC_WORDS if loose else LEGAL_WORDS
    return " ".join(w for w in s.split() if w not in drop)


# ---------------------------------------------------------------- NY IDA

def download_ny_ida():
    """All rows of the NY IDA dataset, cached in cache/ after the first run."""
    if NY_IDA_CACHE.exists():
        with gzip.open(NY_IDA_CACHE, "rt") as f:
            return pd.DataFrame(json.load(f))
    rows, offset = [], 0
    while True:
        r = requests.get(NY_IDA_URL, params={"$limit": PAGE_SIZE, "$offset": offset,
                                             "$order": ":id"}, timeout=120)
        r.raise_for_status()
        page = r.json()
        rows += page
        print(f"  NY IDA rows downloaded: {len(rows):,}", flush=True)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(1)
    NY_IDA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(NY_IDA_CACHE, "wt") as f:
        json.dump(rows, f)
    return pd.DataFrame(rows)


def first_column(df, patterns, exclude=()):
    """The first column whose name matches one of the patterns, in order."""
    for p in patterns:
        for c in df.columns:
            if re.search(p, c) and not any(re.search(x, c) for x in exclude):
                return c
    return None


def tidy_ny_ida(raw):
    """One row per recipient name, project and year, with the exemption dollars."""
    name_col = first_column(raw, [r"^applicant_name$", r"recipient", r"applicant",
                                  r"^project_name$", r"company"])
    year_col = first_column(raw, [r"fiscal_year_end", r"fiscal_year", r"^year"])
    value_col = first_column(raw, [r"^total_exemptions$", r"total_exemption"],
                             exclude=[r"net", r"pilot"])
    project_col = first_column(raw, [r"^project_code$", r"project_code", r"project_id"])
    missing = [n for n, c in [("name", name_col), ("year", year_col),
                              ("exemption value", value_col)] if c is None]
    if missing:
        raise SystemExit(f"NY IDA columns not found for {missing}. Columns are: {list(raw.columns)}")
    print(f"  NY IDA columns used: name={name_col}, year={year_col}, "
          f"value={value_col}, project={project_col}")

    # Not-for-profit applicants (hospitals, colleges) can't be public companies.
    nfp = first_column(raw, [r"not_for_profit"])
    if nfp:
        raw = raw[raw[nfp].fillna("No").str.upper() != "YES"]
    alt_col = "project_name" if "project_name" in raw.columns and name_col != "project_name" else None

    year = raw[year_col].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    out = pd.DataFrame({
        "source": "ny_ida",
        "recipient": raw[name_col],
        "year": year,
        "value": pd.to_numeric(raw[value_col], errors="coerce").fillna(0.0),
        "deal_id": raw[project_col] if project_col else raw[name_col],
        "state": "NY",
        "parent": None,
        "alt_name": raw[alt_col] if alt_col else None,
    })
    return out.dropna(subset=["year"])


# ---------------------------------------------------------- Good Jobs First

def tidy_gjf(path):
    """A Subsidy Tracker CSV export, in the same shape as tidy_ny_ida."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    name_col = first_column(raw, [r"^company$", r"recipient", r"company"], exclude=[r"parent"])
    parent_col = first_column(raw, [r"parent"])
    year_col = first_column(raw, [r"^year$", r"year"])
    value_col = first_column(raw, [r"subsidy_value", r"value", r"amount"])
    state_col = first_column(raw, [r"^location$", r"state"])
    value = (raw[value_col].astype(str).str.replace(r"[$,]", "", regex=True))
    return pd.DataFrame({
        "source": "gjf",
        "recipient": raw[name_col],
        "year": pd.to_numeric(raw[year_col], errors="coerce"),
        "value": pd.to_numeric(value, errors="coerce").fillna(0.0),
        "deal_id": raw.index.astype(str),
        "state": raw[state_col] if state_col else None,
        "parent": raw[parent_col] if parent_col else None,
        "alt_name": None,
    }).dropna(subset=["year"])


# ------------------------------------------------------------ matching

def match_names(names, companies):
    """Map each recipient name to a cik (or NaN), with how it was matched.

    1. exact on the strict clean name; 2. exact on the loose clean name, if it
    has two or more words; 3. fuzzy on the loose name (two or more words),
    against companies sharing its first word, similarity >= FUZZY_CUTOFF.
    """
    companies = companies.assign(strict=companies["name"].map(clean_name),
                                 loose=companies["name"].map(lambda n: clean_name(n, True)))
    strict = companies[companies["strict"].str.len() >= 3].drop_duplicates("strict").set_index("strict")["cik"]
    loose = companies[companies["loose"].str.len() >= 3].drop_duplicates("loose").set_index("loose")["cik"]
    by_first_word = {}
    for name in loose.index:
        if len(name.split()) >= 2:
            by_first_word.setdefault(name.split()[0], []).append(name)

    rows = []
    for i, name in enumerate(names):
        s, l = clean_name(name), clean_name(name, True)
        cik, how, matched, score = np.nan, "none", None, np.nan
        alias = next((c for p, c in ALIASES.items() if re.search(p, name.upper())), None)
        if alias is not None:
            cik, how, matched, score = alias, "alias", None, 1.0
        elif s in strict.index:
            cik, how, matched, score = strict[s], "exact", s, 1.0
        elif len(l.split()) >= 2 and l in loose.index:
            cik, how, matched, score = loose[l], "exact_loose", l, 1.0
        elif len(l.split()) >= 2:
            best = difflib.get_close_matches(l, by_first_word.get(l.split()[0], []), n=1,
                                             cutoff=FUZZY_CUTOFF)
            if best:
                matched = best[0]
                cik, how = loose[matched], "fuzzy"
                score = difflib.SequenceMatcher(None, l, matched).ratio()
        rows.append((name, s, matched, cik, how, score))
        if (i + 1) % 5000 == 0:
            print(f"  names matched so far: {i + 1:,} of {len(names):,}", flush=True)
    return pd.DataFrame(rows, columns=["recipient", "recipient_clean", "matched_clean",
                                       "cik", "match_type", "score"])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gjf-file", help="Good Jobs First Subsidy Tracker CSV export")
    ap.add_argument("--no-ny-ida", action="store_true", help="skip the NY IDA download")
    ap.add_argument("--lag-years", type=int, default=1,
                    help="use abatements from this many years before the fiscal year (default 1)")
    args = ap.parse_args()

    companies = pd.read_csv(config.DATA_DIR / "company_info.csv", usecols=["cik", "name", "hq_state"])
    tax = pd.read_csv(config.DATA_DIR / "tax_company_year.csv", usecols=["cik", "fiscal_year", "revenue"])

    parts = []
    if not args.no_ny_ida:
        print("Downloading NY IDA project data...")
        parts.append(tidy_ny_ida(download_ny_ida()))
    if args.gjf_file:
        print(f"Reading Good Jobs First file {args.gjf_file}...")
        parts.append(tidy_gjf(args.gjf_file))
    if not parts:
        raise SystemExit("Nothing to do: give --gjf-file or drop --no-ny-ida.")
    deals = pd.concat(parts, ignore_index=True)
    # Good Jobs First's parent company is the better name to match on when it exists.
    deals["match_on"] = deals["parent"].where(deals["parent"].notna(), deals["recipient"])
    print(f"  {len(deals):,} abatement rows, {deals['match_on'].nunique():,} distinct names")

    print("Matching names to EDGAR companies...")
    # Applicants are often property LLCs, but the IDA project name often names the
    # real tenant ("Acme Corp Project"), so it is tried when the applicant fails.
    names = pd.concat([deals["match_on"], deals["alt_name"]]).dropna().unique().tolist()
    matches = match_names(names, companies)
    matches.to_csv(config.DATA_DIR / "abatement_name_matches.csv", index=False)
    lookup = matches.set_index("recipient")[["cik", "match_type"]]
    main_hit = lookup.reindex(deals["match_on"]).reset_index(drop=True)
    alt_hit = lookup.reindex(deals["alt_name"]).reset_index(drop=True)
    use_alt = main_hit["cik"].isna() & alt_hit["cik"].notna()
    deals["cik"] = main_hit["cik"].where(~use_alt, alt_hit["cik"]).values
    deals["match_type"] = main_hit["match_type"].where(~use_alt, alt_hit["match_type"] + "_project_name").values

    # Match rate report
    hit = deals["cik"].notna()
    print("\nMatch rate")
    print(f"  names: {matches['cik'].notna().mean():.1%} of {len(matches):,} "
          f"({(matches.match_type == 'exact').sum():,} exact, {(matches.match_type == 'fuzzy').sum():,} fuzzy)")
    print(f"  rows:  {hit.mean():.1%};  dollars: {deals.loc[hit, 'value'].sum() / max(deals['value'].sum(), 1):.1%}")
    print(f"  companies matched: {deals.loc[hit, 'cik'].nunique():,}")
    print("  rows by match type:", deals["match_type"].value_counts().to_dict())
    deals[hit].to_csv(config.DATA_DIR / "abatement_matched_deals.csv", index=False)

    # Company-year features, shifted by the reporting lag
    m = deals[hit].assign(fiscal_year=lambda d: d["year"].astype(int) + args.lag_years)
    agg = (m.groupby(["cik", "fiscal_year"])
             .agg(abate_value=("value", "sum"), abate_deals=("deal_id", "nunique"))
             .reset_index())
    out = tax.merge(agg, on=["cik", "fiscal_year"], how="left")
    out["abate_value"] = out["abate_value"].fillna(0.0)
    out["abate_deals"] = out["abate_deals"].fillna(0).astype(int)
    out["abate_any"] = (out["abate_deals"] > 0).astype(int)
    rev = out["revenue"].where(out["revenue"] > 0)
    out["abate_to_revenue"] = (out["abate_value"] / rev).fillna(0.0)
    out = out.merge(companies[["cik", "hq_state"]], on="cik", how="left")
    out["abate_ny_covered"] = (out["hq_state"] == "NY").astype(int)

    years = deals["year"].astype(int) + args.lag_years
    out["abate_data_year"] = out["fiscal_year"].between(years.min(), years.max()).astype(int)
    cols = ["cik", "fiscal_year", "abate_any", "abate_value", "abate_to_revenue",
            "abate_deals", "abate_ny_covered", "abate_data_year"]
    out[cols].to_csv(config.DATA_DIR / "abatements_company_year.csv", index=False)

    print("\nCoverage")
    print(f"  company-years with an abatement: {out['abate_any'].sum():,} of {len(out):,} "
          f"({out['abate_any'].mean():.2%})")
    by_state = (out[out["abate_any"] == 1].groupby("hq_state")["cik"].nunique()
                .sort_values(ascending=False).head(10))
    print("  companies with an abatement, by HQ state (top 10):")
    for st, n in by_state.items():
        print(f"    {st}: {n}")
    print(f"\nWrote data/abatements_company_year.csv and data/abatement_name_matches.csv")


if __name__ == "__main__":
    main()
