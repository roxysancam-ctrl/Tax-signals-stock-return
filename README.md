# Tax-signals-stock-return
# Do Tax Signals Predict Stock Returns?

**Question:** Using a company's tax data, tax abatements and filing behavior, can we predict whether its stock will beat the S&P 500 over the next 12 months?

This is a solo portfolio project that follows a group capstone on credit card churn (BankChurners: Logit, Random Forest, AdaBoost, XGBoost). It uses the same four models, but on public company data and from an investor's point of view.

- **One row** = one company in one fiscal year (a "company-year").
- **Target** = `beat_sp500`: 1 if the stock's total return over the 12 months after its annual report (10-K) was filed was higher than the S&P 500's, else 0.
- **Models** = Logistic regression, Random Forest, AdaBoost, XGBoost.
- **Payoff** = feature importance (which tax and filing signals matter), then a test portfolio of the top-ranked stocks against the S&P 500.

## Status (15 Oct 2026)

| Step | What | Status |
|---|---|---|
| 1 | Tax figures from SEC EDGAR | Done |
| 2 | Late-filing notices (NT 10-K / NT 10-Q) | Done |
| 3 | Stock returns and the target | Done |
| 4 | Tax abatements (NY IDA) | Done |
| 5 | Controls and the modeling table | Done |
| 6 | Train the four models, feature importance | In progress |
| 7 | Portfolio test vs the S&P 500 | Next |

## The project step by step

### Step 1. Tax figures (`pull_tax_data.py`)

Every US public company files its annual report (10-K) with the SEC, and since about 2010 the numbers come tagged in a machine-readable format (XBRL). The script downloads each company's "company facts" from the SEC's free API and keeps, for every fiscal year, the number from the **first** 10-K that reported it. Later restatements are ignored because investors didn't have them at the time.

Features built from it:
- **Effective tax rate (ETR)** = income tax expense / pretax income.
- **Cash ETR** = cash taxes actually paid / pretax income, and **cash vs book tax** (cash paid / tax expense).
- **Change in ETR** from last year.
- **Deferred taxes**: net deferred tax balance / assets, deferred share of tax expense, valuation allowance, unrecognized tax benefits.
- Profitability (ROA) and an estimate of filer status (large accelerated, accelerated, non-accelerated).

Result: 56,340 company-years for 5,006 companies, fiscal 2010-2025. (8,028 companies were checked; most of the rest are foreign companies that report under IFRS and have no US tax tags.)

### Step 2. Late filing (`pull_nt_filings.py`)

A company that can't file its 10-K or 10-Q on time must file a notice called an **NT 10-K** or **NT 10-Q**. Late filing often means accounting trouble. The script reads EDGAR's filing history for every company and builds:
- `nt_10k` (any late 10-K that year), `nt_10q_count`, `nt_count`
- `nt_max_days_late`, `nt_missed_extension` (still late after the grace period), `nt_never_filed`
- `tenk_filing_lag_days` (days from fiscal year-end to the 10-K)

Result: 19,110 NT notices; about 9% of company-years have a late 10-K.

### Step 3. Returns and the target (`pull_returns.py`)

Daily prices (adjusted for splits and dividends) come from Yahoo Finance's chart API. For each company-year, the 12-month window **starts the first trading day after the 10-K was filed**, because that is when investors first saw the numbers. The benchmark is SPY (an S&P 500 fund) over exactly the same days.

Result: 48,714 company-years with a target; 38.8% beat the S&P 500. Less than half is normal: the S&P 500 is weighted toward its biggest winners, so the typical stock trails it. Last year's return (`prior_ret_12m`) is kept as a control.

### Step 4. Tax abatements (`pull_abatements.py`)

Abatements are tax breaks a government gives a company, usually in exchange for jobs or investment. The free source used is the **New York State IDA** project data (data.ny.gov), which lists every IDA deal and its tax exemptions by year. Recipient names (often "XYZ Holdings LLC") are matched to public companies by cleaned exact names, careful fuzzy matching and a short alias list, and every match was checked by hand.

Features: `abate_any`, `abate_value`, `abate_to_revenue`, `abate_deals`.

Result: 80 companies matched, 404 company-years with an abatement (0.7%). The national source (Good Jobs First Subsidy Tracker) is behind a paywall, so the abatement signal is **New York only, from fiscal 2018**.

### Step 5. Controls and the modeling table (`build_controls.py`, `build_modeling_table.py`)

To check that tax signals add something beyond "big, profitable companies do well", the table adds controls: company size (quintile and log assets), filer status, industry (Fama-French 12), exchange, HQ region, fiscal year-end month, dividend payer, years filing with the SEC, last year's return, and profitability.

Then everything is joined on `cik` + `fiscal_year` and cleaned:
- **Time-based split, never random.** Train = fiscal 2012-2020 (26,660 rows, 40.3% beat). Test = fiscal 2021-2024 (17,008 rows, 33.9% beat). The model never sees the future.
- **Blank tax rates.** ETR is blank in about half the rows, mostly loss years (a tax rate on a loss means nothing). Loss years are kept and treated as their own group: tree models get -1, logit gets the training median plus `pretax_loss` / `pretax_tiny` flags.
- **Outliers** are clipped at the 1st/99th percentile, using training years only.
- **Dropped:** rows with no target yet (fiscal 2025), 341 rows with broken penny-stock prices, and 126 SPACs (cash shells).

Result: `data/modeling_table.csv`, 43,668 company-years for 4,522 companies.

### Step 6. Models and feature importance (in progress)

Train Logit, Random Forest, AdaBoost and XGBoost on 2012-2020 and score them on 2021-2024 (AUC, plus accuracy, precision and recall). Permutation importance shows which features the models lean on. Beating the market is hard, so an AUC of 0.55-0.60 on the test years would count as a real signal. A controls-only model is the baseline the tax features have to beat.

### Step 7. Portfolio test (next)

Each test year, buy the top 10-20% of stocks the best model ranks highest and compare that portfolio to the S&P 500: return, volatility, Sharpe ratio and alpha (using the Fama-French factors from Ken French's data library).

## Data sources

| Source | Used for | Cost |
|---|---|---|
| SEC EDGAR XBRL API (data.sec.gov) | Tax figures, company facts | Free (needs a User-Agent with your email) |
| SEC EDGAR filing index | NT late-filing notices, 10-K dates | Free |
| NY State IDA projects (data.ny.gov/d/9rtk-3fkw) | Tax abatements | Free |
| Yahoo Finance chart API | Daily prices, SPY | Free |
| Ken French data library | Factors for alpha (step 7) | Free |
| Good Jobs First Subsidy Tracker | National abatements | Paywalled, not used |

## Files

### Code (run in this order)

| File | What it does | Writes |
|---|---|---|
| `config.py` | Shared settings: SEC User-Agent, years (2010-2025), folders | |
| `sec_client.py` | Polite SEC downloader: rate limit, caching | `cache/` |
| `companies.py` | Which companies to pull, their filing history | |
| `pull_tax_data.py` | Step 1 | `data/tax_company_year.csv` |
| `pull_nt_filings.py` | Step 2 | `data/filing_company_year.csv`, `data/nt_filings.csv`, `data/company_info.csv` |
| `pull_returns.py` | Step 3 | `data/returns_company_year.csv` |
| `pull_abatements.py` | Step 4 | `data/abatements_company_year.csv`, `data/abatement_name_matches.csv`, `data/abatement_matched_deals.csv` |
| `build_controls.py` | Step 5 (dividends, years filing) | `data/controls_company_year.csv` |
| `build_modeling_table.py` | Step 5 (join + clean) | `data/modeling_table.csv` |
| `requirements.txt` | Python packages | |

### Data (`data/`, about 51 MB)

| File | One row per | Rows | Columns |
|---|---|---|---|
| `tax_company_year.csv` | company-year | 56,340 | 37 |
| `filing_company_year.csv` | company-year | 55,422 | 11 |
| `nt_filings.csv` | late-filing notice | 19,110 | 10 |
| `company_info.csv` | company | 8,028 | 11 |
| `returns_company_year.csv` | company-year | 56,340 | 13 |
| `abatements_company_year.csv` | company-year | 56,340 | 8 |
| `abatement_name_matches.csv` | recipient name | 10,242 | 6 |
| `abatement_matched_deals.csv` | matched IDA deal | 750 | 11 |
| `controls_company_year.csv` | company-year | 56,340 | 5 |
| `modeling_table.csv` | company-year (final) | 43,668 | 54 |

`cache/` (about 1.1 GB of raw SEC, Yahoo and NY downloads) and `__pycache__/` are **not** uploaded; `.gitignore` leaves them out. The scripts rebuild `cache/` on the first run.

## Limitations

- **Survivorship bias.** The company list is today's listed companies, and Yahoo has almost no prices for delisted stocks, so companies that went bust are mostly missing. This likely makes every strategy look a bit better than it was.
- **Today's labels.** Sector, exchange and HQ state are today's values applied to every past year.
- **Abatements are New York only, from fiscal 2018.** Only 0.7% of company-years have one, so this signal is thin.
- **Sparse tax rates.** ETR is blank in about half of the rows (mostly loss years).
- **Short test window.** 2021-2024 is four years and was dominated by a few mega-cap stocks, which is why only 33.9% of stocks beat the index.
- **No trading costs** in the portfolio test.
- **Correlation is not causation.** A tax signal that predicts returns may just be standing in for something else (size, profitability, risk).

## How to re-run it

### Setup

1. `pip install -r requirements.txt` (pandas and requests; the modeling step also needs scikit-learn and xgboost).
2. Open `config.py` and put your own name and email in `USER_AGENT`. The SEC blocks requests without it.

### Run

Start with a few companies, check the CSVs look right, then run everything.

```bash
python pull_tax_data.py   --tickers AAPL MSFT TSLA GME
python pull_nt_filings.py --tickers AAPL MSFT TSLA GME

python pull_tax_data.py   --all     # every listed company; can take an hour
python pull_nt_filings.py --all
```

Add `--universe full-index` to include companies that have since been delisted or gone bankrupt (see survivorship bias). Every download is saved in `cache/`, so a re-run, or a run that crashed halfway, does not download things twice.

In a notebook you can call the same functions, for example `from pull_tax_data import company_years`.

Then the stock returns and the target (needs access to query2.finance.yahoo.com; the script calls Yahoo's chart API directly, so the yfinance package is not needed):

```bash
python pull_returns.py --tickers AAPL MSFT TSLA GME
python pull_returns.py --all
```

Then abatements, controls and the modeling table:

```bash
python pull_abatements.py
# copy the cache to a local disk first (reading it from a network folder is very slow)
python build_controls.py --cache-dir /path/to/local/cache   # dividend payer, years since first SEC filing
python build_modeling_table.py                              # joins everything -> data/modeling_table.csv
```

## Technical details

### What each data file holds

| File | One row per | Key columns |
|---|---|---|
| `tax_company_year.csv` | company, fiscal year | `etr`, `cash_etr`, `cash_to_book_tax`, `etr_change`, `net_deferred_tax_to_assets`, `deferred_share_of_tax`, `valuation_allowance_to_assets`, `unrecognized_tax_benefits_to_assets`, `roa`, `filer_status_est`, plus the raw dollar figures |
| `filing_company_year.csv` | company, fiscal year | `nt_10k`, `nt_10q_count`, `nt_count`, `nt_max_days_late`, `nt_never_filed`, `nt_missed_extension`, `tenk_filing_lag_days` |
| `nt_filings.csv` | NT notice | which report it delayed and how many days late it came |
| `company_info.csv` | company | SIC code, HQ state, exchange, fiscal year-end month, filer category |
| `abatements_company_year.csv` | company, fiscal year | `abate_any`, `abate_value`, `abate_to_revenue`, `abate_deals`, `abate_ny_covered`, `abate_data_year` (from `pull_abatements.py`, NY IDA; matches listed in `abatement_name_matches.csv`) |
| `returns_company_year.csv` | company, fiscal year | `beat_sp500` (the target), `ret_12m`, `sp500_ret_12m`, `excess_ret_12m`, `prior_ret_12m`, `delisted_in_window`, `suspect_price` |

Join the company-year files on `cik` + `fiscal_year`. Returns run for 12 months from the first trading day after the 10-K filing date, with dividends included; the S&P 500 is SPY over the same days. `beat_sp500` is blank when the 12 months haven't ended yet. `suspect_price` = 1 marks rows whose return is impossible or extreme (<= -100% or > +1000%), almost always broken Yahoo prices for penny stocks; drop them when modeling.

### The modeling table

`data/modeling_table.csv` has one row per company-year, joined on `cik` + `fiscal_year`, with a `split` column: `train` = fiscal 2012-2020, `test` = 2021-2024. Rows are dropped when there is no target (mostly fiscal 2025, whose 12 months aren't over), when `suspect_price` = 1, or when the company is a SPAC (SIC 6770, a cash shell parked near $10). The 14 Oct 2026 build has 43,668 company-years for 4,522 companies: 26,660 train (40.3% beat the S&P 500) and 17,008 test (33.9%).

- **Blank ETRs.** `etr_tree` / `cash_etr_tree` fill blanks with -1 (for RF, AdaBoost, XGBoost); `etr_logit` / `cash_etr_logit` fill them with the training-years median, and logit also gets `pretax_loss`, `pretax_tiny` and `pretax_missing` as flags. The raw `etr` / `cash_etr` are kept too.
- **Winsorizing.** Ratios, lags and the prior-year return are clipped at the 1st and 99th percentile of the training years only.
- **Controls.** `size_quintile` (total assets, within each fiscal year), `log_total_assets`, `filer_status_est`, `ff12_industry` (Fama-French 12 from SIC), `exchange` (NYSE / Nasdaq / OTC / Other), `hq_region` (Census region, or Foreign), `fye_month`, `dividend_payer` (common dividends > 0 in that year's 10-K), `years_since_first_filing`, `prior_ret_12m`, `roa`, `pretax_roa`. One-hot encode the text columns in the modeling step.
- **Other blanks** (for example `cash_to_book_tax`, `nt_max_days_late`) are left blank; the script prints how many, and each model handles them.
- **Abatements.** Once `data/abatements_company_year.csv` exists (`cik`, `fiscal_year` and feature columns), re-run `build_modeling_table.py` and it is joined automatically; company-years with no match get 0.

Control limits:
- Sector, exchange and HQ state are today's values from EDGAR, applied to every past year.
- `years_since_first_filing` counts from the first EDGAR filing. EDGAR starts around 1993-1996, so older companies are capped near 20-30 years, and a first filing can come before the IPO.
- **Left out:** Big 4 auditor (the auditor tag only exists from 2021) and number of subsidiaries (only in the Exhibit 21 text, not in XBRL). Neither can be built reliably from SEC data for 2012-2020.

### Design choices (and why)

- **No look-ahead.** Each tax figure is the value from the first 10-K that reported it, not a later restatement. `filed` (tax file) and `tenk_filed` (filing file) are the dates the row became public. Measure the "next 12 months" return from that date, not from the fiscal year-end.
- **Fiscal year.** A year ending June to December is that calendar year; one ending January to May counts as the year before (the Compustat rule). Example: a year ending 31 Jan 2021 is fiscal 2020.
- **Negative or tiny pretax income.** If pretax income is zero, negative, or under 1% of total assets, `etr` and `cash_etr` are left blank, and the flags `pretax_loss` / `pretax_tiny` record why. These flags can be features themselves. Usable ETRs are capped to the 0 to 1 range.
- **Days late.** An NT notice is due one business day after the missed deadline, so the days from the notice to the actual report is roughly how many days late it was. The notice buys 15 extra days for a 10-K and 5 for a 10-Q; `nt_missed_extension` = 1 means the company was late even after that.
- **Winsorizing** happens in the modeling table step, using training years only.

### Known gaps

- **Abatements are NY only and start in 2017.** NY IDA data covers reporting years 2017-2024, used with a one-year lag, so features exist for fiscal 2018-2025 only. 80 companies matched (26 headquartered in NY); most IDA recipients are property LLCs that can't be tied to a listed company. Good Jobs First export is behind a paywall (checked 2026-10-15), so it is not used; `--gjf-file` stays in case a CSV is ever available.
- **Survivorship bias.** The default company list is today's listed companies. `--universe full-index` fixes the EDGAR side, but Yahoo also lacks most delisted stocks, so name this in the limitations.
- **Filer status over time.** `filer_category` in `company_info.csv` is today's status. `filer_status_est` rebuilds it for each year from public float ($75M / $700M thresholds), which is close but not exact after the 2020 rule change.
- **Rows with `from_comparative` = 1** come from the company's first XBRL filing, which reported earlier years as comparatives. Their `filed` date is later than when the market first saw the numbers. Dropping them is the simplest fix.
- **Missing tags.** When a company never tags a total (pretax income, current or deferred tax), the script adds up the domestic/foreign or federal/state/foreign parts, and as a last resort uses after-tax income + tax for pretax income. About 15% of rows still have no pretax income, mostly companies that report no income tax at all.
- **Coverage.** The full run (13 Oct 2026) covered 8,028 listed companies; 5,006 have US-GAAP tax data (most of the rest are foreign filers using IFRS).
- **XBRL coverage** is thin before about 2010, and some small companies use custom tags, which leaves some fields blank.
- **Big 4 auditor** isn't pulled. The `dei:AuditorName` tag only exists from 2021, so earlier years need another source, such as Audit Analytics or the PCAOB Form AP data.

