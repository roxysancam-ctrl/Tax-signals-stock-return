"""Stock returns and the target: did the stock beat the S&P 500 over the next 12 months?

Output: data/returns_company_year.csv, one row per company per fiscal year
(the same rows as tax_company_year.csv; join on cik + fiscal_year).

How the return is measured
- The 12 months start the first trading day AFTER the 10-K was filed
  ('filed' in tax_company_year.csv), because that is when investors first
  saw the numbers. Starting at the fiscal year-end would be look-ahead.
- Prices are adjusted for splits and dividends, so a return is the total
  return an investor got. The S&P 500 is SPY (an S&P 500 fund), also with
  dividends, over exactly the same days.
- beat_sp500 = 1 if the stock's return was higher than SPY's, 0 if not, and
  blank when it can't be known yet (the 12 months haven't ended) or there
  are no prices for the stock.
- If a stock stops trading inside the window (bought out, delisted), the
  return runs to its last price and delisted_in_window = 1. Yahoo has few
  delisted stocks, so most of them are simply missing (survivorship bias).

Also adds prior_ret_12m (the 12 months BEFORE the filing), a control feature.

Usage
    python pull_returns.py --tickers AAPL MSFT TSLA GME   # a few companies
    python pull_returns.py --all                          # every company in tax_company_year.csv
"""
import argparse
import gzip
import time

import numpy as np
import pandas as pd

import config

BENCHMARK = "SPY"
PRICE_START = f"{config.START_YEAR - 1}-01-01"  # one extra year for prior_ret_12m
BATCH_SIZE = 100           # symbols per cache file
SECONDS_BETWEEN_BATCHES = 1
SECONDS_BETWEEN_SYMBOLS = 0.3
MAX_DAYS_TO_FIRST_PRICE = 7  # no price within a week of the filing -> treat as missing
PRICE_CACHE = config.CACHE_DIR / "prices"
SUSPECT_RETURN = 10          # a 12-month return above +1000% is flagged as a likely data error


def yahoo_symbol(ticker):
    """SEC ticker -> Yahoo symbol (Yahoo writes share classes with a dash: BRK-B)."""
    return ticker.upper().replace(".", "-")


# ------------------------------------------------------------- download

YAHOO_CHART = "https://query2.finance.yahoo.com/v8/finance/chart/{}"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}  # Yahoo refuses requests without a browser-like agent


def _download_symbol(session, symbol):
    """Dividend- and split-adjusted daily closes for one symbol (empty frame if Yahoo has none)."""
    params = {"period1": int(pd.Timestamp(PRICE_START).timestamp()), "period2": int(time.time()),
              "interval": "1d", "events": "div,splits"}
    for attempt in range(6):
        try:
            resp = session.get(YAHOO_CHART.format(symbol), params=params, timeout=30)
        except Exception as err:
            print(f"  {symbol}: network error ({err}); retrying")
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 404:          # unknown or delisted symbol
            return pd.DataFrame(columns=["date", "symbol", "close"])
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * 2 ** attempt)     # rate limited or Yahoo hiccup
            continue
        resp.raise_for_status()
        result = (resp.json().get("chart", {}).get("result") or [None])[0]
        if not result or "timestamp" not in result:
            return pd.DataFrame(columns=["date", "symbol", "close"])
        adj = result["indicators"].get("adjclose") or result["indicators"]["quote"]
        closes = adj[0].get("adjclose", adj[0].get("close"))
        dates = pd.to_datetime(result["timestamp"], unit="s").normalize()
        return pd.DataFrame({"date": dates, "symbol": symbol, "close": closes}).dropna()
    raise RuntimeError(f"Gave up on {symbol} after repeated Yahoo errors")


def _download_batch(symbols):
    """Adjusted daily closes for a list of symbols, long format: date, symbol, close.

    Uses Yahoo's chart API directly (the same data yfinance reads, auto-adjusted),
    so the script runs without the yfinance package.
    """
    import requests
    session = requests.Session()
    session.headers.update(YAHOO_HEADERS)
    parts = []
    for symbol in symbols:
        parts.append(_download_symbol(session, symbol))
        time.sleep(SECONDS_BETWEEN_SYMBOLS)
    long = pd.concat(parts, ignore_index=True)
    long = long.drop_duplicates(["date", "symbol"], keep="last")
    # Symbols with no data at all still get one row, so a re-run knows they were tried.
    tried = pd.DataFrame({"symbol": [s for s in symbols if s not in set(long["symbol"])]})
    return pd.concat([long, tried], ignore_index=True)


def load_prices(symbols):
    """Daily adjusted closes (date x symbol), downloading only symbols not in the cache yet."""
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    cached = [pd.read_csv(f, parse_dates=["date"]) for f in sorted(PRICE_CACHE.glob("batch_*.csv.gz"))]
    have = set(pd.concat(cached)["symbol"]) if cached else set()
    todo = [s for s in dict.fromkeys(symbols) if s not in have]
    if todo:
        print(f"Downloading prices for {len(todo):,} symbols ({len(have):,} already cached)")
    next_id = len(cached)
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        long = _download_batch(batch)
        path = PRICE_CACHE / f"batch_{next_id:04d}.csv.gz"
        with gzip.open(path, "wt") as f:
            long.to_csv(f, index=False)
        cached.append(long)
        next_id += 1
        print(f"  {min(start + BATCH_SIZE, len(todo)):,}/{len(todo):,} downloaded")
        time.sleep(SECONDS_BETWEEN_BATCHES)
    prices = pd.concat(cached).dropna(subset=["date", "close"])
    prices = prices[prices["symbol"].isin(set(symbols))]
    return prices.pivot_table(index="date", columns="symbol", values="close").sort_index()


# ---------------------------------------------------------- return math

def window_return(closes, start_after, months=12):
    """Total return of one price series over `months`, starting the first trading day after `start_after`.

    Returns (ret, start_date, end_date, target_end). end_date is earlier than
    target_end when the stock stopped trading, or when the prices run out.
    """
    s = closes.dropna()
    later = s[s.index > start_after]
    if later.empty or (later.index[0] - start_after).days > MAX_DAYS_TO_FIRST_PRICE:
        return np.nan, pd.NaT, pd.NaT, pd.NaT
    start_date = later.index[0]
    target_end = start_date + pd.DateOffset(months=months)
    inside = later[later.index <= target_end]
    return inside.iloc[-1] / inside.iloc[0] - 1, start_date, inside.index[-1], target_end


def prior_return(closes, before, months=12):
    """Total return over the `months` up to (and including) the last trading day on or before `before`."""
    s = closes.dropna()
    s = s[s.index <= before]
    if s.empty or (before - s.index[-1]).days > MAX_DAYS_TO_FIRST_PRICE:
        return np.nan
    first = s[s.index >= s.index[-1] - pd.DateOffset(months=months)]
    if (first.index[0] - (s.index[-1] - pd.DateOffset(months=months))).days > MAX_DAYS_TO_FIRST_PRICE:
        return np.nan  # less than a full year of history
    return first.iloc[-1] / first.iloc[0] - 1


def company_year_returns(rows, prices):
    """rows: cik, ticker, fiscal_year, filed. prices: date x symbol adjusted closes (must include SPY)."""
    bench = prices[BENCHMARK]
    last_price_date = bench.dropna().index[-1]
    out = []
    for r in rows.itertuples(index=False):
        symbol = yahoo_symbol(r.ticker)
        closes = prices[symbol] if symbol in prices else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        ret, start, end, target_end = window_return(closes, r.filed)
        has_start = pd.notna(start)
        # Stopped trading = last price more than a week before the window end,
        # while the benchmark kept trading (so it isn't just the end of our data).
        stopped = has_start and end < min(target_end, last_price_date) - pd.Timedelta(days=7)
        finished = has_start and (target_end <= last_price_date or stopped)
        if finished:
            b = bench[(bench.index >= start) & (bench.index <= end)].dropna()
            sp = b.iloc[-1] / b.iloc[0] - 1
        else:
            ret = sp = np.nan
        out.append({
            "cik": r.cik, "fiscal_year": r.fiscal_year, "symbol": symbol,
            "ret_start": start, "ret_end": end if finished else pd.NaT,
            "ret_12m": ret, "sp500_ret_12m": sp, "excess_ret_12m": ret - sp,
            "beat_sp500": float(ret > sp) if finished else np.nan,
            "delisted_in_window": int(bool(finished and stopped)),
            "has_prices": int(not closes.dropna().empty),
            "prior_ret_12m": prior_return(closes, r.filed),
        })
    df = pd.DataFrame(out)
    df["beat_sp500"] = df["beat_sp500"].astype("Int64")
    # Yahoo's adjusted prices are sometimes broken for penny stocks (negative
    # adjusted closes, unadjusted reverse splits). Flag impossible or extreme
    # returns so modeling can drop them; the rows are kept here.
    extreme = lambda r: (r <= -1) | (r > SUSPECT_RETURN)
    df["suspect_price"] = (extreme(df["ret_12m"]) | extreme(df["prior_ret_12m"])).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--tickers", nargs="+", help="only these tickers, e.g. AAPL MSFT")
    which.add_argument("--all", action="store_true", help="every company in tax_company_year.csv")
    parser.add_argument("--tax-file", default=str(config.DATA_DIR / "tax_company_year.csv"))
    parser.add_argument("--out", default=str(config.DATA_DIR / "returns_company_year.csv"))
    args = parser.parse_args()

    rows = pd.read_csv(args.tax_file, usecols=["cik", "ticker", "fiscal_year", "filed"],
                       parse_dates=["filed"])
    if args.tickers:
        rows = rows[rows["ticker"].str.upper().isin([t.upper() for t in args.tickers])]
    if rows.empty:
        raise SystemExit("None of those tickers are in the tax file.")

    symbols = [BENCHMARK] + sorted(set(rows["ticker"].map(yahoo_symbol)))
    prices = load_prices(symbols)
    df = company_year_returns(rows, prices)

    df.to_csv(args.out, index=False)
    known = df["beat_sp500"].notna()
    print(f"Saved {len(df):,} company-years to {args.out}")
    print(f"  {known.sum():,} have a target; {df['beat_sp500'].mean():.1%} of those beat the S&P 500")
    print(f"  {(df['has_prices'] == 0).sum():,} rows had no Yahoo prices")
    print(f"  {df['suspect_price'].sum():,} rows flagged suspect_price (return <= -100% or > +1000%)")


if __name__ == "__main__":
    main()
