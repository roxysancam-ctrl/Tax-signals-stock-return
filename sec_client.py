"""A small, polite downloader for SEC EDGAR.

- Sends the User-Agent header the SEC requires.
- Waits between requests so we stay under the SEC rate limit.
- Saves every download to cache/ (gzipped), so re-running a script reads
  from disk instead of downloading again. Delete cache/ to force fresh data.
"""
import gzip
import json
import time

import requests

import config

_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT})
_last_request_time = 0.0


def _wait_for_rate_limit():
    global _last_request_time
    min_gap = 1.0 / config.MAX_REQUESTS_PER_SECOND
    elapsed = time.monotonic() - _last_request_time
    if elapsed < min_gap:
        time.sleep(min_gap - elapsed)
    _last_request_time = time.monotonic()


def _check_user_agent():
    if "example.com" in config.USER_AGENT:
        raise SystemExit(
            "Please put your name and email in USER_AGENT in config.py first. "
            "The SEC rejects requests without it."
        )


def _download(url):
    """Return the response text, or None if the SEC says the file does not exist."""
    _check_user_agent()
    for attempt in range(5):
        _wait_for_rate_limit()
        try:
            resp = _session.get(url, timeout=60)
        except requests.RequestException as err:
            print(f"  network error ({err}); retrying")
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.text
        if resp.status_code == 404:
            return None
        if resp.status_code == 403:
            raise SystemExit(
                f"SEC refused {url} (403). Usually this means USER_AGENT in "
                "config.py is missing an email, or we went over the rate limit "
                "(the SEC then blocks you for about 10 minutes)."
            )
        # 429 / 5xx: the server is busy, back off and try again
        print(f"  HTTP {resp.status_code} for {url}; retrying")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Gave up on {url} after 5 tries")


def get(url, cache_name, as_json=True):
    """Download url, or read it from cache/<cache_name>.gz if we already have it.

    Returns parsed JSON (or text if as_json=False), or None if the file
    does not exist on EDGAR (for example, a company with no XBRL data).
    """
    path = config.CACHE_DIR / f"{cache_name}.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            text = f.read()
    else:
        text = _download(url)
        if text is None:
            text = "null" if as_json else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    if as_json:
        return json.loads(text)
    return text or None
