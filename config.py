"""Settings shared by every EDGAR script in this folder.

Edit USER_AGENT before running anything. The SEC blocks requests that do
not say who is asking: https://www.sec.gov/os/accessing-edgar-data
"""
from pathlib import Path

# Required by the SEC: your name (or project name) and an email address.
USER_AGENT = "Your Name your.email@example.com"

# The SEC allows at most 10 requests per second. Stay a little under it.
MAX_REQUESTS_PER_SECOND = 8

# First and last fiscal years to keep. XBRL data is patchy before ~2010,
# and one extra year before the 2012 training start lets us compute
# year-over-year changes.
START_YEAR = 2010
END_YEAR = 2025

# Pretax income smaller than this share of total assets is treated as
# "too small" for a tax rate to mean anything (ETR is set to missing).
TINY_PRETAX_SHARE_OF_ASSETS = 0.01

# Folders (relative to this file, so the scripts work wherever you copy them)
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"    # finished CSVs
CACHE_DIR = PROJECT_DIR / "cache"  # raw SEC downloads, gzipped, so re-runs are fast
