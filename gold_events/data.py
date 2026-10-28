"""Download, cache and align the daily series used by the event studies.

Run ``python -m gold_events.data`` to (re)build ``data/daily.parquet`` and
``reports/data_coverage.md``.

Design
------
* Every series is described by a :class:`SeriesSpec`: a column name and an
  ordered list of *sources* to try (Stooq, Yahoo, FRED).  The first source
  that returns data wins.  Raw downloads are cached as CSV in ``data/cache``
  so re-runs are offline-friendly and rate-limit-friendly.
* All series are aligned on the S&P 500 trading calendar (NYSE trading days
  from 1990).  Slow-moving series (weekly Fed balance sheet, policy target,
  CPI) are forward-filled without limit; market series are forward-filled
  for at most ``MARKET_FFILL_LIMIT`` days so a real outage still shows up.
* Coverage is measured *before* forward-filling, so the report shows what
  the sources actually delivered.

Derived columns
---------------
``real_yield``       10y TIPS real yield (FRED DFII10) from 2003-01-02; before
                     that a proxy = nominal 10y (DGS10) minus trailing 12-month
                     CPI inflation.  ``real_yield_source`` labels each row.
``fed_funds_target`` FRED DFEDTAR (single target) to 2008-12-15, then the
                     *upper* bound of the target range (DFEDTARU).
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PARQUET_PATH = DATA_DIR / "daily.parquet"
COVERAGE_PATH = ROOT / "reports" / "data_coverage.md"

START = pd.Timestamp("1990-01-01")
CACHE_MAX_AGE_DAYS = 1
MARKET_FFILL_LIMIT = 5  # trading days
HTTP_TIMEOUT = 60

# --------------------------------------------------------------------------
# Sources: each returns a float Series indexed by Timestamp, or raises.
# --------------------------------------------------------------------------


def fetch_stooq(symbol: str) -> pd.Series:
    """Daily close from Stooq's CSV endpoint (e.g. ``xauusd``)."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    text = requests.get(url, timeout=HTTP_TIMEOUT).text
    if not text.lstrip().startswith("Date,"):
        # Stooq answers with HTML (bot check) or a plain-text error when it
        # won't serve the file; either way it is not a CSV.
        raise RuntimeError(f"Stooq did not return CSV for {symbol}: {text[:80]!r}")
    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
    return df.set_index("Date")["Close"].astype(float).rename(symbol)


def fetch_yahoo(ticker: str) -> pd.Series:
    """Daily close from Yahoo Finance via yfinance (e.g. ``GC=F``)."""
    import yfinance as yf  # imported lazily: slow and noisy at import time

    hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    s = hist["Close"].astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s.rename(ticker)


def fetch_fred(series_id: str) -> pd.Series:
    """Daily/weekly/monthly observations from FRED's public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = requests.get(url, timeout=HTTP_TIMEOUT).text
    if not text.lstrip().startswith("observation_date"):
        raise RuntimeError(f"FRED did not return CSV for {series_id}: {text[:80]!r}")
    df = pd.read_csv(io.StringIO(text), parse_dates=["observation_date"], na_values=["."])
    return df.set_index("observation_date")[series_id].astype(float)


FETCHERS: dict[str, Callable[[str], pd.Series]] = {
    "stooq": fetch_stooq,
    "yahoo": fetch_yahoo,
    "fred": fetch_fred,
}


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def _cache_path(source: str, symbol: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in symbol)
    return CACHE_DIR / f"{source}_{safe}.csv"


def fetch_cached(source: str, symbol: str, refresh: bool = False) -> pd.Series:
    """Fetch ``symbol`` from ``source``, using a CSV cache younger than a day.

    If the download fails but a stale cache exists, the stale cache is used
    and a warning is logged.
    """
    path = _cache_path(source, symbol)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_DAYS * 86400
    if fresh and not refresh:
        return _read_cache(path)
    try:
        s = FETCHERS[source](symbol)
    except Exception as exc:  # network, rate limit, bot check, parse error
        if path.exists():
            log.warning("%s:%s download failed (%s); using stale cache %s", source, symbol, exc, path)
            return _read_cache(path)
        raise
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s.rename("value").to_csv(path, index_label="date")
    return s


def _read_cache(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["value"].astype(float)


# --------------------------------------------------------------------------
# Series catalogue
# --------------------------------------------------------------------------


@dataclass
class SeriesSpec:
    name: str
    sources: list[tuple[str, str]]  # (source, symbol) in priority order
    ffill_limit: int | None = MARKET_FFILL_LIMIT  # None = unlimited (step series)
    note: str = ""
    freq: str = "daily"  # daily | weekly | monthly; drives staleness tolerance and gap reporting
    component: bool = False  # intermediate input to a derived column (not in the final frame)


SERIES: list[SeriesSpec] = [
    SeriesSpec("gold", [("stooq", "xauusd"), ("yahoo", "GC=F")],
               note="Spot USD/oz (Stooq) or COMEX front-month futures close (Yahoo GC=F)."),
    SeriesSpec("silver", [("stooq", "xagusd"), ("yahoo", "SI=F")],
               note="Spot USD/oz (Stooq) or COMEX front-month futures close (Yahoo SI=F)."),
    SeriesSpec("dxy", [("yahoo", "DX-Y.NYB")], note="ICE US Dollar Index close."),
    SeriesSpec("sp500", [("yahoo", "^GSPC")], note="S&P 500 close; defines the trading calendar."),
    SeriesSpec("vix", [("fred", "VIXCLS")], note="CBOE VIX close."),
    SeriesSpec("dfii10", [("fred", "DFII10")], note="10y TIPS constant-maturity real yield, from 2003-01-02."),
    SeriesSpec("dgs10", [("fred", "DGS10")], note="10y nominal Treasury constant-maturity yield."),
    SeriesSpec("cpi", [("fred", "CPIAUCSL")], ffill_limit=None, freq="monthly", component=True,
               note="CPI-U (monthly, SA); only used for the pre-2003 real-yield proxy."),
    SeriesSpec("fed_funds_target_single", [("fred", "DFEDTAR")], ffill_limit=None, component=True,
               note="Fed funds target rate, single value, to 2008-12-15."),
    SeriesSpec("fed_funds_target_upper", [("fred", "DFEDTARU")], ffill_limit=None, component=True,
               note="Fed funds target range upper bound, from 2008-12-16."),
    SeriesSpec("fed_balance_sheet", [("fred", "WALCL")], ffill_limit=None, freq="weekly",
               note="Fed total assets, $ millions, weekly (Wednesday), forward-filled."),
    SeriesSpec("bank_reserves", [("fred", "WRESBAL")], ffill_limit=None, freq="weekly",
               note="Reserve balances with Federal Reserve Banks, $ millions, weekly, forward-filled."),
]

# Columns of the final aligned frame, in order.
COLUMNS = [
    "gold", "silver", "dxy", "real_yield", "real_yield_source", "dfii10", "dgs10",
    "cpi_yoy", "sp500", "vix", "fed_funds_target", "fed_balance_sheet", "bank_reserves",
]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


@dataclass
class Coverage:
    name: str
    source: str
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    n_obs: int
    missing_days: int  # trading days inside [first, last] with no observation
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)
    note: str = ""
    freq: str = "daily"
    component: bool = False


def fetch_series(spec: SeriesSpec, refresh: bool = False) -> tuple[pd.Series, str]:
    """Try each source in order; return (series, 'source:symbol') of the first that works."""
    errors = []
    for source, symbol in spec.sources:
        try:
            s = fetch_cached(source, symbol, refresh=refresh)
            log.info("%s <- %s:%s (%d obs, %s..%s)", spec.name, source, symbol,
                     len(s), s.index.min().date(), s.index.max().date())
            return s.rename(spec.name), f"{source}:{symbol}"
        except Exception as exc:
            log.warning("%s: %s:%s failed: %s", spec.name, source, symbol, exc)
            errors.append(f"{source}:{symbol}: {exc}")
    raise RuntimeError(f"No source worked for {spec.name}:\n  " + "\n  ".join(errors))


def _gaps(missing: pd.Series, min_len: int = 5) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Runs of consecutive True values of length >= min_len, as (start, end, length)."""
    runs = []
    run_start = None
    for date, is_missing in missing.items():
        if is_missing and run_start is None:
            run_start = date
        elif not is_missing and run_start is not None:
            n = missing.loc[run_start:date].sum()
            if n >= min_len:
                runs.append((run_start, missing.loc[run_start:date].index[-2], int(n)))
            run_start = None
    if run_start is not None:
        n = int(missing.loc[run_start:].sum())
        if n >= min_len:
            runs.append((run_start, missing.index[-1], n))
    return runs


def coverage_of(name: str, raw: pd.Series, calendar: pd.DatetimeIndex, source: str, note: str,
                freq: str = "daily", component: bool = False) -> Coverage:
    """Measure coverage of ``raw`` (unfilled).

    Daily series are measured on the trading ``calendar`` (missing days and
    gaps are meaningful).  Weekly/monthly series are measured on their own
    observation dates from the calendar start, where "missing" has no meaning.
    """
    if freq == "daily":
        on_cal = raw.reindex(calendar)
    else:
        on_cal = raw[raw.index >= calendar[0]]
    valid = on_cal.dropna()
    if valid.empty:
        return Coverage(name, source, None, None, 0, 0, [], note, freq, component)
    first, last = valid.index[0], valid.index[-1]
    missing = on_cal.loc[first:last].isna()
    gaps = _gaps(missing) if freq == "daily" else []
    return Coverage(name, source, first, last, len(valid), int(missing.sum()), gaps, note, freq, component)


def build(refresh: bool = False, start: pd.Timestamp = START) -> tuple[pd.DataFrame, list[Coverage]]:
    """Download everything, align on the trading calendar, derive columns.

    Returns the aligned frame and a coverage list (one entry per column).
    """
    raw: dict[str, pd.Series] = {}
    sources: dict[str, str] = {}
    for spec in SERIES:
        s, src = fetch_series(spec, refresh=refresh)
        raw[spec.name] = s[s.index >= start - pd.Timedelta(days=400)]  # keep a runway for CPI lags
        sources[spec.name] = src

    calendar = raw["sp500"].index[raw["sp500"].index >= start]
    df = pd.DataFrame(index=calendar)
    df.index.name = "date"

    specs = {s.name: s for s in SERIES}
    coverage: list[Coverage] = []
    for name, s in raw.items():
        aligned = s.reindex(calendar)
        if name == "cpi":
            continue  # handled below (monthly -> yoy inflation)
        coverage.append(coverage_of(name, s, calendar, sources[name], specs[name].note,
                                    specs[name].freq, specs[name].component))
        df[name] = aligned.ffill(limit=specs[name].ffill_limit)

    # CPI: trailing 12-month inflation (%), treated as known from the first day
    # of the *following* month (CPI for month m is published mid-month m+1).
    cpi = raw["cpi"]
    cpi_yoy = (cpi / cpi.shift(12) - 1.0) * 100.0
    cpi_yoy.index = cpi_yoy.index + pd.offsets.MonthBegin(1)
    df["cpi_yoy"] = cpi_yoy.reindex(calendar.union(cpi_yoy.index)).ffill().reindex(calendar)
    coverage.append(coverage_of("cpi_yoy", cpi_yoy, calendar, sources["cpi"],
                                "Trailing 12-month CPI inflation, %, monthly (effective from the "
                                "following month), forward-filled.", freq="monthly"))

    # Real yield: TIPS where available, otherwise nominal minus trailing inflation.
    proxy = df["dgs10"] - df["cpi_yoy"]
    df["real_yield"] = df["dfii10"].where(df["dfii10"].notna(), proxy)
    df["real_yield_source"] = np.where(df["dfii10"].notna(), "DFII10",
                                       np.where(proxy.notna(), "DGS10-CPI proxy", None))
    coverage.append(coverage_of("real_yield", df["real_yield"], calendar, "derived",
                                "DFII10 where available (2003-), else DGS10 minus trailing 12m CPI inflation."))

    # Fed funds target: single target to 2008-12-15, then the range's upper bound.
    df["fed_funds_target"] = df["fed_funds_target_single"].where(
        df["fed_funds_target_single"].notna(), df["fed_funds_target_upper"])
    coverage.append(coverage_of("fed_funds_target", df["fed_funds_target"], calendar, "derived",
                                "DFEDTAR to 2008-12-15, then DFEDTARU (upper bound of range)."))
    df = df.drop(columns=["fed_funds_target_single", "fed_funds_target_upper"])

    df = df[COLUMNS]
    order = {c: i for i, c in enumerate(COLUMNS)}
    coverage.sort(key=lambda c: order.get(c.name, 99))
    return df, coverage


def coverage_markdown(coverage: list[Coverage], df: pd.DataFrame) -> str:
    lines = [
        "# Data coverage",
        "",
        f"Built {pd.Timestamp.now():%Y-%m-%d}. Calendar: S&P 500 trading days, "
        f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} ({len(df)} days).",
        "",
        "Coverage is measured before forward-filling. *Missing* counts trading days inside a",
        "daily series' own span with no observation (not meaningful for weekly/monthly series).",
        "",
        "| column | source | first | last | obs | missing | note |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for c in coverage:
        first = f"{c.first:%Y-%m-%d}" if c.first is not None else "-"
        last = f"{c.last:%Y-%m-%d}" if c.last is not None else "-"
        missing = str(c.missing_days) if c.freq == "daily" else "-"
        name = f"{c.name} (component)" if c.component else c.name
        lines.append(f"| {name} | {c.source} | {first} | {last} | {c.n_obs} | {missing} | {c.note} |")

    lines += ["", "## Gaps and warnings", ""]
    stale_tolerance = {"daily": 10, "weekly": 14, "monthly": 50}
    warned = False
    for c in coverage:
        if c.component:
            continue  # spliced into a derived column, which is checked instead
        if c.first is None:
            lines.append(f"- **{c.name}: no data at all.**")
            warned = True
            continue
        if (c.first - df.index[0]).days > 31:
            lines.append(f"- **{c.name}** starts {c.first:%Y-%m-%d}, after the {df.index[0]:%Y-%m-%d} start "
                         f"({(c.first - df.index[0]).days / 365.25:.1f} years missing).")
            warned = True
        if (df.index[-1] - c.last).days > stale_tolerance[c.freq]:
            lines.append(f"- **{c.name}** ends {c.last:%Y-%m-%d}; may be stale.")
            warned = True
        for start, end, n in c.gaps:
            lines.append(f"- **{c.name}** gap of {n} trading days: {start:%Y-%m-%d} to {end:%Y-%m-%d}.")
            warned = True
    if not warned:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def load(path: Path = PARQUET_PATH) -> pd.DataFrame:
    """Load the aligned daily frame built by :func:`build`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m gold_events.data` first")
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build data/daily.parquet from free sources.")
    p.add_argument("--refresh", action="store_true", help="ignore the download cache")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df, coverage = build(refresh=args.refresh)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH)
    report = coverage_markdown(coverage, df)
    COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_PATH.write_text(report)
    print(report)
    print(f"Wrote {PARQUET_PATH} ({df.shape[0]} rows x {df.shape[1]} cols) and {COVERAGE_PATH}")


if __name__ == "__main__":
    main()
