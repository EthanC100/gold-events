# gold-events

An event study of how gold and silver react to geopolitical and policy events,
with controls for the dollar and real yields. Python, pandas, free data only.

Status: data pipeline, event-study maths and the first analysis (US midterm
elections) are done. `data/events.csv` is an empty template waiting for events.

## Quick start

```bash
pip install -e ".[dev]"
python -m gold_events.data       # download + cache + align -> data/daily.parquet, reports/data_coverage.md
python -m gold_events.midterms   # first analysis -> reports/midterms.md
pytest                           # tests for the event-study maths
```

The first command takes a minute or two (a dozen HTTP downloads). Raw downloads are
cached in `data/cache/` for a day; pass `--refresh` to force re-downloading.

## Layout

```
gold_events/
  data.py          download, cache, align, coverage report
  event_study.py   raw and net returns around a date (the maths)
  midterms.py      first analysis: every US midterm since 1994
data/
  events.csv       event list (empty template: date, category, event, countries, severity, source)
  daily.parquet    aligned daily frame (built, not committed)
  cache/           raw CSV downloads (built, not committed)
reports/
  data_coverage.md what each source delivered, and the gaps
  midterms.md      the midterm tables and reading
tests/
  test_event_study.py
```

## Modules

### `gold_events.data`

Builds one aligned daily DataFrame from 1990 to today and saves it as parquet.

* **Sources.** Each series has an ordered list of sources to try. Gold and silver
  try Stooq spot (`xauusd`, `xagusd`) first and fall back to Yahoo COMEX front-month
  futures (`GC=F`, `SI=F`). DXY and the S&P 500 come from Yahoo (`DX-Y.NYB`, `^GSPC`).
  Everything else comes from FRED's public CSV endpoint: `VIXCLS`, `DFII10`, `DGS10`,
  `CPIAUCSL`, `DFEDTAR`/`DFEDTARU`, `WALCL`, `WRESBAL`. FRED's LBMA gold and silver
  fixes (`GOLDAMGBD228NLBM` etc.) were discontinued and now return 404, so they are
  not used.
* **Caching.** Each raw download is stored as `data/cache/<source>_<symbol>.csv`
  and reused for 24 hours. If a download fails and a stale cache exists, the stale
  copy is used with a warning.
* **Calendar.** The S&P 500's trading days define the index, so "trading day" in the
  event study means an NYSE trading day. Market series (gold, silver, DXY, yields, VIX)
  are forward-filled for at most 5 days so a real outage still shows as a gap. Step
  series (policy rate, weekly balance sheet and reserves, monthly CPI) are forward-filled
  without limit.
* **Derived columns.**
  * `real_yield`: the 10y TIPS yield (`DFII10`) from 2003-01-02. Before that, a proxy
    equal to the 10y nominal yield (`DGS10`) minus trailing 12-month CPI inflation.
    `real_yield_source` says which one each row uses (`DFII10` or `DGS10-CPI proxy`).
    The CPI for month *m* is treated as known from the first day of month *m+1*
    (it is published around the 10th-15th), so the proxy never uses an unpublished print.
  * `fed_funds_target`: the single target rate (`DFEDTAR`) to 2008-12-15, then the
    upper bound of the target range (`DFEDTARU`).
* **Coverage report.** `reports/data_coverage.md` lists, per column, the source that won,
  first/last date, observation count, missing trading days inside the span, and any gap
  of 5 or more consecutive trading days. Series that start after 1990 or end more than a
  few days before today are called out.

Columns of `data/daily.parquet`: `gold`, `silver`, `dxy`, `real_yield`,
`real_yield_source`, `dfii10`, `dgs10`, `cpi_yoy`, `sp500`, `vix`, `fed_funds_target`,
`fed_balance_sheet` ($ millions), `bank_reserves` ($ millions).

### `gold_events.event_study`

The maths, kept separate from data access so it is easy to test.

* **Day 0** is chosen by the caller. Two helpers: `first_trading_day_after(index, date)`
  (used for elections, whose results arrive overnight) and
  `first_trading_day_on_or_after(index, date)` (for events that hit during a trading day).
* **Raw return** at horizon *h* is the log return from the close of day -1 (the last
  close before the event is known) to the close of day *h-1*, so it spans exactly *h*
  trading days starting with day 0:

  `raw_h = ln P[day h-1] - ln P[day -1]`

* **Betas** come from an OLS with intercept of the asset's daily log return on the daily
  change in ln(DXY) and the daily change in the real yield (percentage points), over the
  250 trading days ending at day -1. Nothing after day -1 is used. If fewer than 200
  valid days survive NaN-dropping, the betas (and therefore the net return) are NaN.
* **Net return** removes what the controls explain over the same window. The intercept
  is not subtracted: it is the asset's drift, not something the dollar or yields explain.

  `net_h = raw_h - beta_dxy * dln(DXY)_h - beta_ry * d(real_yield)_h`

`event_study(df, day0)` returns an `EventStudyResult` whose `.table` has one row per
(asset, horizon) with `raw`, `net`, `explained`, the control changes, the betas and the
number of observations behind them. Everything is in log-return units.

### `gold_events.midterms`

Runs the event study on the eight midterms since 1994 with day 0 = the Wednesday after
the election, writes per-election tables for gold and silver, averages split by whether
the House changed hands (1994, 2006, 2010, 2018, 2022 yes; 1998, 2002, 2014 no), an
attribution table showing what the controls explained at 20 days, and a one-paragraph
reading generated from the numbers. Output: `reports/midterms.md`.

## Data caveats

* **Gold and silver before August 2000.** Stooq was not reachable from the environment
  where this repository was built (it answers with a JavaScript bot check), so the
  committed `reports/` were produced from Yahoo futures, which start 2000-08-30. That
  leaves the 1994 and 1998 midterms without price data; they show as `n/a`. Re-running
  `python -m gold_events.data` from a network where Stooq responds should extend both
  series back to 1990 and fill those rows in. Other free sources were tried and did not
  work: FRED's LBMA series (discontinued, 404), LBMA's own price feed and Nasdaq Data
  Link (both behind bot protection without an account).
* **Futures vs spot.** Yahoo `GC=F`/`SI=F` are continuous front-month futures. They can
  carry small jumps on contract rolls and occasional odd prints, and they trade on a few
  days NYSE is closed (dropped by the alignment). Stooq spot is preferred when available.
* **Real yield before 2003** is a proxy (nominal minus trailing inflation), which is a
  backward-looking measure and much smoother than market-implied real yields. Betas
  estimated on it for 2002 and earlier are correspondingly weaker.
* **Weekly series** (`fed_balance_sheet`, `bank_reserves`) are Wednesday values
  forward-filled; the coverage report marks these as not having a meaningful "missing" count.

## First result: midterms

See `reports/midterms.md` for the full tables. Headline from the six midterms with data
(2002-2022): gold's average move over the 20 trading days after a midterm was about +2.3%
raw and +1.8% net of the dollar and real yields; silver about +5% raw and +4.4% net. The
first-day reaction averaged slightly negative for both. Splitting by whether the House
changed hands produces groups of four and two, and the 2010 silver move (+21% net) alone
dominates the "changed hands" average, so no conclusion about control of the House
survives the sample size.

## `data/events.csv`

Header only for now: `date, category, event, countries, severity, source`. Suggested
conventions: ISO dates, `countries` as a semicolon-separated list of ISO codes,
`severity` as an integer scale you define once and stick to.

## Tests

`tests/test_event_study.py` builds synthetic prices where the asset is exactly
beta times the controls plus known idiosyncratic shocks, and checks that: betas are
recovered exactly; raw returns span the right days; the net return equals the planted
shock; a shock on day 0 does not leak into the betas; windows are honoured; NaNs are
dropped rather than propagated; and day-0 helpers behave around weekends.
