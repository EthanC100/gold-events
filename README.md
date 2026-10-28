# gold-events

An event study of how gold and silver react to geopolitical and policy events,
with controls for the dollar and real yields. Python, pandas, free data only.

Status: data pipeline, event-study maths, the midterm analysis, a 68-event list with
hand and measured severity, and a first (v1) regression model fitted on 2000-2021.

## Quick start

```bash
pip install -e ".[dev]"
python -m gold_events.data       # download + cache + align -> data/daily.parquet, reports/data_coverage.md
python -m gold_events.midterms   # first analysis -> reports/midterms.md
python -m gold_events.events     # measured severity -> columns added in place to data/events.csv
python -m gold_events.mechanical # index-flagged events -> data/events_mechanical.csv, reports/events_mechanical.md
python -m gold_events.model      # v1 fit + refit on 2000-08-30..2021-12-31 -> reports/v1_fit.md
pytest                           # tests for the maths (event study, scoring, model)
```

The first command takes a minute or two (a dozen HTTP downloads). Raw downloads are
cached in `data/cache/` for a day; pass `--refresh` to force re-downloading.

## Layout

```
gold_events/
  data.py          download, cache, align, coverage report
  event_study.py   raw and net returns around a date (the maths)
  midterms.py      first analysis: every US midterm since 1994
  events.py        event list + measured severity from GPR, EPU and consumer sentiment
  mechanical.py    mechanical event list: days when GPR or EPU is in its top 5%, merged into runs
  model.py         v1 regression: returns on controls + severity x event terms, and the refit
data/
  events.csv       68 hand-curated events with hand and measured severity columns
  events_mechanical.csv  202 index-flagged events 1990-2021, category left blank
  daily.parquet    aligned daily frame (built, not committed)
  cache/           raw downloads (built, not committed)
reports/
  data_coverage.md what each source delivered, and the gaps
  midterms.md      the midterm tables and reading
  events_mechanical.md  mechanical list vs hand list: overlap, misses, largest unlisted spikes
  v1_fit.md        v1 model coefficients, errors against baselines, refit comparison, readings
tests/
  test_event_study.py, test_events.py, test_mechanical.py, test_model.py
```

## Modules

### `gold_events.data`

Builds one aligned daily DataFrame from 1990 to today and saves it as parquet.

* **Sources.** Each series has an ordered list of sources to try. Gold and silver
  try Stooq spot (`xauusd`, `xagusd`) first and fall back to Yahoo COMEX front-month
  futures (`GC=F`, `SI=F`). DXY and the S&P 500 come from Yahoo (`DX-Y.NYB`, `^GSPC`).
  Macro series come from FRED's public CSV endpoint: `VIXCLS`, `DFII10`, `DGS10`,
  `CPIAUCSL`, `DFEDTAR`/`DFEDTARU`, `WALCL`, `WRESBAL`, `UMCSENT`. The daily
  Geopolitical Risk index (Caldara-Iacoviello, `GPRD`) comes from matteoiacoviello.com
  and the daily US Economic Policy Uncertainty index from policyuncertainty.com; both are
  defined on every calendar day and the aligned frame keeps the trading-day value. FRED's
  LBMA gold and silver fixes (`GOLDAMGBD228NLBM` etc.) were discontinued and now return
  404, so they are not used.
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
`fed_balance_sheet` ($ millions), `bank_reserves` ($ millions), `gpr`, `epu`, `umcsent`.

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

### `gold_events.events`

Reads the hand columns of `data/events.csv` and writes the measured columns back in place
(re-running overwrites only the measured columns).

* **GPR and EPU jump.** For each event, the mean of the index over days 0..4 minus the
  mean over days -5..-1, where "day" is a calendar day because both indices exist on
  every day, and day 0 is the event date itself (or the next available day).
* **Severity.** The same before/after jump is computed for every day from 1990 to 2021 as
  a reference distribution. `gpr_z` and `epu_z` are the event's jumps as z-scores against
  it, and `severity_measured` is the larger of the two (continuous, negative when both
  indices fell into the event). `gpr_score`, `epu_score` and `severity_quintile` are the
  1-5 quintile version, kept for reference.
* **Consumer sentiment.** Michigan sentiment in the calendar month after the event minus
  the month before, left blank when the month after is past December 2021 (the repository
  stores nothing after 2021).
* **Clustering flag.** `near_other_event` is true when another event's day 0 is within 20
  trading days; `nearest_event_days` gives the distance.

### `gold_events.mechanical`

A second event list that needs no judgement. Over trading days 1990-2021, a day is flagged
when GPR or EPU exceeds its own 95th percentile; flagged days within 5 trading days of the
previous flagged day are merged into one event dated at its first flagged day, keeping the
peak of each index, which index triggered it, and `peak_z` (the larger peak as a z-score of
daily levels) for ranking. `severity_measured` is the same jump z-score as the hand list so
the two are comparable in the model. The report compares the two lists: overlap within 5
trading days of a start, overlap anywhere inside a mechanical event's span, hand-list events
the indices never flagged, and the ten largest mechanical events absent from the hand list.
The merge rule turns long crises into single onset-dated events (the Iraq build-up runs from
2002-12-18 to 2003-05-22), which is why the span-based overlap is reported too.

### `gold_events.model`

The v1 regression, fitted strictly on windows that end on or before 2021-12-31 (the data
frame is truncated before anything is computed; tests check the cut).

* **Observations.** One per trading day t: the h-day log return from the close of t-1 to
  the close of t+h-1, for h = 5 and 20, so the rows are overlapping windows.
* **Regressors over the same window.** Change in the 10y real yield, change in log DXY,
  change in log VIX, change in the daily GPR index, and one term per event category equal
  to severity x 1[an event of that category has its day 0 at t]. The event term is
  therefore non-zero only on the row that *is* that event's window.
* **Estimator.** Ridge on the event terms only, with the controls unpenalised; the penalty
  is picked by 5-fold blocked (contiguous in time) cross-validation on the standardised
  design. Standard errors are Newey-West with lag h in a sandwich around the ridge
  estimator, back-transformed to the natural scale. The unpenalised OLS on the same
  design is reported alongside.
* **Baselines.** In-sample RMSE of no-change (predict zero) and controls-only OLS, on all
  windows and on event windows alone, for both severity versions.
* **Refit.** The second half of the report repeats the exercise with the continuous
  severity three ways (hand list by category, hand list pooled into one "any event x
  severity" term, mechanical list pooled), compares cross-validated RMSE against the
  controls-only baseline on all windows and on each list's event windows, reports the mean
  out-of-fold abnormal return on each list's windows as a selection check, and refits the
  hand pooled term without September 11, whose severity (z = 19) dominates the scale.

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
* **Severity scales.** The quintile version (`severity_quintile`) is compressed: 46 of the
  68 events score 5. The continuous version (`severity_measured`) is dominated by September
  11 (z = 19, several times any other event), so a term using it leans on that one row;
  winsorising or logging it is a v2 item.
* **Nothing after 2021 is stored.** `sentiment_after` is blank where the month after the
  event is past December 2021, and the model never reads data past 2021-12-31.

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

`tests/test_events.py` checks the before/after windows, the z-score and quintile scoring,
weekend events and the clustering flag on planted jumps. `tests/test_mechanical.py` checks
the per-index thresholds and the run merging on planted spikes. `tests/test_model.py` checks
that the design rows equal the event-study returns, that events land on the right row (by
category and pooled), that the sample end is a hard cut, that ridge at zero penalty equals
OLS and at a huge penalty reduces to controls-only OLS, that out-of-fold residuals are real,
and the HAC covariance at lag 0.

`tests/test_event_study.py` builds synthetic prices where the asset is exactly
beta times the controls plus known idiosyncratic shocks, and checks that: betas are
recovered exactly; raw returns span the right days; the net return equals the planted
shock; a shock on day 0 does not leak into the betas; windows are honoured; NaNs are
dropped rather than propagated; and day-0 helpers behave around weekends.
