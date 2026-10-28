# Data coverage

Built 2026-10-28. Calendar: S&P 500 trading days, 1990-01-02 to 2026-10-27 (9273 days).

Coverage is measured before forward-filling. *Missing* counts trading days inside a
daily series' own span with no observation (not meaningful for weekly/monthly series).

| column | source | first | last | obs | missing | note |
|---|---|---|---|---:|---:|---|
| gold | yahoo:GC=F | 2000-08-30 | 2026-10-27 | 6559 | 19 | Spot USD/oz (Stooq) or COMEX front-month futures close (Yahoo GC=F). |
| silver | yahoo:SI=F | 2000-08-30 | 2026-10-27 | 6560 | 18 | Spot USD/oz (Stooq) or COMEX front-month futures close (Yahoo SI=F). |
| dxy | yahoo:DX-Y.NYB | 1990-01-02 | 2026-10-27 | 9271 | 2 | ICE US Dollar Index close. |
| real_yield | derived | 1990-01-02 | 2026-10-27 | 9273 | 0 | DFII10 where available (2003-), else DGS10 minus trailing 12m CPI inflation. |
| dfii10 | fred:DFII10 | 2003-01-02 | 2026-10-26 | 5948 | 44 | 10y TIPS constant-maturity real yield, from 2003-01-02. |
| dgs10 | fred:DGS10 | 1990-01-02 | 2026-10-26 | 9196 | 76 | 10y nominal Treasury constant-maturity yield. |
| cpi_yoy | fred:CPIAUCSL | 1990-02-01 | 2026-10-01 | 440 | - | Trailing 12-month CPI inflation, %, monthly (effective from the following month), forward-filled. |
| sp500 | yahoo:^GSPC | 1990-01-02 | 2026-10-27 | 9273 | 0 | S&P 500 close; defines the trading calendar. |
| vix | fred:VIXCLS | 1990-01-02 | 2026-10-26 | 9269 | 3 | CBOE VIX close. |
| fed_funds_target | derived | 1990-01-02 | 2026-10-27 | 9273 | 0 | DFEDTAR to 2008-12-15, then DFEDTARU (upper bound of range). |
| fed_balance_sheet | fred:WALCL | 2002-12-18 | 2026-10-21 | 1245 | - | Fed total assets, $ millions, weekly (Wednesday), forward-filled. |
| bank_reserves | fred:WRESBAL | 2002-12-18 | 2026-10-21 | 1245 | - | Reserve balances with Federal Reserve Banks, $ millions, weekly, forward-filled. |
| gpr | gpr:GPRD | 1990-01-02 | 2026-10-26 | 9272 | 0 | Daily Geopolitical Risk index (Caldara-Iacoviello, 1985-2019=100); value on the trading day. |
| epu | epu:daily_policy_index | 1990-01-02 | 2026-10-27 | 9273 | 0 | Daily US Economic Policy Uncertainty index (Baker-Bloom-Davis); value on the trading day. |
| umcsent | fred:UMCSENT | 1990-02-01 | 2026-09-01 | 440 | - | University of Michigan consumer sentiment, monthly, forward-filled. |
| fed_funds_target_single (component) | fred:DFEDTAR | 1990-01-02 | 2008-12-15 | 4780 | 0 | Fed funds target rate, single value, to 2008-12-15. |
| fed_funds_target_upper (component) | fred:DFEDTARU | 2008-12-16 | 2026-10-27 | 4493 | 0 | Fed funds target range upper bound, from 2008-12-16. |

## Gaps and warnings

- **gold** starts 2000-08-30, after the 1990-01-02 start (10.7 years missing).
- **silver** starts 2000-08-30, after the 1990-01-02 start (10.7 years missing).
- **dfii10** starts 2003-01-02, after the 1990-01-02 start (13.0 years missing).
- **fed_balance_sheet** starts 2002-12-18, after the 1990-01-02 start (13.0 years missing).
- **bank_reserves** starts 2002-12-18, after the 1990-01-02 start (13.0 years missing).
