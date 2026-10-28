# v1 model fit: gold and silver returns on controls and event terms

Built 2026-10-28. Sample: windows starting after 2000-08-30 and ending on or before 2021-12-31 (the data frame is cut at 2021-12-31 before anything is computed). Last price used: 2021-12-31.

## Setup

For every trading day t, y = log return from the close of t-1 to the close of t+h-1 (h = 5, 20).
Regressors over the same window: change in the 10y real yield, change in log DXY, change in log VIX,
change in the daily GPR index, and one term per event category equal to severity x 1[an event of that
category has its day 0 at t]. Day 0 = the event date if it is a trading day, else the next one.
Severity is either the hand score (1-5) or the measured score (max of the GPR and EPU jump quintiles).

Fit: ridge on the event terms only (controls unpenalised), penalty chosen by 5-fold blocked
cross-validation; Newey-West standard errors (lag h) because the windows overlap. Coefficients are in
percent of return. *Event windows* in the error tables are the rows whose window starts at an event's
day 0, i.e. the rows where the event terms can make any difference. A category term is identified by
only as many rows as it has events, so its standard error and t-statistic are unreliable for categories
with fewer than about five events, whatever the number says.

## Events in sample

50 of 68 events fall in the sample. By category:

| category | events in sample | mean severity_hand | mean severity_measured |
|---|---:|---:|---:|
| Banking or financial stress | 7 | 3.1 | 4.3 |
| Central bank policy shock | 16 | 3.2 | 3.8 |
| Election or leadership shock | 5 | 3.2 | 4.8 |
| Military escalation | 9 | 2.9 | 4.6 |
| Pandemic or disaster | 2 | 4.0 | 5.0 |
| Reserve-currency and gold-system event | 0 | - | - |
| Sanctions and trade war | 4 | 3.0 | 4.5 |
| Sovereign debt crisis | 4 | 3.2 | 5.0 |
| Terror attack | 3 | 3.0 | 5.0 |

Categories with no event in the sample are dropped from the fit: Reserve-currency and gold-system event.

13 in-sample events are within 20 trading days of another event (2001-09-11, 2001-10-08, 2008-09-08, 2008-09-15, 2008-10-08, 2011-02-11, 2011-03-11, 2011-03-21, 2019-07-31, 2019-08-05, 2020-03-03, 2020-03-11, 2020-03-16); their 20-day windows overlap, so their category terms share the same returns.

## Gold

### Gold, 5-day windows

Cross-validated ridge penalty on the event terms (standardised units): hand 1e+07 (top of the grid: event terms shrunk to ~0); measured 1e+07 (top of the grid: event terms shrunk to ~0). CV MSE (units 1e-4) at lambda = 0 vs chosen: hand 4.802 vs 4.775; measured 4.802 vs 4.775.

Ridge fit (what cross-validation keeps):

| term | units | hand: coef | SE | t | measured: coef | SE | t |
|---|---|---:|---:|---:|---:|---:|---:|
| d_real_yield | per +1pp real yield | -2.97 | 0.60 | -5.0 | -2.97 | 0.60 | -5.0 |
| dlog_dxy | per +1% DXY | -0.98 | 0.06 | -15.3 | -0.98 | 0.06 | -15.3 |
| dlog_vix | per +1% VIX | -0.00 | 0.00 | -0.2 | -0.00 | 0.00 | -0.2 |
| d_gpr | per +100 GPR points | +0.19 | 0.07 | +2.8 | +0.19 | 0.07 | +2.8 |
| Banking or financial stress | % per severity point | +0.00 | 0.00 | +0.3 | +0.00 | 0.00 | +0.3 |
| Central bank policy shock | % per severity point | +0.00 | 0.00 | +0.4 | +0.00 | 0.00 | +0.2 |
| Election or leadership shock | % per severity point | +0.00 | 0.00 | +0.2 | -0.00 | 0.00 | -0.0 |
| Military escalation | % per severity point | -0.00 | 0.00 | -0.4 | +0.00 | 0.00 | +0.3 |
| Pandemic or disaster | % per severity point | -0.00 | 0.00 | -1.3 | -0.00 | 0.00 | -1.4 |
| Sanctions and trade war | % per severity point | +0.00 | 0.00 | +1.3 | +0.00 | 0.00 | +1.2 |
| Sovereign debt crisis | % per severity point | +0.00 | 0.00 | +1.4 | +0.00 | 0.00 | +1.4 |
| Terror attack | % per severity point | +0.00 | 0.00 | +1.0 | +0.00 | 0.00 | +0.9 |

Unpenalised event terms (plain OLS on the same design, for comparison):

| category (unpenalised) | hand: % per point | SE | t | measured: % per point | SE | t |
|---|---:|---:|---:|---:|---:|---:|
| Banking or financial stress | +0.25 | 0.73 | +0.3 | +0.14 | 0.46 | +0.3 |
| Central bank policy shock | +0.11 | 0.25 | +0.4 | +0.05 | 0.19 | +0.3 |
| Election or leadership shock | +0.13 | 0.52 | +0.3 | -0.00 | 0.35 | -0.0 |
| Military escalation | -0.11 | 0.25 | -0.4 | +0.06 | 0.17 | +0.3 |
| Pandemic or disaster | -0.75 | 0.08 | -9.6 | -0.61 | 0.08 | -7.6 |
| Sanctions and trade war | +0.26 | 0.15 | +1.7 | +0.17 | 0.11 | +1.6 |
| Sovereign debt crisis | +0.78 | 0.33 | +2.4 | +0.49 | 0.24 | +2.0 |
| Terror attack | +0.57 | 0.21 | +2.7 | +0.23 | 0.23 | +1.0 |

In-sample error:

| model | RMSE all windows (n=5364) | RMSE event windows (n=50) |
|---|---:|---:|
| no change (predict 0) | 2.46% | 3.73% |
| controls only (OLS) | 2.13% | 3.37% |
| controls + events (hand), ridge | 2.13% | 3.37% |
| controls + events (hand), unpenalised | 2.12% | 3.17% |
| controls + events (measured), ridge | 2.13% | 3.37% |
| controls + events (measured), unpenalised | 2.12% | 3.21% |

- |t| >= 2, hand severity, ridge: none.
- |t| >= 2, hand severity, unpenalised: Pandemic or disaster (-0.75% per point, t=-9.6); Sovereign debt crisis (+0.78% per point, t=+2.4); Terror attack (+0.57% per point, t=+2.7).
- |t| >= 2, measured severity, ridge: none.
- |t| >= 2, measured severity, unpenalised: Pandemic or disaster (-0.61% per point, t=-7.6); Sovereign debt crisis (+0.49% per point, t=+2.0).

### Gold, 20-day windows

Cross-validated ridge penalty on the event terms (standardised units): hand 1e+07 (top of the grid: event terms shrunk to ~0); measured 1e+07 (top of the grid: event terms shrunk to ~0). CV MSE (units 1e-4) at lambda = 0 vs chosen: hand 17.807 vs 17.452; measured 17.668 vs 17.452.

Ridge fit (what cross-validation keeps):

| term | units | hand: coef | SE | t | measured: coef | SE | t |
|---|---|---:|---:|---:|---:|---:|---:|
| d_real_yield | per +1pp real yield | -4.30 | 0.95 | -4.5 | -4.30 | 0.95 | -4.5 |
| dlog_dxy | per +1% DXY | -0.85 | 0.11 | -7.8 | -0.85 | 0.11 | -7.8 |
| dlog_vix | per +1% VIX | +0.01 | 0.01 | +1.0 | +0.01 | 0.01 | +1.0 |
| d_gpr | per +100 GPR points | +0.32 | 0.13 | +2.4 | +0.32 | 0.13 | +2.4 |
| Banking or financial stress | % per severity point | +0.00 | 0.00 | +0.6 | +0.00 | 0.00 | +0.3 |
| Central bank policy shock | % per severity point | +0.00 | 0.00 | +0.1 | -0.00 | 0.00 | -0.0 |
| Election or leadership shock | % per severity point | -0.00 | 0.00 | -0.6 | -0.00 | 0.00 | -0.8 |
| Military escalation | % per severity point | -0.00 | 0.00 | -0.6 | +0.00 | 0.00 | +0.1 |
| Pandemic or disaster | % per severity point | +0.00 | 0.00 | +1.3 | +0.00 | 0.00 | +1.4 |
| Sanctions and trade war | % per severity point | +0.00 | 0.00 | +1.2 | +0.00 | 0.00 | +1.1 |
| Sovereign debt crisis | % per severity point | +0.00 | 0.00 | +0.9 | +0.00 | 0.00 | +0.8 |
| Terror attack | % per severity point | +0.00 | 0.00 | +1.5 | +0.00 | 0.00 | +1.3 |

Unpenalised event terms (plain OLS on the same design, for comparison):

| category (unpenalised) | hand: % per point | SE | t | measured: % per point | SE | t |
|---|---:|---:|---:|---:|---:|---:|
| Banking or financial stress | +1.05 | 1.28 | +0.8 | +0.30 | 1.13 | +0.3 |
| Central bank policy shock | +0.07 | 0.49 | +0.1 | -0.00 | 0.35 | -0.0 |
| Election or leadership shock | -0.42 | 0.77 | -0.5 | -0.38 | 0.48 | -0.8 |
| Military escalation | -0.24 | 0.36 | -0.7 | +0.03 | 0.28 | +0.1 |
| Pandemic or disaster | +0.42 | 0.08 | +5.4 | +0.35 | 0.06 | +6.1 |
| Sanctions and trade war | +0.46 | 0.30 | +1.5 | +0.30 | 0.22 | +1.4 |
| Sovereign debt crisis | +1.09 | 0.95 | +1.1 | +0.60 | 0.65 | +0.9 |
| Terror attack | +0.67 | 0.42 | +1.6 | +0.56 | 0.28 | +2.0 |

In-sample error:

| model | RMSE all windows (n=5349) | RMSE event windows (n=49) |
|---|---:|---:|
| no change (predict 0) | 4.72% | 5.89% |
| controls only (OLS) | 4.01% | 6.04% |
| controls + events (hand), ridge | 4.01% | 6.04% |
| controls + events (hand), unpenalised | 4.01% | 5.74% |
| controls + events (measured), ridge | 4.01% | 6.04% |
| controls + events (measured), unpenalised | 4.01% | 5.87% |

- |t| >= 2, hand severity, ridge: none.
- |t| >= 2, hand severity, unpenalised: Pandemic or disaster (+0.42% per point, t=+5.4).
- |t| >= 2, measured severity, ridge: none.
- |t| >= 2, measured severity, unpenalised: Pandemic or disaster (+0.35% per point, t=+6.1); Terror attack (+0.56% per point, t=+2.0).

## Silver

### Silver, 5-day windows

Cross-validated ridge penalty on the event terms (standardised units): hand 251; measured 6.31e+03. CV MSE (units 1e-4) at lambda = 0 vs chosen: hand 15.731 vs 15.730; measured 15.800 vs 15.739.

Ridge fit (what cross-validation keeps):

| term | units | hand: coef | SE | t | measured: coef | SE | t |
|---|---|---:|---:|---:|---:|---:|---:|
| d_real_yield | per +1pp real yield | -3.53 | 1.01 | -3.5 | -3.63 | 1.04 | -3.5 |
| dlog_dxy | per +1% DXY | -1.60 | 0.10 | -15.3 | -1.60 | 0.11 | -15.2 |
| dlog_vix | per +1% VIX | -0.05 | 0.01 | -7.1 | -0.05 | 0.01 | -7.0 |
| d_gpr | per +100 GPR points | +0.34 | 0.11 | +3.0 | +0.36 | 0.12 | +3.1 |
| Banking or financial stress | % per severity point | -0.62 | 1.23 | -0.5 | -0.24 | 0.38 | -0.6 |
| Central bank policy shock | % per severity point | +0.02 | 0.45 | +0.0 | -0.05 | 0.14 | -0.3 |
| Election or leadership shock | % per severity point | +0.25 | 0.82 | +0.3 | +0.04 | 0.26 | +0.1 |
| Military escalation | % per severity point | -0.12 | 0.32 | -0.4 | +0.05 | 0.11 | +0.5 |
| Pandemic or disaster | % per severity point | -3.32 | 0.74 | -4.5 | -1.16 | 0.68 | -1.7 |
| Sanctions and trade war | % per severity point | +0.36 | 0.16 | +2.3 | +0.12 | 0.06 | +1.9 |
| Sovereign debt crisis | % per severity point | +0.85 | 0.33 | +2.6 | +0.28 | 0.13 | +2.2 |
| Terror attack | % per severity point | +0.76 | 0.33 | +2.3 | +0.13 | 0.15 | +0.9 |

Unpenalised event terms (plain OLS on the same design, for comparison):

| category (unpenalised) | hand: % per point | SE | t | measured: % per point | SE | t |
|---|---:|---:|---:|---:|---:|---:|
| Banking or financial stress | -0.65 | 1.29 | -0.5 | -0.53 | 0.81 | -0.7 |
| Central bank policy shock | +0.02 | 0.47 | +0.0 | -0.10 | 0.31 | -0.3 |
| Election or leadership shock | +0.26 | 0.86 | +0.3 | +0.08 | 0.56 | +0.1 |
| Military escalation | -0.13 | 0.33 | -0.4 | +0.11 | 0.24 | +0.5 |
| Pandemic or disaster | -3.47 | 0.73 | -4.8 | -2.53 | 1.15 | -2.2 |
| Sanctions and trade war | +0.38 | 0.17 | +2.3 | +0.26 | 0.11 | +2.3 |
| Sovereign debt crisis | +0.89 | 0.35 | +2.5 | +0.61 | 0.22 | +2.8 |
| Terror attack | +0.79 | 0.33 | +2.4 | +0.28 | 0.33 | +0.9 |

In-sample error:

| model | RMSE all windows (n=5364) | RMSE event windows (n=50) |
|---|---:|---:|
| no change (predict 0) | 4.40% | 7.52% |
| controls only (OLS) | 3.89% | 6.36% |
| controls + events (hand), ridge | 3.87% | 5.52% |
| controls + events (hand), unpenalised | 3.87% | 5.52% |
| controls + events (measured), ridge | 3.88% | 5.89% |
| controls + events (measured), unpenalised | 3.88% | 5.68% |

- |t| >= 2, hand severity, ridge: Pandemic or disaster (-3.32% per point, t=-4.5); Sanctions and trade war (+0.36% per point, t=+2.3); Sovereign debt crisis (+0.85% per point, t=+2.6); Terror attack (+0.76% per point, t=+2.3).
- |t| >= 2, hand severity, unpenalised: Pandemic or disaster (-3.47% per point, t=-4.8); Sanctions and trade war (+0.38% per point, t=+2.3); Sovereign debt crisis (+0.89% per point, t=+2.5); Terror attack (+0.79% per point, t=+2.4).
- |t| >= 2, measured severity, ridge: Sovereign debt crisis (+0.28% per point, t=+2.2).
- |t| >= 2, measured severity, unpenalised: Pandemic or disaster (-2.53% per point, t=-2.2); Sanctions and trade war (+0.26% per point, t=+2.3); Sovereign debt crisis (+0.61% per point, t=+2.8).

### Silver, 20-day windows

Cross-validated ridge penalty on the event terms (standardised units): hand 1e+07 (top of the grid: event terms shrunk to ~0); measured 1e+07 (top of the grid: event terms shrunk to ~0). CV MSE (units 1e-4) at lambda = 0 vs chosen: hand 61.058 vs 60.585; measured 60.876 vs 60.585.

Ridge fit (what cross-validation keeps):

| term | units | hand: coef | SE | t | measured: coef | SE | t |
|---|---|---:|---:|---:|---:|---:|---:|
| d_real_yield | per +1pp real yield | -4.70 | 1.63 | -2.9 | -4.70 | 1.63 | -2.9 |
| dlog_dxy | per +1% DXY | -1.46 | 0.20 | -7.2 | -1.46 | 0.20 | -7.2 |
| dlog_vix | per +1% VIX | -0.06 | 0.02 | -3.8 | -0.06 | 0.02 | -3.8 |
| d_gpr | per +100 GPR points | +1.06 | 0.22 | +4.9 | +1.06 | 0.22 | +4.9 |
| Banking or financial stress | % per severity point | -0.00 | 0.00 | -0.3 | -0.00 | 0.00 | -0.6 |
| Central bank policy shock | % per severity point | -0.00 | 0.00 | -0.4 | -0.00 | 0.00 | -1.0 |
| Election or leadership shock | % per severity point | +0.00 | 0.00 | +0.7 | +0.00 | 0.00 | +0.7 |
| Military escalation | % per severity point | -0.00 | 0.00 | -0.1 | +0.00 | 0.00 | +0.7 |
| Pandemic or disaster | % per severity point | -0.00 | 0.00 | -0.1 | +0.00 | 0.00 | +0.2 |
| Sanctions and trade war | % per severity point | +0.00 | 0.00 | +1.0 | +0.00 | 0.00 | +0.9 |
| Sovereign debt crisis | % per severity point | +0.00 | 0.00 | +0.7 | +0.00 | 0.00 | +0.6 |
| Terror attack | % per severity point | +0.00 | 0.00 | +1.1 | +0.00 | 0.00 | +1.1 |

Unpenalised event terms (plain OLS on the same design, for comparison):

| category (unpenalised) | hand: % per point | SE | t | measured: % per point | SE | t |
|---|---:|---:|---:|---:|---:|---:|
| Banking or financial stress | -0.39 | 1.60 | -0.2 | -0.66 | 1.19 | -0.6 |
| Central bank policy shock | -0.26 | 0.61 | -0.4 | -0.47 | 0.41 | -1.1 |
| Election or leadership shock | +1.04 | 1.31 | +0.8 | +0.66 | 0.88 | +0.7 |
| Military escalation | -0.09 | 0.63 | -0.1 | +0.36 | 0.49 | +0.7 |
| Pandemic or disaster | -0.12 | 1.05 | -0.1 | +0.24 | 0.96 | +0.2 |
| Sanctions and trade war | +1.08 | 0.94 | +1.1 | +0.68 | 0.68 | +1.0 |
| Sovereign debt crisis | +0.90 | 1.04 | +0.9 | +0.48 | 0.71 | +0.7 |
| Terror attack | +0.80 | 0.91 | +0.9 | +0.86 | 0.66 | +1.3 |

In-sample error:

| model | RMSE all windows (n=5349) | RMSE event windows (n=49) |
|---|---:|---:|
| no change (predict 0) | 8.62% | 8.99% |
| controls only (OLS) | 7.61% | 8.71% |
| controls + events (hand), ridge | 7.61% | 8.71% |
| controls + events (hand), unpenalised | 7.60% | 8.50% |
| controls + events (measured), ridge | 7.61% | 8.71% |
| controls + events (measured), unpenalised | 7.60% | 8.34% |

- |t| >= 2, hand severity, ridge: none.
- |t| >= 2, hand severity, unpenalised: none.
- |t| >= 2, measured severity, ridge: none.
- |t| >= 2, measured severity, unpenalised: none.

## Reading

The controls carry the model. Over 20-day windows gold moves about -4.3% per percentage point of real yield and -0.85% per 1% move in the dollar (5-day: -3.0% and -0.98%), with the expected signs, t-statistics well beyond 2, and no dependence on which severity is used. A rise in the GPR index over the window is also associated with higher gold and silver, after the dollar and real yields are taken into account. On gold's event windows the controls alone cut the 5-day RMSE from 3.73% to 3.37% and raise the 20-day RMSE from 5.89% to 6.04%; unpenalised event terms take those to 3.17% and 5.74%, while the cross-validated ridge versions sit at 3.37% and 6.04%. On silver's event windows the controls alone cut the 5-day RMSE from 7.52% to 6.36% and cut the 20-day RMSE from 8.99% to 8.71%; unpenalised event terms take those to 5.52% and 8.50%, while the cross-validated ridge versions sit at 5.52% and 8.71%. Cross-validation is the key result: in 6 of the 8 fits it pushes the event penalty to the top of the grid, i.e. it finds that category effects fitted on one stretch of the sample do not help predict returns on another stretch, so the event terms are shrunk to nothing. The unpenalised fits do show a few categories with |t| >= 2, but none keeps the same sign and |t| >= 2 for gold across both horizons and both severity versions, and none does for silver. Gold's Pandemic or disaster term even flips sign between 5 and 20 days. The terms that do appear rest on two to four events each (the pandemic category is Fukushima and Covid, nothing else), and their t-statistics come from a handful of rows, so they describe those episodes rather than estimate a category effect. Measured severity adds little beyond an on/off indicator: 46 of 68 events score 5 on it, because the larger of two quintile ranks against all days is almost always high for events chosen for being notable. The honest summary for v1 is that gold's and silver's reactions to these events are mostly transmitted through real yields, the dollar and the geopolitical-risk index, and the residual per-category effects are too small and too few to pin down. Next steps that would change that: more events per category, a proper out-of-sample test on 2022 onward once v1 is frozen, and a severity measure with real spread.
