# v1 model fit: gold and silver returns on controls and event terms

Built 2026-10-28. Sample: windows starting after 2000-08-30 and ending on or before 2021-12-31 (the data frame is cut at 2021-12-31 before anything is computed). Last price used: 2021-12-31.

## Setup

For every trading day t, y = log return from the close of t-1 to the close of t+h-1 (h = 5, 20).
Regressors over the same window: change in the 10y real yield, change in log DXY, change in log VIX,
change in the daily GPR index, and one term per event category equal to severity x 1[an event of that
category has its day 0 at t]. Day 0 = the event date if it is a trading day, else the next one.
Severity in this section is either the hand score (1-5) or the quintile score (max of the GPR and EPU
jump quintiles, `severity_quintile`); the continuous measured severity is used in the refit section below.

Fit: ridge on the event terms only (controls unpenalised), penalty chosen by 5-fold blocked
cross-validation; Newey-West standard errors (lag h) because the windows overlap. Coefficients are in
percent of return. *Event windows* in the error tables are the rows whose window starts at an event's
day 0, i.e. the rows where the event terms can make any difference. A category term is identified by
only as many rows as it has events, so its standard error and t-statistic are unreliable for categories
with fewer than about five events, whatever the number says.

## Events in sample

50 of 68 events fall in the sample. By category:

| category | events in sample | mean severity_hand | mean severity_quintile |
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

Cross-validated ridge penalty on the event terms (standardised units): hand 1e+07 (top of the grid: event terms shrunk to ~0); quintile 1e+07 (top of the grid: event terms shrunk to ~0). CV MSE (units 1e-4) at lambda = 0 vs chosen: hand 4.802 vs 4.775; quintile 4.802 vs 4.775.

Ridge fit (what cross-validation keeps):

| term | units | hand: coef | SE | t | quintile: coef | SE | t |
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

| category (unpenalised) | hand: % per point | SE | t | quintile: % per point | SE | t |
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
| controls + events (quintile), ridge | 2.13% | 3.37% |
| controls + events (quintile), unpenalised | 2.12% | 3.21% |

- |t| >= 2, hand severity, ridge: none.
- |t| >= 2, hand severity, unpenalised: Pandemic or disaster (-0.75% per point, t=-9.6); Sovereign debt crisis (+0.78% per point, t=+2.4); Terror attack (+0.57% per point, t=+2.7).
- |t| >= 2, quintile severity, ridge: none.
- |t| >= 2, quintile severity, unpenalised: Pandemic or disaster (-0.61% per point, t=-7.6); Sovereign debt crisis (+0.49% per point, t=+2.0).

### Gold, 20-day windows

Cross-validated ridge penalty on the event terms (standardised units): hand 1e+07 (top of the grid: event terms shrunk to ~0); quintile 1e+07 (top of the grid: event terms shrunk to ~0). CV MSE (units 1e-4) at lambda = 0 vs chosen: hand 17.807 vs 17.452; quintile 17.668 vs 17.452.

Ridge fit (what cross-validation keeps):

| term | units | hand: coef | SE | t | quintile: coef | SE | t |
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

| category (unpenalised) | hand: % per point | SE | t | quintile: % per point | SE | t |
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
| controls + events (quintile), ridge | 4.01% | 6.04% |
| controls + events (quintile), unpenalised | 4.01% | 5.87% |

- |t| >= 2, hand severity, ridge: none.
- |t| >= 2, hand severity, unpenalised: Pandemic or disaster (+0.42% per point, t=+5.4).
- |t| >= 2, quintile severity, ridge: none.
- |t| >= 2, quintile severity, unpenalised: Pandemic or disaster (+0.35% per point, t=+6.1); Terror attack (+0.56% per point, t=+2.0).

## Silver

### Silver, 5-day windows

Cross-validated ridge penalty on the event terms (standardised units): hand 251; quintile 6.31e+03. CV MSE (units 1e-4) at lambda = 0 vs chosen: hand 15.731 vs 15.730; quintile 15.800 vs 15.739.

Ridge fit (what cross-validation keeps):

| term | units | hand: coef | SE | t | quintile: coef | SE | t |
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

| category (unpenalised) | hand: % per point | SE | t | quintile: % per point | SE | t |
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
| controls + events (quintile), ridge | 3.88% | 5.89% |
| controls + events (quintile), unpenalised | 3.88% | 5.68% |

- |t| >= 2, hand severity, ridge: Pandemic or disaster (-3.32% per point, t=-4.5); Sanctions and trade war (+0.36% per point, t=+2.3); Sovereign debt crisis (+0.85% per point, t=+2.6); Terror attack (+0.76% per point, t=+2.3).
- |t| >= 2, hand severity, unpenalised: Pandemic or disaster (-3.47% per point, t=-4.8); Sanctions and trade war (+0.38% per point, t=+2.3); Sovereign debt crisis (+0.89% per point, t=+2.5); Terror attack (+0.79% per point, t=+2.4).
- |t| >= 2, quintile severity, ridge: Sovereign debt crisis (+0.28% per point, t=+2.2).
- |t| >= 2, quintile severity, unpenalised: Pandemic or disaster (-2.53% per point, t=-2.2); Sanctions and trade war (+0.26% per point, t=+2.3); Sovereign debt crisis (+0.61% per point, t=+2.8).

### Silver, 20-day windows

Cross-validated ridge penalty on the event terms (standardised units): hand 1e+07 (top of the grid: event terms shrunk to ~0); quintile 1e+07 (top of the grid: event terms shrunk to ~0). CV MSE (units 1e-4) at lambda = 0 vs chosen: hand 61.058 vs 60.585; quintile 60.876 vs 60.585.

Ridge fit (what cross-validation keeps):

| term | units | hand: coef | SE | t | quintile: coef | SE | t |
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

| category (unpenalised) | hand: % per point | SE | t | quintile: % per point | SE | t |
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
| controls + events (quintile), ridge | 7.61% | 8.71% |
| controls + events (quintile), unpenalised | 7.60% | 8.34% |

- |t| >= 2, hand severity, ridge: none.
- |t| >= 2, hand severity, unpenalised: none.
- |t| >= 2, quintile severity, ridge: none.
- |t| >= 2, quintile severity, unpenalised: none.

## Reading

The controls carry the model. Over 20-day windows gold moves about -4.3% per percentage point of real yield and -0.85% per 1% move in the dollar (5-day: -3.0% and -0.98%), with the expected signs, t-statistics well beyond 2, and no dependence on which severity is used. A rise in the GPR index over the window is also associated with higher gold and silver, after the dollar and real yields are taken into account. On gold's event windows the controls alone cut the 5-day RMSE from 3.73% to 3.37% and raise the 20-day RMSE from 5.89% to 6.04%; unpenalised event terms take those to 3.17% and 5.74%, while the cross-validated ridge versions sit at 3.37% and 6.04%. On silver's event windows the controls alone cut the 5-day RMSE from 7.52% to 6.36% and cut the 20-day RMSE from 8.99% to 8.71%; unpenalised event terms take those to 5.52% and 8.50%, while the cross-validated ridge versions sit at 5.52% and 8.71%. Cross-validation is the key result: in 6 of the 8 fits it pushes the event penalty to the top of the grid, i.e. it finds that category effects fitted on one stretch of the sample do not help predict returns on another stretch, so the event terms are shrunk to nothing. The unpenalised fits do show a few categories with |t| >= 2, but none keeps the same sign and |t| >= 2 for gold across both horizons and both severity versions, and none does for silver. Gold's Pandemic or disaster term even flips sign between 5 and 20 days. The terms that do appear rest on two to four events each (the pandemic category is Fukushima and Covid, nothing else), and their t-statistics come from a handful of rows, so they describe those episodes rather than estimate a category effect. Quintile severity adds little beyond an on/off indicator: 46 of 68 events score 5 on it, because the larger of two quintile ranks against all days is almost always high for events chosen for being notable. The honest summary for v1 is that gold's and silver's reactions to these events are mostly transmitted through real yields, the dollar and the geopolitical-risk index, and the residual per-category effects are too small and too few to pin down. Next steps that would change that: more events per category, a proper out-of-sample test on 2022 onward once v1 is frozen, and a severity measure with real spread.

# Refit: continuous severity, pooled event term, mechanical list

Same windows, controls, sample (2000-08-30 to 2021-12-31), ridge and blocked 5-fold CV as above.
Severity is now the continuous measured score, capped: the larger of the GPR and EPU 5-day jumps, each as
a z-score against all daily jumps 1990-2021, winsorised at z = 3 (`severity_capped`). Three event designs
are compared with the controls-only baseline:

1. hand list, one severity x indicator term per category;
2. hand list, a single `any_event` = severity x 1[any hand-list event has its day 0 at t];
3. mechanical list (`data/events_mechanical.csv`, days when GPR or EPU is in its top 5%, merged into
   runs), the same single pooled term with the same jump-based severity.

A fourth row, the hand list's pooled term with September 11 removed, is a robustness check: that event's
uncapped severity (z = 19) was several times any other; the cap limits it to 3, the same as the next
largest events, but it remains the one event everyone would name first.

In sample: 50 hand-list events (7 with negative severity, i.e. both indices fell into the event; 11 capped at 3) and 138 mechanical events (5 negative; 12 capped). Severity is used as given, so a negative-severity event enters with its sign flipped.

*CV RMSE* is the root mean squared out-of-fold residual from the blocked cross-validation at the chosen
penalty (for controls only, plain OLS refitted per fold), on all windows and on the windows that start at
a hand-list or a mechanical event. *Mean abnormal return* is the average out-of-fold controls-only residual
on those windows, with its naive standard error: a selection check, since a hand-picked list of events
that 'moved gold' should show a mean abnormal return the mechanical list does not.

## Gold, 5-day windows

| model | lambda | CV RMSE all (n=5364) | CV RMSE hand windows (n=50) | CV RMSE mechanical windows (n=138) | in-sample RMSE all |
|---|---|---:|---:|---:|---:|
| no change (predict 0) | - | 2.46% | 3.73% | 3.05% | 2.46% |
| controls only | - | 2.19% | 3.48% | 2.71% | 2.13% |
| hand list, category terms | 1e+07 (top of the grid: event terms shrunk to ~0) | 2.19% | 3.48% | 2.71% | 2.13% |
| hand list, pooled term | 1.58e+03 | 2.18% | 3.45% | 2.70% | 2.13% |
| mechanical list, pooled term | 1e+07 (top of the grid: event terms shrunk to ~0) | 2.19% | 3.48% | 2.71% | 2.13% |
| hand list, pooled term, ex 9/11 | 1.58e+03 | 2.18% | 3.46% | 2.71% | 2.13% |

Mean return and mean abnormal (controls-only, out-of-fold) return on event windows:

| windows | n | mean raw return | mean abnormal return (SE) |
|---|---:|---:|---:|
| all | 5364 | +0.18% | +0.00% (0.03) |
| hand-list events | 50 | +0.40% | +0.45% (0.49) |
| mechanical events | 138 | +0.35% | +0.23% (0.23) |

Event coefficients (% of return per unit of severity):

| model | term | ridge: coef | SE | t | unpenalised: coef | SE | t |
|---|---|---:|---:|---:|---:|---:|---:|
| hand list, category terms | Banking or financial stress | +0.000 | 0.001 | +0.4 | +0.611 | 1.486 | +0.4 |
| hand list, category terms | Central bank policy shock | -0.000 | 0.000 | -0.4 | -0.246 | 0.671 | -0.4 |
| hand list, category terms | Election or leadership shock | +0.000 | 0.000 | +0.5 | +0.410 | 0.779 | +0.5 |
| hand list, category terms | Military escalation | +0.000 | 0.000 | +0.7 | +0.229 | 0.293 | +0.8 |
| hand list, category terms | Pandemic or disaster | -0.001 | 0.001 | -1.3 | -2.039 | 0.294 | -6.9 |
| hand list, category terms | Sanctions and trade war | +0.000 | 0.000 | +1.2 | +0.282 | 0.245 | +1.2 |
| hand list, category terms | Sovereign debt crisis | +0.001 | 0.001 | +1.4 | +1.772 | 0.304 | +5.8 |
| hand list, category terms | Terror attack | +0.000 | 0.000 | +0.9 | +0.417 | 0.443 | +0.9 |
| hand list, pooled term | any_event | +0.247 | 0.222 | +1.1 | +0.320 | 0.287 | +1.1 |
| mechanical list, pooled term | any_event | +0.000 | 0.000 | +0.6 | +0.102 | 0.160 | +0.6 |
| hand list, pooled term, ex 9/11 | any_event | +0.207 | 0.231 | +0.9 | +0.268 | 0.298 | +0.9 |

## Gold, 20-day windows

| model | lambda | CV RMSE all (n=5349) | CV RMSE hand windows (n=49) | CV RMSE mechanical windows (n=137) | in-sample RMSE all |
|---|---|---:|---:|---:|---:|
| no change (predict 0) | - | 4.72% | 5.89% | 5.00% | 4.72% |
| controls only | - | 4.18% | 6.18% | 4.32% | 4.01% |
| hand list, category terms | 1e+07 (top of the grid: event terms shrunk to ~0) | 4.18% | 6.18% | 4.32% | 4.01% |
| hand list, pooled term | 0 (no shrinkage) | 4.18% | 6.06% | 4.30% | 4.01% |
| mechanical list, pooled term | 1e+07 (top of the grid: event terms shrunk to ~0) | 4.18% | 6.18% | 4.32% | 4.01% |
| hand list, pooled term, ex 9/11 | 0 (no shrinkage) | 4.18% | 6.06% | 4.30% | 4.01% |

Mean return and mean abnormal (controls-only, out-of-fold) return on event windows:

| windows | n | mean raw return | mean abnormal return (SE) |
|---|---:|---:|---:|
| all | 5349 | +0.70% | -0.00% (0.06) |
| hand-list events | 49 | +0.82% | +0.74% (0.89) |
| mechanical events | 137 | +1.15% | +0.20% (0.37) |

Event coefficients (% of return per unit of severity):

| model | term | ridge: coef | SE | t | unpenalised: coef | SE | t |
|---|---|---:|---:|---:|---:|---:|---:|
| hand list, category terms | Banking or financial stress | +0.001 | 0.001 | +0.7 | +1.829 | 2.438 | +0.8 |
| hand list, category terms | Central bank policy shock | +0.000 | 0.001 | +0.4 | +0.626 | 1.424 | +0.4 |
| hand list, category terms | Election or leadership shock | -0.000 | 0.001 | -0.4 | -0.464 | 1.254 | -0.4 |
| hand list, category terms | Military escalation | +0.000 | 0.000 | +0.2 | +0.129 | 0.602 | +0.2 |
| hand list, category terms | Pandemic or disaster | +0.001 | 0.000 | +1.3 | +1.154 | 0.256 | +4.5 |
| hand list, category terms | Sanctions and trade war | +0.000 | 0.000 | +0.7 | +0.349 | 0.522 | +0.7 |
| hand list, category terms | Sovereign debt crisis | +0.002 | 0.001 | +1.3 | +3.023 | 0.776 | +3.9 |
| hand list, category terms | Terror attack | +0.000 | 0.000 | +1.4 | +0.902 | 0.505 | +1.8 |
| hand list, pooled term | any_event | +0.715 | 0.516 | +1.4 | +0.715 | 0.516 | +1.4 |
| mechanical list, pooled term | any_event | +0.000 | 0.000 | +0.2 | +0.049 | 0.216 | +0.2 |
| hand list, pooled term, ex 9/11 | any_event | +0.722 | 0.543 | +1.3 | +0.722 | 0.543 | +1.3 |

## Silver, 5-day windows

| model | lambda | CV RMSE all (n=5364) | CV RMSE hand windows (n=50) | CV RMSE mechanical windows (n=138) | in-sample RMSE all |
|---|---|---:|---:|---:|---:|
| no change (predict 0) | - | 4.40% | 7.52% | 4.96% | 4.40% |
| controls only | - | 3.97% | 6.55% | 4.26% | 3.89% |
| hand list, category terms | 1.58e+03 | 3.96% | 6.33% | 4.28% | 3.87% |
| hand list, pooled term | 1e+07 (top of the grid: event terms shrunk to ~0) | 3.97% | 6.55% | 4.26% | 3.89% |
| mechanical list, pooled term | 1e+07 (top of the grid: event terms shrunk to ~0) | 3.97% | 6.55% | 4.26% | 3.89% |
| hand list, pooled term, ex 9/11 | 1e+07 (top of the grid: event terms shrunk to ~0) | 3.97% | 6.55% | 4.26% | 3.89% |

Mean return and mean abnormal (controls-only, out-of-fold) return on event windows:

| windows | n | mean raw return | mean abnormal return (SE) |
|---|---:|---:|---:|
| all | 5364 | +0.14% | -0.00% (0.05) |
| hand-list events | 50 | -0.60% | -0.13% (0.94) |
| mechanical events | 138 | +0.58% | +0.38% (0.36) |

Event coefficients (% of return per unit of severity):

| model | term | ridge: coef | SE | t | unpenalised: coef | SE | t |
|---|---|---:|---:|---:|---:|---:|---:|
| hand list, category terms | Banking or financial stress | -0.602 | 1.992 | -0.3 | -0.782 | 2.584 | -0.3 |
| hand list, category terms | Central bank policy shock | -1.460 | 0.908 | -1.6 | -1.891 | 1.177 | -1.6 |
| hand list, category terms | Election or leadership shock | +0.441 | 1.025 | +0.4 | +0.573 | 1.323 | +0.4 |
| hand list, category terms | Military escalation | +0.280 | 0.322 | +0.9 | +0.363 | 0.417 | +0.9 |
| hand list, category terms | Pandemic or disaster | -7.635 | 2.029 | -3.8 | -9.905 | 1.193 | -8.3 |
| hand list, category terms | Sanctions and trade war | +0.323 | 0.231 | +1.4 | +0.419 | 0.326 | +1.3 |
| hand list, category terms | Sovereign debt crisis | +1.352 | 0.569 | +2.4 | +1.747 | 0.728 | +2.4 |
| hand list, category terms | Terror attack | +0.428 | 0.472 | +0.9 | +0.557 | 0.615 | +0.9 |
| hand list, pooled term | any_event | -0.000 | 0.000 | -0.4 | -0.234 | 0.564 | -0.4 |
| mechanical list, pooled term | any_event | +0.000 | 0.000 | +0.5 | +0.125 | 0.251 | +0.5 |
| hand list, pooled term, ex 9/11 | any_event | -0.000 | 0.000 | -0.6 | -0.346 | 0.582 | -0.6 |

## Silver, 20-day windows

| model | lambda | CV RMSE all (n=5349) | CV RMSE hand windows (n=49) | CV RMSE mechanical windows (n=137) | in-sample RMSE all |
|---|---|---:|---:|---:|---:|
| no change (predict 0) | - | 8.62% | 8.99% | 8.92% | 8.62% |
| controls only | - | 7.78% | 8.76% | 7.84% | 7.61% |
| hand list, category terms | 1e+07 (top of the grid: event terms shrunk to ~0) | 7.78% | 8.76% | 7.84% | 7.61% |
| hand list, pooled term | 6.31e+03 | 7.78% | 8.75% | 7.83% | 7.61% |
| mechanical list, pooled term | 100 | 7.78% | 8.78% | 7.83% | 7.61% |
| hand list, pooled term, ex 9/11 | 6.31e+03 | 7.78% | 8.75% | 7.83% | 7.61% |

Mean return and mean abnormal (controls-only, out-of-fold) return on event windows:

| windows | n | mean raw return | mean abnormal return (SE) |
|---|---:|---:|---:|
| all | 5349 | +0.57% | -0.02% (0.11) |
| hand-list events | 49 | +0.39% | +0.87% (1.26) |
| mechanical events | 137 | +1.52% | +0.27% (0.67) |

Event coefficients (% of return per unit of severity):

| model | term | ridge: coef | SE | t | unpenalised: coef | SE | t |
|---|---|---:|---:|---:|---:|---:|---:|
| hand list, category terms | Banking or financial stress | +0.000 | 0.002 | +0.0 | +0.040 | 2.850 | +0.0 |
| hand list, category terms | Central bank policy shock | -0.001 | 0.001 | -1.3 | -2.039 | 1.656 | -1.2 |
| hand list, category terms | Election or leadership shock | +0.001 | 0.001 | +0.5 | +1.059 | 2.016 | +0.5 |
| hand list, category terms | Military escalation | +0.000 | 0.000 | +0.9 | +0.828 | 0.935 | +0.9 |
| hand list, category terms | Pandemic or disaster | -0.000 | 0.002 | -0.3 | -0.807 | 2.497 | -0.3 |
| hand list, category terms | Sanctions and trade war | +0.000 | 0.001 | +0.6 | +0.770 | 1.318 | +0.6 |
| hand list, category terms | Sovereign debt crisis | +0.002 | 0.001 | +1.2 | +2.915 | 0.893 | +3.3 |
| hand list, category terms | Terror attack | +0.001 | 0.001 | +1.1 | +1.295 | 1.149 | +1.1 |
| hand list, pooled term | any_event | +0.224 | 0.300 | +0.7 | +0.488 | 0.652 | +0.7 |
| mechanical list, pooled term | any_event | -0.225 | 0.351 | -0.6 | -0.229 | 0.358 | -0.6 |
| hand list, pooled term, ex 9/11 | any_event | +0.235 | 0.316 | +0.7 | +0.511 | 0.687 | +0.7 |

## Reading

**Does any event term help out of sample?** Barely. Across the 12 comparisons (two metals, two horizons, three designs) the largest improvement in cross-validated RMSE over controls-only is 0.005 percentage points on all windows and 0.22 points on the hand-list event windows (silver, 5-day, hand list, category terms), against event-window RMSEs of 3 to 9%. In 7 of the 12 fits cross-validation pushes the penalty to the top of the grid and removes the event terms outright. The mechanical list's pooled term never improves CV RMSE by more than 0.000 points.

**Which terms survive the shrinkage with |t| >= 2?** Pandemic or disaster (silver, 5-day, hand list, category terms: -7.64% per unit, t = -3.8); Sovereign debt crisis (silver, 5-day, hand list, category terms: +1.35% per unit, t = +2.4). No pooled term survives; the hand list's pooled term for gold is +0.25% per unit at 5 days (t = +1.1) and +0.72% at 20 days (t = +1.4). The category terms that pass for silver at 5 days are the same two-to-four-event categories as before (the pandemic category is Fukushima and Covid), and their standard errors come from that handful of rows.

**Does the hand-picked list look biased?** Not in the sense of being days when gold happened to jump: the mean out-of-fold abnormal gold return on hand-list windows is +0.45% at 5 days and +0.74% at 20 days (SE 0.49 and 0.89), versus +0.23% and +0.20% on mechanical windows (SE 0.23 and 0.37), and none of these means is two standard errors from zero. The list is selected on something else: volatility. Controls-only CV RMSE on hand-list windows is 1.6x its value on all windows, against 1.2x for the mechanical windows, so the hand list is a list of days when markets were moving a lot, chosen knowing they did. It also covers different ground from the indices: 20 of its 68 events are never flagged by GPR or EPU (central-bank decisions, sovereign-debt episodes, the gold-system events), while the mechanical list is full of EPU spikes on US fiscal news that the hand list ignores, and 7 hand-list events have negative measured severity because the indices fell into them.

**Verdict for v1.** With real yields, the dollar, the VIX and the GPR index in the regression, neither the hand-picked events nor the mechanically flagged ones add a detectable, out-of-sample-stable effect on gold or silver at 5 or 20 days, pooled or by category, with hand, quintile or capped continuous severity. The mechanical list's merge rule turns long crises into single onset-dated events (the Iraq build-up runs from 2002-12-18 to 2003-05-22), which is the right unit for some questions and the wrong one for an invasion-day study.
