# v1 out of sample: 2022-01-01 to the latest data

Built 2026-10-28. Test windows: every h-day window (h = 5, 20) whose base close is 2021-12-31 or later and whose last close is on or before 2026-10-27, the latest price in the data. v1 is the fit on 2000-08-30 to 2021-12-31 (`reports/v1_fit.md`, hand-severity ridge design); only its control coefficients and intercept are used, since cross-validation shrank the event terms to zero (silver at 5 days kept small category terms at a finite penalty; they are dropped here too, as specified). **Nothing is refitted**: the coefficients below are reproduced from the ≤2021 frame (cut before anything is computed) and match the v1 report; they are also saved to `data/v1_coefficients.csv`.

Windows overlap (one per trading day), so the effective number of independent observations is roughly n/h, and the same episode shows up in up to h consecutive rows.

## Frozen v1 coefficients (% of return)

| asset | h | intercept | d_real_yield (per +1pp) | dlog_dxy (per +1%) | dlog_vix (per +1%) | d_gpr (per +100) |
|---|---:|---:|---:|---:|---:|---:|
| gold | 5 | +0.152 | -2.97 | -0.98 | -0.001 | +0.19 |
| gold | 20 | +0.593 | -4.30 | -0.85 | +0.008 | +0.32 |
| silver | 5 | +0.111 | -3.53 | -1.60 | -0.048 | +0.34 |
| silver | 20 | +0.414 | -4.70 | -1.46 | -0.059 | +1.06 |

## Gold

### Gold, 5-day windows (n = 1205, 2022-01-03 to 2026-10-21 window starts)

| model | RMSE | mean error (realised − predicted) | R² vs zero | direction hit rate |
|---|---:|---:|---:|---:|
| no change (predict 0) | 2.64% | - | - | 57% positive |
| **frozen v1** | **2.34%** | +0.25% | +0.215 | **68%** |
| controls-only OLS refitted on the test period (in-sample) | 2.32% | +0.00% | +0.226 | 70% |

Five largest windows removed (by |realised return|; window starts 2026-01-20, 2026-03-13, 2026-03-16, 2026-03-17, 2026-03-18; n = 1200):

| model | RMSE | mean error | R² vs zero | direction hit rate |
|---|---:|---:|---:|---:|
| no change (predict 0) | 2.54% | - | - | 57% positive |
| **frozen v1** | **2.22%** | +0.28% | +0.232 | **68%** |
| controls-only OLS refitted on the test period (in-sample) | 2.20% | +0.00% | +0.246 | 70% |

Coefficient drift (frozen v1 vs the same controls refitted on the test windows; Newey-West SE, lag h):

| term | units | v1 (frozen, 2000-2021) | refit on test period | refit SE (NW) | t of difference |
|---|---|---:|---:|---:|---:|
| d_real_yield | per +1pp real yield | -2.97 | -4.04 | 1.13 | -0.9 |
| dlog_dxy | per +1% DXY | -0.98 | -0.94 | 0.15 | +0.2 |
| dlog_vix | per +1% VIX | -0.00 | +0.00 | 0.01 | +0.1 |
| d_gpr | per +100 GPR points | +0.19 | +0.23 | 0.14 | +0.3 |
| intercept | % per window | +0.15 | +0.42 | - | - |

### Gold, 20-day windows (n = 1190, 2022-01-03 to 2026-09-30 window starts)

| model | RMSE | mean error (realised − predicted) | R² vs zero | direction hit rate |
|---|---:|---:|---:|---:|
| no change (predict 0) | 5.14% | - | - | 60% positive |
| **frozen v1** | **4.28%** | +1.14% | +0.308 | **77%** |
| controls-only OLS refitted on the test period (in-sample) | 4.09% | +0.00% | +0.367 | 76% |

Five largest windows removed (by |realised return|; window starts 2025-12-30, 2025-12-31, 2026-02-24, 2026-02-25, 2026-02-27; n = 1185):

| model | RMSE | mean error | R² vs zero | direction hit rate |
|---|---:|---:|---:|---:|
| no change (predict 0) | 5.02% | - | - | 60% positive |
| **frozen v1** | **4.15%** | +1.15% | +0.314 | **77%** |
| controls-only OLS refitted on the test period (in-sample) | 3.96% | +0.00% | +0.378 | 76% |

Coefficient drift (frozen v1 vs the same controls refitted on the test windows; Newey-West SE, lag h):

| term | units | v1 (frozen, 2000-2021) | refit on test period | refit SE (NW) | t of difference |
|---|---|---:|---:|---:|---:|
| d_real_yield | per +1pp real yield | -4.30 | -5.10 | 1.60 | -0.5 |
| dlog_dxy | per +1% DXY | -0.85 | -0.99 | 0.22 | -0.6 |
| dlog_vix | per +1% VIX | +0.01 | +0.03 | 0.01 | +1.4 |
| d_gpr | per +100 GPR points | +0.32 | +0.40 | 0.28 | +0.3 |
| intercept | % per window | +0.59 | +1.80 | - | - |

## Silver

### Silver, 5-day windows (n = 1205, 2022-01-03 to 2026-10-21 window starts)

| model | RMSE | mean error (realised − predicted) | R² vs zero | direction hit rate |
|---|---:|---:|---:|---:|
| no change (predict 0) | 5.54% | - | - | 54% positive |
| **frozen v1** | **5.07%** | +0.34% | +0.164 | **65%** |
| controls-only OLS refitted on the test period (in-sample) | 5.05% | +0.00% | +0.169 | 65% |

Five largest windows removed (by |realised return|; window starts 2026-01-20, 2026-01-26, 2026-01-27, 2026-01-29, 2026-01-30; n = 1200):

| model | RMSE | mean error | R² vs zero | direction hit rate |
|---|---:|---:|---:|---:|
| no change (predict 0) | 5.12% | - | - | 54% positive |
| **frozen v1** | **4.66%** | +0.43% | +0.172 | **65%** |
| controls-only OLS refitted on the test period (in-sample) | 4.64% | +0.00% | +0.179 | 65% |

Coefficient drift (frozen v1 vs the same controls refitted on the test windows; Newey-West SE, lag h):

| term | units | v1 (frozen, 2000-2021) | refit on test period | refit SE (NW) | t of difference |
|---|---|---:|---:|---:|---:|
| d_real_yield | per +1pp real yield | -3.53 | -2.59 | 2.07 | +0.5 |
| dlog_dxy | per +1% DXY | -1.60 | -1.82 | 0.30 | -0.7 |
| dlog_vix | per +1% VIX | -0.05 | -0.05 | 0.02 | -0.3 |
| d_gpr | per +100 GPR points | +0.34 | +0.46 | 0.28 | +0.4 |
| intercept | % per window | +0.11 | +0.44 | - | - |

### Silver, 20-day windows (n = 1190, 2022-01-03 to 2026-09-30 window starts)

| model | RMSE | mean error (realised − predicted) | R² vs zero | direction hit rate |
|---|---:|---:|---:|---:|
| no change (predict 0) | 10.09% | - | - | 57% positive |
| **frozen v1** | **9.08%** | +1.52% | +0.190 | **69%** |
| controls-only OLS refitted on the test period (in-sample) | 8.93% | +0.00% | +0.217 | 70% |

Five largest windows removed (by |realised return|; window starts 2025-12-17, 2025-12-19, 2025-12-26, 2025-12-30, 2025-12-31; n = 1185):

| model | RMSE | mean error | R² vs zero | direction hit rate |
|---|---:|---:|---:|---:|
| no change (predict 0) | 9.72% | - | - | 57% positive |
| **frozen v1** | **8.68%** | +1.35% | +0.202 | **69%** |
| controls-only OLS refitted on the test period (in-sample) | 8.55% | +0.00% | +0.227 | 70% |

Coefficient drift (frozen v1 vs the same controls refitted on the test windows; Newey-West SE, lag h):

| term | units | v1 (frozen, 2000-2021) | refit on test period | refit SE (NW) | t of difference |
|---|---|---:|---:|---:|---:|
| d_real_yield | per +1pp real yield | -4.70 | -2.51 | 3.78 | +0.6 |
| dlog_dxy | per +1% DXY | -1.46 | -1.66 | 0.48 | -0.4 |
| dlog_vix | per +1% VIX | -0.06 | -0.07 | 0.03 | -0.4 |
| d_gpr | per +100 GPR points | +1.06 | +1.78 | 0.60 | +1.2 |
| intercept | % per window | +0.41 | +1.81 | - | - |

## Test-period events through the event-study machinery with v1 betas

Day 0 as listed; raw = log return from the close of day −1 to the close of day h−1; *explained* = Σ β_v1 × (change in the control over the same window) using the frozen coefficients above (no intercept); net = raw − explained; *v1 prediction* adds the intercept. *Trailing-beta net* is the repository's standard event study (betas on the dollar and real yields over the 250 days before day 0), for comparison. All in percent.

### Gold

| event | day 0 | h | raw | explained by v1 controls | **net (v1 betas)** | v1 prediction | trailing-beta net | Δ real yield (pp) | Δ DXY | Δ VIX | Δ GPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Russia invades Ukraine | 2022-02-24 | 5 | +0.62% | -0.11% | **+0.73%** | +0.04% | +0.23% | -0.31 | +1.2% | -1% | +93 |
| Russia invades Ukraine | 2022-02-24 | 20 | +1.40% | -2.53% | **+3.93%** | -1.94% | +3.41% | -0.08 | +2.5% | -27% | -164 |
| Russian central-bank reserves frozen (announced Sat 26 Feb) | 2022-02-28 | 5 | +4.10% | -1.31% | **+5.41%** | -1.15% | +3.86% | -0.36 | +2.1% | +15% | -166 |
| Russian central-bank reserves frozen (announced Sat 26 Feb) | 2022-02-28 | 20 | +3.47% | -3.10% | **+6.57%** | -2.51% | +5.95% | +0.10 | +2.2% | -28% | -165 |
| May CPI shock: 75bp hike priced for the June FOMC | 2022-06-10 | 5 | -0.16% | -1.48% | **+1.32%** | -1.32% | +2.34% | +0.37 | +0.4% | +23% | +17 |
| May CPI shock: 75bp hike priced for the June FOMC | 2022-06-10 | 20 | -6.76% | -5.51% | **-1.25%** | -4.92% | -2.15% | +0.38 | +4.5% | +0% | -3 |
| Fed's first 75bp hike since 1994 | 2022-06-15 | 5 | +1.36% | +2.02% | **-0.65%** | +2.17% | -0.79% | -0.27 | -1.3% | -12% | -15 |
| Fed's first 75bp hike since 1994 | 2022-06-15 | 20 | -6.12% | -1.32% | **-4.80%** | -0.73% | -6.75% | -0.27 | +2.8% | -21% | +29 |
| Silicon Valley Bank closed by regulators | 2023-03-10 | 5 | +4.71% | +1.62% | **+3.08%** | +1.78% | +3.21% | -0.29 | -0.8% | +2% | -35 |
| Silicon Valley Bank closed by regulators | 2023-03-10 | 20 | +9.94% | +5.24% | **+4.70%** | +5.83% | +5.65% | -0.57 | -3.4% | -21% | +26 |
| Hamas attack on Israel (Sat 7 Oct) | 2023-10-09 | 5 | +5.09% | +0.21% | **+4.88%** | +0.36% | +4.95% | -0.18 | +0.6% | +10% | +126 |
| Hamas attack on Israel (Sat 7 Oct) | 2023-10-09 | 20 | +8.02% | +1.99% | **+6.03%** | +2.58% | +6.21% | -0.29 | -1.0% | -16% | +14 |
| 'Liberation Day' tariff round (announced after the close 2 Apr) | 2025-04-03 | 5 | -2.78% | +0.40% | **-3.18%** | +0.55% | -2.37% | +0.19 | -0.9% | +45% | +74 |
| 'Liberation Day' tariff round (announced after the close 2 Apr) | 2025-04-03 | 20 | +1.75% | +2.53% | **-0.78%** | +3.13% | +0.88% | +0.12 | -3.5% | +13% | -10 |

### Silver

| event | day 0 | h | raw | explained by v1 controls | **net (v1 betas)** | v1 prediction | trailing-beta net | Δ real yield (pp) | Δ DXY | Δ VIX | Δ GPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Russia invades Ukraine | 2022-02-24 | 5 | +2.53% | -0.53% | **+3.06%** | -0.42% | +3.43% | -0.31 | +1.2% | -1% | +93 |
| Russia invades Ukraine | 2022-02-24 | 20 | +2.47% | -3.37% | **+5.84%** | -2.95% | +6.85% | -0.08 | +2.5% | -27% | -164 |
| Russian central-bank reserves frozen (announced Sat 26 Feb) | 2022-02-28 | 5 | +7.17% | -3.33% | **+10.50%** | -3.22% | +8.84% | -0.36 | +2.1% | +15% | -166 |
| Russian central-bank reserves frozen (announced Sat 26 Feb) | 2022-02-28 | 20 | +6.45% | -3.82% | **+10.26%** | -3.40% | +10.89% | +0.10 | +2.2% | -28% | -165 |
| May CPI shock: 75bp hike priced for the June FOMC | 2022-06-10 | 5 | +0.32% | -3.00% | **+3.32%** | -2.89% | +3.74% | +0.37 | +0.4% | +23% | +17 |
| May CPI shock: 75bp hike priced for the June FOMC | 2022-06-10 | 20 | -13.38% | -8.46% | **-4.92%** | -8.05% | -5.89% | +0.38 | +4.5% | +0% | -3 |
| Fed's first 75bp hike since 1994 | 2022-06-15 | 5 | +2.22% | +3.50% | **-1.27%** | +3.61% | -1.01% | -0.27 | -1.3% | -12% | -15 |
| Fed's first 75bp hike since 1994 | 2022-06-15 | 20 | -14.18% | -1.26% | **-12.91%** | -0.85% | -14.13% | -0.27 | +2.8% | -21% | +29 |
| Silicon Valley Bank closed by regulators | 2023-03-10 | 5 | +7.52% | +2.18% | **+5.33%** | +2.29% | +4.93% | -0.29 | -0.8% | +2% | -35 |
| Silicon Valley Bank closed by regulators | 2023-03-10 | 20 | +22.29% | +9.10% | **+13.19%** | +9.51% | +14.78% | -0.57 | -3.4% | -21% | +26 |
| Hamas attack on Israel (Sat 7 Oct) | 2023-10-09 | 5 | +5.36% | -0.34% | **+5.70%** | -0.23% | +6.00% | -0.18 | +0.6% | +10% | +126 |
| Hamas attack on Israel (Sat 7 Oct) | 2023-10-09 | 20 | +7.43% | +3.85% | **+3.58%** | +4.27% | +4.85% | -0.29 | -1.0% | -16% | +14 |
| 'Liberation Day' tariff round (announced after the close 2 Apr) | 2025-04-03 | 5 | -12.90% | -1.14% | **-11.76%** | -1.03% | -13.99% | +0.19 | -0.9% | +45% | +74 |
| 'Liberation Day' tariff round (announced after the close 2 Apr) | 2025-04-03 | 20 | -6.93% | +3.61% | **-10.54%** | +4.03% | -11.29% | +0.12 | -3.5% | +13% | -10 |

Net moves (v1 betas) summarised:

| event | gold 5d | gold 20d | silver 5d | silver 20d |
|---|---:|---:|---:|---:|
| Russia invades Ukraine | +0.7% | +3.9% | +3.1% | +5.8% |
| Russian central-bank reserves frozen (announced Sat 26 Feb) | +5.4% | +6.6% | +10.5% | +10.3% |
| May CPI shock: 75bp hike priced for the June FOMC | +1.3% | -1.2% | +3.3% | -4.9% |
| Fed's first 75bp hike since 1994 | -0.7% | -4.8% | -1.3% | -12.9% |
| Silicon Valley Bank closed by regulators | +3.1% | +4.7% | +5.3% | +13.2% |
| Hamas attack on Israel (Sat 7 Oct) | +4.9% | +6.0% | +5.7% | +3.6% |
| 'Liberation Day' tariff round (announced after the close 2 Apr) | -3.2% | -0.8% | -11.8% | -10.5% |

## Verdict

**What was tested.** The 2000-2021 control coefficients, applied unchanged to 1,205 five-day and 1,190 twenty-day windows from January 2022 to the latest close. The regressors are the changes in real yields, the dollar, the VIX and the GPR index *over the same window*, so this is a test of whether the sensitivities still hold, not a forecast made before the window opened: a user of these numbers would need a view on the controls first.

**Against no-change.** The frozen v1 beats predicting zero on all four asset-horizon pairs, by 9% to 17% of RMSE (gold 11% at 5 days, 17% at 20), and still does with the five largest windows removed, so the result is not carried by a few extreme weeks. Direction is right 65% to 77% of the time (gold 68% at 5 days and 77% at 20), above the share of rising windows in every case (57% and 60% for gold).

**Against a refit on the test period.** Refitting the same four controls on 2022-2026 (an in-sample fit, the most the specification could have done) lowers RMSE by only 0.02 to 0.19 points; the refitted sensitivities sit within one and a half standard errors of the frozen ones everywhere (largest difference t = 1.4, dlog_vix (gold 20-day)). What did move is the intercept: gold's per-window drift went from +0.59% to +1.80% at 20 days, silver's from +0.41% to +1.81%. That is the whole of v1's bias: the mean error is +0.25% and +1.14% for gold and +0.34% and +1.52% for silver, i.e. both metals rose by more than their real-yield and dollar moves account for, consistently, through 2022-2026. The betas did not drift; the level did.

**Events.** Through the event-study machinery with the v1 betas, net gold moves at 5/20 days were: Russia invades Ukraine +0.7%/+3.9%; Russian central-bank reserves frozen +5.4%/+6.6%; May CPI shock: 75bp hike priced for the June FOMC +1.3%/-1.2%; Fed's first 75bp hike since 1994 -0.7%/-4.8%; Silicon Valley Bank closed by regulators +3.1%/+4.7%; Hamas attack on Israel +4.9%/+6.0%; 'Liberation Day' tariff round -3.2%/-0.8%. The invasion and the reserve freeze, SVB and the October 2023 attack all left gold higher than its controls explain at both horizons; the June 2022 rate shock left it lower, with the controls explaining most of the 20-day drop after the CPI print; the 2025 tariff round shows a net fall at 5 days that had mostly closed by 20 days for gold and had not for silver. The v1-beta and trailing-beta net moves have the same sign in 96% of the event rows and never differ by more than 2.2 points, another sign that the control sensitivities are stable.

**Verdict.** v1's control betas survive five years out of sample intact: they explain as much of the 5- and 20-day variation in 2022-2026 as a model fitted on that period would, and the event-study net moves computed with them agree with the trailing-beta versions. What v1 misses is a persistent upward drift in both metals that no change-in-controls model can carry, worth about a point per 20-day window for gold and one and a half for silver. Used for what it was built for, separating an event's move from the part the dollar and yields explain, v1 is fit for purpose on the test period; used as a return forecaster it would have been right on direction about seven times in ten but systematically too low.
