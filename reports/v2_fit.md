# v2: v1 plus GLD holdings and central-bank buying

v2 adds two demand factors to the v1 specification: the change in SPDR GLD holdings over the window
(tonnes) and official-sector net gold purchases during the window at the lagged quarterly pace (IMF
IRFCL reporters' quarterly net purchases, lagged one quarter so only published data is used, interpolated
to daily, times h/63). GLD starts 2004-11-18, so v2 trains on 2004-11 to 2021-12; the v1 specification
is refitted on exactly those rows for a like-for-like comparison. Fit, ridge and cross-validation are
as in v1. The central-bank series is built from the IMF's reserves template rather than the World Gold
Council file (which needs a Goldhub login): it covers reporting central banks only, so it understates
WGC's estimate, which includes unreported buying.

## Coefficients (train, percent of return per unit)

| asset | h | term | v1 (2000-2021) | v1 on v2 rows | **v2** | v2 SE | n |
|---|---:|---|---:|---:|---:|---:|---:|
| gold | 5 | intercept | +0.152 | +0.166 | **+0.120** | - | 4305 |
| gold | 5 | d_real_yield (per +1pp real yield) | -2.969 | -4.363 | **-3.433** | 0.868 | |
| gold | 5 | dlog_dxy (per +1% DXY) | -0.980 | -0.996 | **-0.934** | 0.070 | |
| gold | 5 | dlog_vix (per +1% VIX) | -0.001 | -0.000 | **-0.003** | 0.004 | |
| gold | 5 | d_gpr (per +100 GPR points) | +0.195 | +0.156 | **+0.154** | 0.078 | |
| gold | 5 | d_gld (per +10 t GLD holdings) | - | - | **+0.579** | 0.061 | |
| gold | 5 | cb_flow (per 100 t official buying in the window) | - | - | **-0.548** | 0.805 | |
| gold | 20 | intercept | +0.593 | +0.621 | **+0.512** | - | 4290 |
| gold | 20 | d_real_yield (per +1pp real yield) | -4.299 | -7.089 | **-5.314** | 1.105 | |
| gold | 20 | dlog_dxy (per +1% DXY) | -0.852 | -0.850 | **-0.851** | 0.112 | |
| gold | 20 | dlog_vix (per +1% VIX) | +0.008 | +0.014 | **+0.008** | 0.007 | |
| gold | 20 | d_gpr (per +100 GPR points) | +0.321 | +0.401 | **+0.372** | 0.190 | |
| gold | 20 | d_gld (per +10 t GLD holdings) | - | - | **+0.496** | 0.061 | |
| gold | 20 | cb_flow (per 100 t official buying in the window) | - | - | **-0.763** | 0.720 | |
| silver | 5 | intercept | +0.111 | +0.142 | **+0.071** | - | 4305 |
| silver | 5 | d_real_yield (per +1pp real yield) | -3.528 | -6.468 | **-4.943** | 1.436 | |
| silver | 5 | dlog_dxy (per +1% DXY) | -1.600 | -1.716 | **-1.618** | 0.114 | |
| silver | 5 | dlog_vix (per +1% VIX) | -0.048 | -0.050 | **-0.053** | 0.007 | |
| silver | 5 | d_gpr (per +100 GPR points) | +0.342 | +0.269 | **+0.268** | 0.129 | |
| silver | 5 | d_gld (per +10 t GLD holdings) | - | - | **+0.925** | 0.102 | |
| silver | 5 | cb_flow (per 100 t official buying in the window) | - | - | **-0.861** | 1.462 | |
| silver | 20 | intercept | +0.414 | +0.544 | **+0.394** | - | 4290 |
| silver | 20 | d_real_yield (per +1pp real yield) | -4.701 | -8.289 | **-5.635** | 1.950 | |
| silver | 20 | dlog_dxy (per +1% DXY) | -1.458 | -1.578 | **-1.579** | 0.214 | |
| silver | 20 | dlog_vix (per +1% VIX) | -0.059 | -0.057 | **-0.066** | 0.016 | |
| silver | 20 | d_gpr (per +100 GPR points) | +1.064 | +1.112 | **+1.068** | 0.362 | |
| silver | 20 | d_gld (per +10 t GLD holdings) | - | - | **+0.745** | 0.108 | |
| silver | 20 | cb_flow (per 100 t official buying in the window) | - | - | **-1.292** | 1.368 | |

Training windows: gold 5d 2004-11-19..2021-12-27 (4305); gold 20d 2004-11-19..2021-12-03 (4290); silver 5d 2004-11-19..2021-12-27 (4305); silver 20d 2004-11-19..2021-12-03 (4290).

## Frozen test on 2022 to latest

Windows with base close on or after 2021-12-31; last window ends 2026-10-27. Mean error is the unexplained drift (realised minus predicted). 'Refit v2' is an OLS of the v2 controls
on the test windows themselves, an in-sample bound. 'Trimmed' removes the five largest windows by |return|.

| asset | h | n | model | RMSE | mean error (drift) | hit rate | R2 oos | RMSE trimmed | drift trimmed |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| gold | 5 | 1205 | no change | 2.64% | +0.34% | - | +0.000 | 2.54% | +0.37% |
| gold | 5 | 1205 | frozen v1 | 2.34% | +0.25% | 68% | +0.215 | 2.22% | +0.28% |
| gold | 5 | 1205 | **frozen v2** | 2.22% | +0.30% | 71% | +0.292 | 2.12% | +0.33% |
| gold | 5 | 1205 | refit v2 (bound) | 2.18% | +0.00% | 73% | +0.316 | - | - |
| gold | 20 | 1190 | no change | 5.14% | +1.38% | - | +0.000 | 5.02% | +1.40% |
| gold | 20 | 1190 | frozen v1 | 4.28% | +1.14% | 77% | +0.308 | 4.15% | +1.15% |
| gold | 20 | 1190 | **frozen v2** | 3.99% | +1.36% | 80% | +0.398 | 3.88% | +1.37% |
| gold | 20 | 1190 | refit v2 (bound) | 3.58% | +0.00% | 80% | +0.514 | - | - |
| silver | 5 | 1205 | no change | 5.54% | +0.36% | - | +0.000 | 5.12% | +0.45% |
| silver | 5 | 1205 | frozen v1 | 5.07% | +0.34% | 65% | +0.164 | 4.66% | +0.43% |
| silver | 5 | 1205 | **frozen v2** | 4.91% | +0.42% | 66% | +0.215 | 4.50% | +0.50% |
| silver | 5 | 1205 | refit v2 (bound) | 4.84% | +0.00% | 70% | +0.238 | - | - |
| silver | 20 | 1190 | no change | 10.09% | +1.54% | - | +0.000 | 9.72% | +1.37% |
| silver | 20 | 1190 | frozen v1 | 9.08% | +1.52% | 69% | +0.190 | 8.68% | +1.35% |
| silver | 20 | 1190 | **frozen v2** | 8.76% | +1.75% | 72% | +0.245 | 8.38% | +1.59% |
| silver | 20 | 1190 | refit v2 (bound) | 8.22% | -0.00% | 71% | +0.335 | - | - |

## Drift: does v2 shrink it?

| asset | h | v1 drift | v2 drift | change | v1 RMSE | v2 RMSE | change |
|---|---:|---:|---:|---:|---:|---:|---:|
| gold | 5 | +0.25% | +0.30% | +0.05% (+19% of |drift|) | 2.34% | 2.22% | -0.12 |
| gold | 20 | +1.14% | +1.36% | +0.22% (+20% of |drift|) | 4.28% | 3.99% | -0.29 |
| silver | 5 | +0.34% | +0.42% | +0.07% (+22% of |drift|) | 5.07% | 4.91% | -0.16 |
| silver | 20 | +1.52% | +1.75% | +0.24% (+15% of |drift|) | 9.08% | 8.76% | -0.32 |

## Stability of the v2 coefficients: train (2004-2021) vs test (2022-latest, OLS)

| asset | h | term | train | SE | test refit | SE | gap in train SEs |
|---|---:|---|---:|---:|---:|---:|---:|
| gold | 5 | d_real_yield | -3.433 | 0.868 | -3.236 | 0.984 | +0.2 |
| gold | 5 | dlog_dxy | -0.934 | 0.070 | -0.875 | 0.126 | +0.8 |
| gold | 5 | dlog_vix | -0.003 | 0.004 | +0.000 | 0.009 | +0.7 |
| gold | 5 | d_gpr | +0.154 | 0.078 | +0.237 | 0.131 | +1.1 |
| gold | 5 | d_gld | +0.579 | 0.061 | +0.869 | 0.142 | +4.7 ** |
| gold | 5 | cb_flow | -0.548 | 0.805 | -0.011 | 1.505 | +0.7 |
| gold | 5 | intercept | +0.120 | - | +0.380 | - | - |
| gold | 20 | d_real_yield | -5.314 | 1.105 | -3.704 | 1.172 | +1.5 |
| gold | 20 | dlog_dxy | -0.851 | 0.112 | -0.781 | 0.164 | +0.6 |
| gold | 20 | dlog_vix | +0.008 | 0.007 | +0.012 | 0.014 | +0.6 |
| gold | 20 | d_gpr | +0.372 | 0.190 | +0.446 | 0.245 | +0.4 |
| gold | 20 | d_gld | +0.496 | 0.061 | +0.959 | 0.166 | +7.6 ** |
| gold | 20 | cb_flow | -0.763 | 0.720 | +0.299 | 1.720 | +1.5 |
| gold | 20 | intercept | +0.512 | - | +1.527 | - | - |
| silver | 5 | d_real_yield | -4.943 | 1.436 | -1.088 | 1.964 | +2.7 ** |
| silver | 5 | dlog_dxy | -1.618 | 0.114 | -1.694 | 0.284 | -0.7 |
| silver | 5 | dlog_vix | -0.053 | 0.007 | -0.054 | 0.021 | -0.1 |
| silver | 5 | d_gpr | +0.268 | 0.129 | +0.471 | 0.255 | +1.6 |
| silver | 5 | d_gld | +0.925 | 0.102 | +1.584 | 0.281 | +6.5 ** |
| silver | 5 | cb_flow | -0.861 | 1.462 | +1.119 | 3.267 | +1.4 |
| silver | 5 | intercept | +0.071 | - | +0.323 | - | - |
| silver | 20 | d_real_yield | -5.635 | 1.950 | -0.092 | 3.778 | +2.8 ** |
| silver | 20 | dlog_dxy | -1.579 | 0.214 | -1.283 | 0.428 | +1.4 |
| silver | 20 | dlog_vix | -0.066 | 0.016 | -0.103 | 0.036 | -2.4 ** |
| silver | 20 | d_gpr | +1.068 | 0.362 | +1.839 | 0.574 | +2.1 ** |
| silver | 20 | d_gld | +0.745 | 0.108 | +1.657 | 0.368 | +8.4 ** |
| silver | 20 | cb_flow | -1.292 | 1.368 | +1.467 | 3.434 | +2.0 ** |
| silver | 20 | intercept | +0.394 | - | +1.166 | - | - |

## Reading

**GLD flows are the strong new factor; central-bank buying is not.** In training, a 10-tonne change in GLD holdings over a 5-day window goes with a +0.58% gold move (t = +9.5; t = +8.1 at 20 days, +9.1 and +6.9 for silver). That is contemporaneous, not predictive: ETF creations happen on the days gold rises, so the factor explains the window after the fact and is of no use to a forecast unless the flow itself is forecast. The lagged central-bank pace is -0.76% per 100 tonnes bought in a 20-day window (t = -1.1; -0.7 at 5 days), a slow variable that cannot explain week-to-week moves and barely moves the fit. **Drift in the test period.** v2 beats v1's RMSE on 4 of 4 asset-horizon pairs and reduces the absolute drift on 0 of 4: gold's 20-day mean error goes from +1.14% under v1 to +1.36% under v2 (5-day: +0.25% to +0.30%; silver 20-day: +1.52% to +1.75%). So v2 explains more of the week-to-week variance but misses more of the trend: the 2024-2026 rally ran with flat-to-negative GLD holdings and only modest reported official buying over the windows in which it happened, so a model that leans on those flows predicts less and leaves a larger unexplained drift. The drift is not an omitted-flow story, at least not one these two measured flows capture. **Stability.** Coefficients whose test-period refit sits more than two training SEs from the frozen value: d_gld (gold 5d), d_gld (gold 20d), d_real_yield (silver 5d), d_gld (silver 5d), d_real_yield (silver 20d), dlog_vix (silver 20d), d_gpr (silver 20d), d_gld (silver 20d), cb_flow (silver 20d). The GLD beta is the least stable of all: for gold at 20 days it goes from +0.50% per 10 tonnes in training to +0.96% in the test period, so even the contemporaneous relationship is not a fixed number. The v1 controls' betas are the ones that matter for forecasting, and the rolling-window section below shows how far they have moved.

# Rolling coefficients and the switching rule

The v1 controls-only specification (h-day log return on changes in the real yield, log DXY, log VIX and GPR) refitted by OLS on five-year windows stepped quarterly, window ends from 2005-09-30 to 2026-10-27 (86 windows). Figures: `reports/figures/rolling_<asset>_h<h>.png`, one panel per coefficient with a one-standard-error band and the v1 value as a dashed line.

## Rule check (h = 5, the forward log's horizon)

A coefficient passes when the v1 (2000-2021) value lies inside the interquartile range of the rolling estimates whose window ends in the last five years.

| asset | term | v1 value | rolling IQR, last 5y | latest window | inside? |
|---|---|---:|---:|---:|:---:|
| gold | intercept | +0.152 | [+0.210, +0.334] | +0.419 | **no** |
| gold | d_real_yield | -2.969 | [-7.532, -4.277] | -4.276 (SE 1.11) | **no** |
| gold | dlog_dxy | -0.980 | [-0.906, -0.719] | -0.913 (SE 0.15) | **no** |
| gold | dlog_vix | -0.001 | [+0.002, +0.007] | -0.002 (SE 0.01) | **no** |
| gold | d_gpr | +0.195 | [+0.276, +0.326] | +0.262 (SE 0.14) | **no** |
| silver | intercept | +0.111 | [+0.239, +0.409] | +0.436 | **no** |
| silver | d_real_yield | -3.528 | [-11.455, -3.428] | -3.157 (SE 2.02) | yes |
| silver | dlog_dxy | -1.600 | [-1.784, -1.615] | -1.764 (SE 0.30) | **no** |
| silver | dlog_vix | -0.048 | [-0.037, -0.032] | -0.055 (SE 0.02) | **no** |
| silver | d_gpr | +0.342 | [+0.484, +0.580] | +0.492 (SE 0.27) | **no** |

**Decision:** 9 of 10 coefficients fall outside the recent interquartile range, so the forward log switches to the most recent five-year window's coefficients, versioned `v2-rolling` (`data/v2_rolling_coefficients.csv`; `data/forward_model.txt` now reads `v2-rolling`).

## Coefficients: v1, latest five-year window, full 2000-2026 sample

| asset | h | term | v1 (2000-2021) | latest 5y window | full sample to latest |
|---|---:|---|---:|---:|---:|
| gold | 5 | intercept | +0.152 | +0.419 | +0.198 |
| gold | 5 | d_real_yield | -2.969 | -4.276 | -3.083 |
| gold | 5 | dlog_dxy | -0.980 | -0.913 | -0.977 |
| gold | 5 | dlog_vix | -0.001 | -0.002 | -0.001 |
| gold | 5 | d_gpr | +0.195 | +0.262 | +0.207 |
| gold | 20 | intercept | +0.593 | +1.755 | +0.803 |
| gold | 20 | d_real_yield | -4.299 | -4.980 | -4.197 |
| gold | 20 | dlog_dxy | -0.852 | -1.003 | -0.874 |
| gold | 20 | dlog_vix | +0.008 | +0.024 | +0.011 |
| gold | 20 | d_gpr | +0.321 | +0.412 | +0.361 |
| silver | 5 | intercept | +0.111 | +0.436 | +0.171 |
| silver | 5 | d_real_yield | -3.528 | -3.157 | -3.511 |
| silver | 5 | dlog_dxy | -1.600 | -1.764 | -1.631 |
| silver | 5 | dlog_vix | -0.048 | -0.055 | -0.050 |
| silver | 5 | d_gpr | +0.342 | +0.492 | +0.390 |
| silver | 20 | intercept | +0.414 | +1.720 | +0.698 |
| silver | 20 | d_real_yield | -4.701 | -2.323 | -4.110 |
| silver | 20 | dlog_dxy | -1.458 | -1.695 | -1.481 |
| silver | 20 | dlog_vix | -0.059 | -0.073 | -0.061 |
| silver | 20 | d_gpr | +1.064 | +1.797 | +1.228 |

## Range of the rolling estimates

| asset | h | term | min | 25% | median | 75% | max | windows where |t| >= 2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| gold | 5 | intercept | -0.046 | +0.081 | +0.180 | +0.284 | +0.453 | - |
| gold | 5 | d_real_yield | -11.868 | -7.619 | -6.575 | -0.765 | +2.452 | 55 of 86 |
| gold | 5 | dlog_dxy | -1.627 | -1.112 | -0.839 | -0.750 | -0.552 | 86 of 86 |
| gold | 5 | dlog_vix | -0.023 | -0.001 | +0.004 | +0.008 | +0.021 | 1 of 86 |
| gold | 5 | d_gpr | -0.059 | +0.055 | +0.131 | +0.261 | +0.422 | 27 of 86 |
| gold | 20 | intercept | -0.142 | +0.285 | +0.733 | +1.146 | +1.902 | - |
| gold | 20 | d_real_yield | -15.558 | -10.463 | -6.766 | -3.463 | -0.121 | 62 of 86 |
| gold | 20 | dlog_dxy | -1.577 | -0.989 | -0.807 | -0.706 | -0.431 | 86 of 86 |
| gold | 20 | dlog_vix | -0.034 | +0.004 | +0.011 | +0.022 | +0.065 | 17 of 86 |
| gold | 20 | d_gpr | -0.105 | +0.148 | +0.497 | +0.652 | +1.224 | 27 of 86 |
| silver | 5 | intercept | -0.252 | +0.011 | +0.191 | +0.299 | +0.508 | - |
| silver | 5 | d_real_yield | -20.172 | -10.488 | -7.532 | -1.100 | +1.705 | 46 of 86 |
| silver | 5 | dlog_dxy | -2.422 | -1.818 | -1.592 | -1.327 | -0.997 | 86 of 86 |
| silver | 5 | dlog_vix | -0.090 | -0.054 | -0.038 | -0.033 | -0.013 | 70 of 86 |
| silver | 5 | d_gpr | +0.048 | +0.177 | +0.274 | +0.490 | +0.888 | 27 of 86 |
| silver | 20 | intercept | -0.966 | +0.027 | +0.926 | +1.137 | +2.025 | - |
| silver | 20 | d_real_yield | -22.720 | -11.765 | -5.864 | -2.657 | +0.270 | 30 of 86 |
| silver | 20 | dlog_dxy | -2.206 | -1.851 | -1.642 | -1.142 | -0.608 | 86 of 86 |
| silver | 20 | dlog_vix | -0.103 | -0.068 | -0.050 | -0.034 | +0.004 | 33 of 86 |
| silver | 20 | d_gpr | +0.051 | +0.671 | +1.386 | +1.833 | +2.721 | 36 of 86 |

## Reading (rolling)

**The rule fires: 9 of 10 coefficients are outside the last five years' interquartile range, so the forward log now uses the latest five-year window as `v2-rolling`.** The picture behind it is one of regimes, not noise. Gold's real-yield beta at 5 days was near zero in windows ending before mid-2013 (median -0.6% per pp), fell to between -7% and -12% in windows ending 2014-2021 (median -7.8%), and has come back to about -4.3% in the latest window, so the v1 value of -3.0% is an average over two different worlds. The dollar beta has been more stable (between about -0.7 and -1.6 throughout) and the GPR beta has risen from roughly zero before 2020 to about +0.26% per 100 points. The intercept, gold's drift per window after the controls, is at a two-decade high (+0.42% per 5-day window, +1.76% per 20-day window), which is the same unexplained drift the out-of-sample test found. Two cautions: the interquartile range of about twenty quarterly windows is narrow, so the test is demanding by construction, and the latest window's betas carry standard errors of 1.1 (real yield) and 0.15 (dollar) percentage points, so switching trades a stale average for a noisier recent estimate. The switch is what the pre-set rule says to do; the forward scorecard will show whether it earns its keep. The v2-rolling coefficients are the 2021-10-28 to 2026-10-27 window.
