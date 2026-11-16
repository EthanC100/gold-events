# Forward scorecard

Updated 2026-11-16. 4 weeks logged, 2 with realised moves.

## Frozen model, 5-day prediction vs realised

Prediction = the frozen coefficients named in each row's `model_version` (v1 fitted to 2021-12-31, or
v2-rolling, the latest five-year window) applied to last week's control changes.
Hit rate = share of weeks with the right sign. MAE is compared with predicting no change.

| asset | weeks | direction hit rate | mean error (realised - pred) | MAE | MAE of no change |
|---|---:|---:|---:|---:|---:|
| gold | 2 | 1.00 | +1.68% | 1.68% | 2.06% |
| silver | 2 | 1.00 | +0.37% | 2.19% | 3.05% |

## Probabilities: Brier score (lower is better)

0 of 5 questions resolved. Pooled over every logged probability on a resolved question.

| forecaster | probabilities scored | Brier |
|---|---:|---:|
| me | 0 | - |
| claude | 0 | - |
| market | 0 | - |

## Rows

| week | model | pred gold | realised gold | pred silver | realised silver | GPR | GPR change | my tension | Claude tension |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-10-28 | v2-rolling | +0.27% | +0.53% | -0.71% | -2.53% | 165 | -56 |  | 6.0 |
| 2026-11-02 | v2-rolling | +0.47% | +3.59% | +1.01% | +3.57% | 165 | -56 |  |  |
| 2026-11-09 | v2-rolling | -0.12% | - | -2.50% | - | 98 | -68 |  |  |
| 2026-11-16 | v2-rolling | +0.35% | - | +0.81% | - | 114 | -80 |  |  |
