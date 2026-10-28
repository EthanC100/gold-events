# Gold and silver around US midterm elections

Built 2026-10-28 from `data/daily.parquet` (gold/silver source: see `reports/data_coverage.md`).

Day 0 = first trading day after election day. Returns are log returns from the close of the
day before day 0 to the close of day h-1 (h = 1, 5, 20 trading days). *Net* subtracts the move
explained by the same-window change in the dollar index (DXY) and the 10y real yield, using
betas from an OLS on the 250 trading days before day 0. Real yield is the TIPS yield from
2003 and a nominal-minus-trailing-CPI proxy before that.

**Not computable in this build:** 1994, 1998 (gold/silver data start 2000-08-30; re-run `python -m gold_events.data` from a machine where Stooq is reachable to extend the series back to 1990).

## Gold

| Midterm | Day 0 | House flipped | raw 1d | net 1d | raw 5d | net 5d | raw 20d | net 20d |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| 1994-11-08 | 1994-11-09 | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| 1998-11-03 | 1998-11-04 | no | n/a | n/a | n/a | n/a | n/a | n/a |
| 2002-11-05 | 2002-11-06 | no | -0.22% | -0.38% | +1.96% | +1.13% | +1.25% | +1.32% |
| 2006-11-07 | 2006-11-08 | yes | -1.50% | -1.54% | -0.30% | -0.62% | +0.76% | -3.60% |
| 2010-11-02 | 2010-11-03 | yes | -1.43% | -1.59% | +3.86% | +4.68% | +2.25% | +6.26% |
| 2014-11-04 | 2014-11-05 | no | -1.90% | -1.47% | -0.39% | +0.17% | +3.46% | +5.65% |
| 2018-11-06 | 2018-11-07 | yes | +0.20% | -0.14% | -2.03% | -1.15% | +1.16% | +0.88% |
| 2022-11-08 | 2022-11-09 | yes | -0.13% | +0.59% | +3.48% | +1.01% | +4.67% | +0.23% |

### Gold: averages

| Group | n | raw 1d | net 1d | raw 5d | net 5d | raw 20d | net 20d |
|---|---:|---:|---:|---:|---:|---:|---:|
| All midterms | 6 | -0.83% | -0.75% | +1.10% | +0.87% | +2.26% | +1.79% |
| House changed hands | 4 | -0.72% | -0.67% | +1.25% | +0.98% | +2.21% | +0.94% |
| House did not change | 2 | -1.06% | -0.92% | +0.78% | +0.65% | +2.35% | +3.48% |

## Silver

| Midterm | Day 0 | House flipped | raw 1d | net 1d | raw 5d | net 5d | raw 20d | net 20d |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| 1994-11-08 | 1994-11-09 | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| 1998-11-03 | 1998-11-04 | no | n/a | n/a | n/a | n/a | n/a | n/a |
| 2002-11-05 | 2002-11-06 | no | -0.27% | -0.39% | +2.31% | +1.88% | +1.55% | +1.77% |
| 2006-11-07 | 2006-11-08 | yes | -0.96% | -0.94% | +1.83% | +1.44% | +7.53% | +1.10% |
| 2010-11-02 | 2010-11-03 | yes | -1.62% | -2.00% | +15.18% | +16.74% | +13.38% | +21.31% |
| 2014-11-04 | 2014-11-05 | no | -3.25% | -2.73% | -1.67% | -1.02% | +2.68% | +5.08% |
| 2018-11-06 | 2018-11-07 | yes | +0.48% | +0.01% | -3.56% | -2.21% | -0.47% | -0.21% |
| 2022-11-08 | 2022-11-09 | yes | -0.84% | +0.53% | +0.10% | -4.56% | +5.58% | -2.56% |

### Silver: averages

| Group | n | raw 1d | net 1d | raw 5d | net 5d | raw 20d | net 20d |
|---|---:|---:|---:|---:|---:|---:|---:|
| All midterms | 6 | -1.08% | -0.92% | +2.37% | +2.05% | +5.04% | +4.42% |
| House changed hands | 4 | -0.74% | -0.60% | +3.39% | +2.85% | +6.50% | +4.91% |
| House did not change | 2 | -1.76% | -1.56% | +0.32% | +0.43% | +2.11% | +3.43% |

## Attribution at 20 days

What the controls explain: beta x same-window change, for each event.

| Midterm | Asset | beta DXY | beta real yield | dlog DXY 20d | d real yield 20d (pp) | explained | raw | net |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2002 | gold | -0.59 | -0.006 | +0.25% | -0.15 | -0.07% | +1.25% | +1.32% |
| 2002 | silver | -0.52 | +0.006 | +0.25% | -0.15 | -0.22% | +1.55% | +1.77% |
| 2006 | gold | -1.36 | -0.003 | -3.16% | -0.18 | +4.36% | +0.76% | -3.60% |
| 2006 | silver | -2.17 | +0.024 | -3.16% | -0.18 | +6.43% | +7.53% | +1.10% |
| 2010 | gold | -0.58 | -0.031 | +5.07% | +0.34 | -4.00% | +2.25% | +6.26% |
| 2010 | silver | -1.32 | -0.036 | +5.07% | +0.34 | -7.93% | +13.38% | +21.31% |
| 2014 | gold | -0.82 | -0.049 | +2.25% | +0.07 | -2.19% | +3.46% | +5.65% |
| 2014 | silver | -0.98 | -0.029 | +2.25% | +0.07 | -2.41% | +2.68% | +5.08% |
| 2018 | gold | -1.00 | -0.044 | +0.51% | -0.18 | +0.28% | +1.16% | +0.88% |
| 2018 | silver | -1.41 | -0.025 | +0.51% | -0.18 | -0.27% | -0.47% | -0.21% |
| 2022 | gold | -0.54 | -0.045 | -4.21% | -0.48 | +4.44% | +4.67% | +0.23% |
| 2022 | silver | -1.09 | -0.074 | -4.21% | -0.48 | +8.14% | +5.58% | -2.56% |

## Reading

Across the 6 midterms with price data (2002-2022), gold on average rose +2.26% over the 20 trading days after the election; once the part explained by the dollar and real yields is removed the net move is +1.79%, so the move is mostly not explained by the dollar or real yields. Silver rose +5.04% raw and +4.42% net over the same window. Splitting by outcome, gold's net 20-day move averaged +0.94% when the House changed hands (n=4) versus +3.48% when it did not (n=2); on day one the split is -0.67% versus -0.92%. The individual net 20-day gold moves range from -3.60% to +6.26% (5 of 6 positive), so with this few events the averages are driven by one or two years and should be read as anecdotes, not as evidence of a reliable midterm effect.
