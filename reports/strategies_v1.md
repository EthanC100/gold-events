# Strategy test v1 (in sample, 2000-08-30 to 2021-12-31)

Everything below is in sample and nothing after 2021-12-31 is read. Each test is run on the hand list
(`data/events.csv`) and on the mechanical list (`data/events_mechanical.csv`); the mechanical results are
the ones that matter, because the hand list was selected with hindsight.

Conventions: day 0 is the event date (or the next trading day). The net 5-day move is the event study's
net return from the close of day -1 to the close of day 4, after removing what the dollar and the real
yield explain (betas from the 250 days before day 0); the net 20-day move ends at the close of day 19.
Trades enter at the close that ends the 5-day window and exit at the close that ends the 20-day window:
15 trading days of the metal's own return, 0.1% round-trip cost, no overlapping positions (an event
whose entry falls on or before the open position's exit is skipped). Sharpe ratios are annualised from
the strategy's daily return series over the whole sample with flat days at zero, so they are comparable
with buy-and-hold. Max drawdown is the largest fall of cumulative P&L, as a simple return. *t* is the average
net trade return over its
standard error. 'Ex top 5' removes the five most profitable trades.

## H3: retention of the 5-day move at 20 days

For events whose net 5-day move exceeded +1% (or fell below -1%): ratio = net 20-day / net 5-day.
'Retains half' is the share of events with ratio >= 0.5; 'same sign' the share with ratio > 0. The mean
ratio is swung by events whose 5-day move was barely over the threshold, so read the median. The last
column is the metal's own (raw, gross) return from the close of day 4 to the close of day 19 on those
events: what a trade entered at day 5 actually earns, as opposed to the net move the ratio measures.

| asset | list | direction | n | mean ratio | min | 25% | median | 75% | max | retains half | same sign | mean net 5d | mean net 20d | raw day 4 to 19 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gold | hand | up | 23 | 0.97 | -15.4 | 0.91 | 1.69 | 2.16 | 4.5 | 87% | 91% | +3.29% | +4.38% | +1.13% |
| gold | hand | down | 12 | 0.97 | -2.4 | 0.59 | 1.41 | 1.89 | 3.1 | 75% | 75% | -3.10% | -3.05% | +0.40% |
| gold | mechanical | up | 40 | 1.15 | -5.7 | 0.57 | 1.19 | 1.90 | 5.7 | 75% | 88% | +2.98% | +3.50% | +0.41% |
| gold | mechanical | down | 36 | 0.68 | -3.6 | -0.44 | 0.67 | 1.27 | 4.9 | 53% | 67% | -2.45% | -1.63% | +0.74% |
| silver | hand | up | 18 | 1.56 | -1.7 | 0.40 | 1.39 | 1.88 | 8.5 | 67% | 83% | +5.23% | +7.28% | +0.51% |
| silver | hand | down | 19 | 1.13 | -3.4 | -0.26 | 0.51 | 2.14 | 12.3 | 53% | 68% | -5.50% | -3.74% | +2.67% |
| silver | mechanical | up | 57 | 1.72 | -5.8 | 0.34 | 1.22 | 2.31 | 11.1 | 72% | 75% | +3.74% | +4.86% | +1.24% |
| silver | mechanical | down | 42 | 0.91 | -5.0 | -0.13 | 0.69 | 1.74 | 7.4 | 55% | 71% | -3.82% | -3.39% | +1.15% |

## Rules on gold

Benchmarks: no position = 0. Buy-and-hold gold over the full sample: +8.9% a year (log), Sharpe 0.51, max drawdown -44.4%. The unconditional 15-day log return (every window in the sample) averages +0.53% with SD +4.07%, positive 55% of the time: that is what 'buy and hold over the same holding periods' earns on average with no signal, before cost.

### All trades

| rule | trades | hit rate | avg/trade after cost | t | total | Sharpe (ann.) | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| a: buy if net 5d > +1% [hand] | 21 | 0.67 | +0.29% | 0.3 | +6.1% | 0.05 | -24.1% |
| b: sell if net 5d < -1% [hand] | 12 | 0.50 | -0.50% | -0.3 | -5.9% | -0.06 | -20.3% |
| c: buy after any event [hand] | 43 | 0.51 | -0.31% | -0.4 | -13.4% | -0.09 | -37.5% |
| a: buy if net 5d > +1% [mechanical] | 34 | 0.53 | -0.21% | -0.3 | -7.0% | -0.06 | -20.8% |
| b: sell if net 5d < -1% [mechanical] | 33 | 0.30 | -0.96% | -1.2 | -31.8% | -0.25 | -39.9% |
| c: buy after any event [mechanical] | 100 | 0.62 | +0.47% | 1.2 | +47.3% | 0.25 | -28.0% |
| d: tension rule (weekly GPR change > 1 SD = 36.6) [independent] | 59 | 0.56 | +0.16% | 0.3 | +9.4% | 0.06 | -36.9% |

### Same trades, five most profitable removed

| rule | trades | hit rate | avg/trade after cost | t | total | Sharpe (ann.) | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| a: buy if net 5d > +1% [hand] | 16 | 0.56 | -1.22% | -1.0 | -19.6% | -0.19 | -29.8% |
| b: sell if net 5d < -1% [hand] | 7 | 0.14 | -3.51% | -2.8 | -24.6% | -0.30 | -23.8% |
| c: buy after any event [hand] | 38 | 0.45 | -1.23% | -1.9 | -46.7% | -0.37 | -44.8% |
| a: buy if net 5d > +1% [mechanical] | 29 | 0.45 | -1.13% | -1.6 | -32.9% | -0.29 | -30.0% |
| b: sell if net 5d < -1% [mechanical] | 28 | 0.18 | -2.37% | -4.1 | -66.5% | -0.57 | -49.5% |
| c: buy after any event [mechanical] | 95 | 0.60 | +0.10% | 0.3 | +9.8% | 0.06 | -34.9% |
| d: tension rule [independent] | 54 | 0.52 | -0.53% | -1.0 | -28.4% | -0.18 | -38.1% |

### Buy-and-hold over the same holding windows

Average gross long return of the metal over each rule's own windows, no cost (for buy rules this is
the rule's return before cost; for the sell rule it is what being long instead would have made).

| rule | list | trades | long over same windows, avg | rule avg after cost | unconditional avg |
|---|---|---:|---:|---:|---:|
| a: buy if net 5d > +1% | hand | 21 | +0.39% | +0.29% | +0.53% |
| b: sell if net 5d < -1% | hand | 12 | +0.40% | -0.50% | +0.53% |
| c: buy after any event | hand | 43 | -0.21% | -0.31% | +0.53% |
| a: buy if net 5d > +1% | mechanical | 34 | -0.11% | -0.21% | +0.53% |
| b: sell if net 5d < -1% | mechanical | 33 | +0.86% | -0.96% | +0.53% |
| c: buy after any event | mechanical | 100 | +0.57% | +0.47% | +0.53% |
| d: tension rule (20-day hold) | independent | 59 | +0.26% | +0.16% | +0.70% (20-day) |

### Stability by year

Fixed rules have nothing to fit, so blocked cross-validation by year reduces to asking whether each
year's trades, taken alone, make money. For the tension rule the 1-SD threshold is re-estimated on
the other years (leave-one-year-out). 'Years +' is the number of years with a positive average trade
out of the years that had trades.

| rule | list | years with trades | years + | mean of yearly avgs | worst year | best year |
|---|---|---:|---:|---:|---|---|
| a: buy if net 5d > +1% | hand | 14 | 8 | +0.66% | 2013 (-9.0%, n=2) | 2006 (+6.2%, n=1) |
| b: sell if net 5d < -1% | hand | 9 | 5 | -0.11% | 2011 (-3.9%, n=1) | 2009 (+3.3%, n=1) |
| c: buy after any event | hand | 20 | 10 | -0.11% | 2013 (-9.0%, n=2) | 2006 (+6.2%, n=1) |
| a: buy if net 5d > +1% | mechanical | 17 | 9 | +0.02% | 2011 (-5.2%, n=2) | 2006 (+5.0%, n=1) |
| b: sell if net 5d < -1% | mechanical | 17 | 6 | -0.96% | 2020 (-7.6%, n=1) | 2012 (+4.8%, n=1) |
| c: buy after any event | mechanical | 21 | 14 | +0.58% | 2008 (-3.1%, n=5) | 2005 (+4.2%, n=1) |
| d: tension rule (LOYO threshold) | independent | 22 | 9 | +0.16% | 2013 (-4.6%, n=2) | 2006 (+6.4%, n=1) |

## Rules on silver

Benchmarks: no position = 0. Buy-and-hold silver over the full sample: +7.3% a year (log), Sharpe 0.23, max drawdown -75.8%. The unconditional 15-day log return (every window in the sample) averages +0.43% with SD +7.50%, positive 53% of the time: that is what 'buy and hold over the same holding periods' earns on average with no signal, before cost.

### All trades

| rule | trades | hit rate | avg/trade after cost | t | total | Sharpe (ann.) | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| a: buy if net 5d > +1% [hand] | 18 | 0.44 | +0.41% | 0.2 | +7.4% | 0.04 | -31.8% |
| b: sell if net 5d < -1% [hand] | 17 | 0.41 | -0.51% | -0.2 | -8.6% | -0.04 | -33.7% |
| c: buy after any event [hand] | 43 | 0.49 | +0.00% | 0.0 | +0.0% | 0.00 | -54.8% |
| a: buy if net 5d > +1% [mechanical] | 48 | 0.60 | +0.44% | 0.3 | +20.9% | 0.08 | -45.6% |
| b: sell if net 5d < -1% [mechanical] | 37 | 0.32 | -0.97% | -0.8 | -35.9% | -0.16 | -55.0% |
| c: buy after any event [mechanical] | 100 | 0.61 | +0.42% | 0.5 | +42.2% | 0.12 | -55.2% |
| d: tension rule (weekly GPR change > 1 SD = 36.6) [independent] | 59 | 0.49 | -0.75% | -0.7 | -44.0% | -0.15 | -65.5% |

### Same trades, five most profitable removed

| rule | trades | hit rate | avg/trade after cost | t | total | Sharpe (ann.) | max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| a: buy if net 5d > +1% [hand] | 13 | 0.23 | -3.38% | -2.0 | -44.0% | -0.29 | -37.0% |
| b: sell if net 5d < -1% [hand] | 12 | 0.17 | -4.88% | -3.5 | -58.5% | -0.36 | -49.1% |
| c: buy after any event [hand] | 38 | 0.42 | -1.54% | -1.4 | -58.5% | -0.22 | -65.8% |
| a: buy if net 5d > +1% [mechanical] | 43 | 0.56 | -1.04% | -0.8 | -44.8% | -0.17 | -57.3% |
| b: sell if net 5d < -1% [mechanical] | 32 | 0.22 | -3.10% | -3.5 | -99.3% | -0.48 | -66.1% |
| c: buy after any event [mechanical] | 95 | 0.59 | -0.31% | -0.4 | -29.4% | -0.08 | -67.3% |
| d: tension rule [independent] | 54 | 0.44 | -2.25% | -2.3 | -121.4% | -0.43 | -72.5% |

### Buy-and-hold over the same holding windows

Average gross long return of the metal over each rule's own windows, no cost (for buy rules this is
the rule's return before cost; for the sell rule it is what being long instead would have made).

| rule | list | trades | long over same windows, avg | rule avg after cost | unconditional avg |
|---|---|---:|---:|---:|---:|
| a: buy if net 5d > +1% | hand | 18 | +0.51% | +0.41% | +0.43% |
| b: sell if net 5d < -1% | hand | 17 | +0.41% | -0.51% | +0.43% |
| c: buy after any event | hand | 43 | +0.10% | +0.00% | +0.43% |
| a: buy if net 5d > +1% | mechanical | 48 | +0.54% | +0.44% | +0.43% |
| b: sell if net 5d < -1% | mechanical | 37 | +0.87% | -0.97% | +0.43% |
| c: buy after any event | mechanical | 100 | +0.52% | +0.42% | +0.43% |
| d: tension rule (20-day hold) | independent | 59 | -0.65% | -0.75% | +0.57% (20-day) |

### Stability by year

Fixed rules have nothing to fit, so blocked cross-validation by year reduces to asking whether each
year's trades, taken alone, make money. For the tension rule the 1-SD threshold is re-estimated on
the other years (leave-one-year-out). 'Years +' is the number of years with a positive average trade
out of the years that had trades.

| rule | list | years with trades | years + | mean of yearly avgs | worst year | best year |
|---|---|---:|---:|---:|---|---|
| a: buy if net 5d > +1% | hand | 11 | 5 | -1.01% | 2008 (-16.5%, n=1) | 2011 (+12.0%, n=3) |
| b: sell if net 5d < -1% | hand | 13 | 6 | -0.49% | 2011 (-14.4%, n=1) | 2020 (+18.8%, n=1) |
| c: buy after any event | hand | 20 | 10 | +0.23% | 2013 (-12.2%, n=2) | 2011 (+12.2%, n=3) |
| a: buy if net 5d > +1% | mechanical | 19 | 10 | +0.39% | 2021 (-10.7%, n=2) | 2001 (+8.5%, n=1) |
| b: sell if net 5d < -1% | mechanical | 17 | 9 | +0.39% | 2006 (-9.5%, n=1) | 2004 (+22.9%, n=1) |
| c: buy after any event | mechanical | 21 | 14 | +0.52% | 2008 (-7.3%, n=5) | 2006 (+6.0%, n=3) |
| d: tension rule (LOYO threshold) | independent | 22 | 10 | -1.07% | 2008 (-14.6%, n=1) | 2006 (+11.9%, n=1) |

## Verdicts

**Rule a: buy if net 5d > +1%.** Buy when the net 5-day move exceeds +1%. Hand list: 21 trades, +0.29% per trade after cost (t = 0.3), Sharpe 0.05, -1.22% ex top 5, 8 of 14 years positive. Mechanical list: 34 trades, -0.21% per trade (t = -0.3), Sharpe -0.06, -1.13% ex top 5, 9 of 17 years positive. On the mechanical list it loses money after cost. Verdict: does not survive.

**Rule b: sell if net 5d < -1%.** Sell short when the net 5-day move is below -1%. Hand list: 12 trades, -0.50% per trade after cost (t = -0.3), Sharpe -0.06, -3.51% ex top 5, 5 of 9 years positive. Mechanical list: 33 trades, -0.96% per trade (t = -1.2), Sharpe -0.25, -2.37% ex top 5, 6 of 17 years positive. On the mechanical list it loses money after cost. Verdict: does not survive.

**Rule c: buy after any event.** Buy after every event. Hand list: 43 trades, -0.31% per trade after cost (t = -0.4), Sharpe -0.09, -1.23% ex top 5, 10 of 20 years positive. Mechanical list: 100 trades, +0.47% per trade (t = 1.2), Sharpe 0.25, +0.10% ex top 5, 14 of 21 years positive. On the mechanical list it covers its costs and survives removing the five best trades, but it does not beat simply being long over random 15-day windows (+0.53%): it is gold's drift, not the signal. Verdict: survives costs but not the drift benchmark.

**Rule d: tension.** Buy at the close of any week whose mean GPR rose by more than one SD of weekly changes (36.6 index points), hold 20 trading days. Independent of both lists. 59 trades, +0.16% per trade after cost (t = 0.3), hit rate 0.56, Sharpe 0.06, max drawdown -36.9%; -0.53% ex top 5; 9 of 22 years positive with the threshold fitted on the other years. The unconditional 20-day return is +0.70%. Verdict: positive on average but indistinguishable from zero, and negative without its five best trades.

**Overall.** Buy-and-hold gold made +8.9% a year over this sample with a Sharpe of 0.51; any rule that is long gold for a fraction of the time inherits a slice of that drift, so the bar is the unconditional 15-day return of +0.53%, not zero. The retention numbers (H3) look strong on their own: after a net 5-day move above +1%, the median gold event keeps 1.7x of it at 20 days on the hand list and 1.2x on the mechanical list. But that is the *net* move, after the dollar and real yields are removed, and a position in the metal earns the raw move: from day 4 to day 19 those same events returned +0.41% gross per event on the mechanical list (+1.13% on the hand list, falling to +0.39% once overlapping events are skipped), against +0.53% for any random window. The event-specific component persists; the part you can trade does not beat drift.
