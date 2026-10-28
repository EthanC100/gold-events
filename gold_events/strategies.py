"""Strategy test v1, in sample only (2000-08-30 to 2021-12-31).

Run ``python -m gold_events.strategies`` to write ``reports/strategies_v1.md``.

Conventions (shared with :mod:`gold_events.event_study`)
--------------------------------------------------------
* Day 0 = the event date if it is a trading day, else the next one.
* The *net 5-day move* is the event study's 5-day net return: close of day -1
  to close of day 4, minus what the dollar and the real yield explain, with
  betas from the 250 days before day 0.  The *net 20-day move* likewise ends
  at the close of day 19.
* "Enter at the close of day 5, hold to day 20" is implemented as: enter at
  the close that ends the 5-day window (day 4), exit at the close that ends
  the 20-day window (day 19).  Fifteen trading days of returns per trade.
* Trades never overlap: an event whose entry date is on or before the exit
  date of the open position is skipped.
* Returns are log returns of the asset itself (the position is in the metal,
  not in the net return).  Cost: 0.1% round trip, charged per trade.
* Sharpe ratio: from the strategy's daily return series over the whole
  sample (flat days count as zero), annualised with sqrt(252).  Maximum
  drawdown: largest peak-to-trough fall of the cumulative log P&L,
  expressed as a simple return.
* "Five largest trades removed" drops the five most profitable trades.

Rule (d), the tension rule, uses the weekly mean of the daily GPR index over
each week's trading days; a signal week is one whose change in weekly mean
exceeds one standard deviation of all weekly changes in the sample (in the
yearly stability check the standard deviation is taken from the other
years).  Entry at the week's last close, exit 20 trading days later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_events import data
from gold_events.event_study import event_study, first_trading_day_on_or_after
from gold_events.events import load_events
from gold_events.mechanical import load_mechanical

REPORT_PATH = data.ROOT / "reports" / "strategies_v1.md"
SAMPLE_START = pd.Timestamp("2000-08-30")
SAMPLE_END = pd.Timestamp("2021-12-31")
ENTRY_DAY, EXIT_DAY = 4, 19  # ends of the 5- and 20-day windows
HOLD_DAYS = EXIT_DAY - ENTRY_DAY
COST = 0.001
THRESH = 0.01
TOP_K = 5
LISTS = ("hand", "mechanical")
RULES = {
    "a: buy if net 5d > +1%": ("a", "Buy when the net 5-day move exceeds +1%."),
    "b: sell if net 5d < -1%": ("b", "Sell short when the net 5-day move is below -1%."),
    "c: buy after any event": ("c", "Buy after every event."),
}


# --------------------------------------------------------------------------
# Event table: net moves and the trade window for every event
# --------------------------------------------------------------------------


def event_table(df: pd.DataFrame, events: pd.DataFrame, asset: str) -> pd.DataFrame:
    """One row per in-sample event with net 5/20-day moves and the holding-period return."""
    df = df.loc[:SAMPLE_END]
    idx = df.index
    price = np.log(df[asset].to_numpy())
    rows = []
    for _, e in events.iterrows():
        if e.date < SAMPLE_START or e.date > SAMPLE_END:
            continue
        day0 = first_trading_day_on_or_after(idx, e.date)
        pos0 = int(idx.get_loc(day0))
        if pos0 + EXIT_DAY >= len(idx):
            continue  # the 20-day window would run past the sample end
        res = event_study(df, day0, horizons=(5, 20), assets=(asset,))
        entry, exit_ = pos0 + ENTRY_DAY, pos0 + EXIT_DAY
        rows.append({
            "date": e.date, "event": e.get("event", ""), "day0": day0, "pos0": pos0,
            "net5": res.get(asset, 5, "net"), "net20": res.get(asset, 20, "net"),
            "raw5": res.get(asset, 5, "raw"), "raw20": res.get(asset, 20, "raw"),
            "entry_date": idx[entry], "exit_date": idx[exit_], "entry_pos": entry, "exit_pos": exit_,
            "hold_ret": float(price[exit_] - price[entry]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# H3: retention
# --------------------------------------------------------------------------


def retention(table: pd.DataFrame, direction: str) -> dict:
    """Ratio net20/net5 for events whose net 5-day move exceeded +1% (up) or fell below -1% (down)."""
    t = table.dropna(subset=["net5", "net20"])
    sel = t[t.net5 > THRESH] if direction == "up" else t[t.net5 < -THRESH]
    ratio = sel.net20 / sel.net5
    if ratio.empty:
        return {"n": 0}
    q = ratio.quantile([0, 0.25, 0.5, 0.75, 1.0])
    return {"n": len(ratio), "mean": float(ratio.mean()), "min": float(q[0]), "q25": float(q[0.25]),
            "median": float(q[0.5]), "q75": float(q[0.75]), "max": float(q[1.0]),
            "share_half": float((ratio >= 0.5).mean()), "share_sign": float((ratio > 0).mean()),
            "mean_net5": float(sel.net5.mean()), "mean_net20": float(sel.net20.mean()),
            "mean_hold": float(sel.hold_ret.mean())}


# --------------------------------------------------------------------------
# Trades
# --------------------------------------------------------------------------


@dataclass
class Trade:
    entry_pos: int
    exit_pos: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int  # +1 long, -1 short
    gross: float    # direction x log return of the asset
    net: float      # gross - COST


def _non_overlapping(candidates: pd.DataFrame, direction) -> list[Trade]:
    trades, open_exit = [], -1
    for r in candidates.sort_values("entry_pos").itertuples():
        if r.entry_pos <= open_exit:
            continue
        d = direction(r) if callable(direction) else direction
        gross = d * r.hold_ret
        trades.append(Trade(r.entry_pos, r.exit_pos, r.entry_date, r.exit_date, d, gross, gross - COST))
        open_exit = r.exit_pos
    return trades


def event_rule_trades(table: pd.DataFrame, rule: str) -> list[Trade]:
    t = table.dropna(subset=["net5"]) if rule in ("a", "b") else table
    if rule == "a":
        return _non_overlapping(t[t.net5 > THRESH], +1)
    if rule == "b":
        return _non_overlapping(t[t.net5 < -THRESH], -1)
    return _non_overlapping(t, +1)


def weekly_gpr(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly mean of the daily GPR on trading days, with the week's last trading day."""
    g = df.loc[SAMPLE_START - pd.Timedelta(days=7):SAMPLE_END, "gpr"]
    wk = g.groupby(g.index.to_period("W-FRI"))
    out = pd.DataFrame({"gpr_mean": wk.mean(), "last_day": wk.apply(lambda s: s.index[-1])})
    out["change"] = out["gpr_mean"].diff()
    return out[out.last_day >= SAMPLE_START].dropna()


def tension_trades(df: pd.DataFrame, asset: str, sd: float | None = None,
                   weeks: pd.DataFrame | None = None) -> tuple[list[Trade], float]:
    """Rule (d): buy at the close of a week whose GPR rose by more than ``sd``, hold 20 trading days."""
    df = df.loc[:SAMPLE_END]
    weeks = weekly_gpr(df) if weeks is None else weeks
    sd = float(weeks["change"].std()) if sd is None else sd
    idx = df.index
    price = np.log(df[asset].to_numpy())
    rows = []
    for w in weeks[weeks.change > sd].itertuples():
        entry = int(idx.get_loc(w.last_day))
        exit_ = entry + 20
        if exit_ >= len(idx):
            continue
        rows.append({"entry_pos": entry, "exit_pos": exit_, "entry_date": idx[entry], "exit_date": idx[exit_],
                     "hold_ret": float(price[exit_] - price[entry])})
    return _non_overlapping(pd.DataFrame(rows), +1) if rows else [], sd


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def daily_pnl(trades: list[Trade], df: pd.DataFrame, asset: str) -> pd.Series:
    """Strategy daily log return over the whole sample; flat days are zero; cost charged on exit day."""
    df = df.loc[SAMPLE_START:SAMPLE_END]
    r = np.log(df[asset]).diff().fillna(0.0).to_numpy()
    full = np.log(df[asset].to_numpy())  # positions refer to the full (pre-start) index, so map by date
    pnl = pd.Series(0.0, index=df.index)
    for t in trades:
        days = pnl.index[(pnl.index > t.entry_date) & (pnl.index <= t.exit_date)]
        pnl.loc[days] += t.direction * pd.Series(r, index=df.index).loc[days]
        pnl.loc[t.exit_date] -= COST
    return pnl


def max_drawdown(pnl: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative log P&L, as a simple return (exp(dd) - 1)."""
    cum = pnl.cumsum()
    return float(np.expm1((cum - cum.cummax()).min()))


def metrics(trades: list[Trade], df: pd.DataFrame, asset: str) -> dict:
    if not trades:
        return {"n": 0, "hit": float("nan"), "avg": float("nan"), "t": float("nan"), "total": 0.0,
                "sharpe": float("nan"), "maxdd": 0.0}
    net = np.array([t.net for t in trades])
    pnl = daily_pnl(trades, df, asset)
    sd = pnl.std()
    return {
        "n": len(net), "hit": float((net > 0).mean()), "avg": float(net.mean()),
        "t": float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))) if len(net) > 1 else float("nan"),
        "total": float(net.sum()),
        "sharpe": float(pnl.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
        "maxdd": max_drawdown(pnl),
    }


def without_largest(trades: list[Trade], k: int = TOP_K) -> list[Trade]:
    return sorted(trades, key=lambda t: t.net)[:-k] if len(trades) > k else []


def same_windows_long(trades: list[Trade]) -> float:
    """Average gross long return over the rule's own holding windows (no cost, no direction)."""
    return float(np.mean([t.direction * t.gross for t in trades])) if trades else float("nan")


def unconditional(df: pd.DataFrame, asset: str, hold: int) -> dict:
    """All overlapping hold-day log returns in sample, and full-sample buy-and-hold stats."""
    df = df.loc[SAMPLE_START:SAMPLE_END]
    p = np.log(df[asset].to_numpy())
    windows = p[hold:] - p[:-hold]
    daily = pd.Series(np.diff(p), index=df.index[1:])
    years = (df.index[-1] - df.index[0]).days / 365.25
    return {"mean": float(windows.mean()), "sd": float(windows.std()), "share_pos": float((windows > 0).mean()),
            "bh_annual": float((p[-1] - p[0]) / years), "bh_sharpe": float(daily.mean() / daily.std() * np.sqrt(252)),
            "bh_maxdd": max_drawdown(daily)}


def yearly(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["n", "avg", "hit"])
    t = pd.DataFrame({"year": [x.entry_date.year for x in trades], "net": [x.net for x in trades]})
    return t.groupby("year").agg(n=("net", "size"), avg=("net", "mean"), hit=("net", lambda s: (s > 0).mean()))


def tension_yearly_loyo(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Rule (d) year by year with the threshold estimated on the other years."""
    weeks = weekly_gpr(df.loc[:SAMPLE_END])
    rows = []
    for year in sorted(weeks.last_day.dt.year.unique()):
        sd = float(weeks.loc[weeks.last_day.dt.year != year, "change"].std())
        trades, _ = tension_trades(df, asset, sd=sd, weeks=weeks[weeks.last_day.dt.year == year])
        if trades:
            net = np.array([t.net for t in trades])
            rows.append({"year": year, "n": len(net), "avg": net.mean(), "hit": (net > 0).mean()})
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame(columns=["n", "avg", "hit"])


# --------------------------------------------------------------------------
# Run + report
# --------------------------------------------------------------------------


def _pct(x, nd=2):
    return "n/a" if pd.isna(x) else f"{100 * x:+.{nd}f}%"


def _f(x, nd=2):
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def _metrics_row(label: str, m: dict) -> str:
    return (f"| {label} | {m['n']} | {_f(m['hit'])} | {_pct(m['avg'])} | {_f(m['t'], 1)} | {_pct(m['total'], 1)} | "
            f"{_f(m['sharpe'])} | {_pct(m['maxdd'], 1)} |")


METRICS_HEAD = ["| rule | trades | hit rate | avg/trade after cost | t | total | Sharpe (ann.) | max drawdown |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]


def run(df: pd.DataFrame, lists: dict[str, pd.DataFrame], assets=("gold", "silver")) -> dict:
    out = {"tables": {}, "retention": {}, "trades": {}, "tension": {}, "uncond": {}}
    for asset in assets:
        out["uncond"][asset] = unconditional(df, asset, HOLD_DAYS)
        out["uncond20"] = out.get("uncond20", {})
        out["uncond20"][asset] = unconditional(df, asset, 20)
        for lname, ev in lists.items():
            tab = event_table(df, ev, asset)
            out["tables"][(asset, lname)] = tab
            for d in ("up", "down"):
                out["retention"][(asset, lname, d)] = retention(tab, d)
            for rname, (code, _) in RULES.items():
                out["trades"][(asset, lname, rname)] = event_rule_trades(tab, code)
        trades, sd = tension_trades(df, asset)
        out["tension"][asset] = {"trades": trades, "sd": sd, "loyo": tension_yearly_loyo(df, asset)}
    return out


def report(res: dict, df: pd.DataFrame) -> str:
    L = ["# Strategy test v1 (in sample, 2000-08-30 to 2021-12-31)", "",
         "Everything below is in sample and nothing after 2021-12-31 is read. Each test is run on the hand list",
         "(`data/events.csv`) and on the mechanical list (`data/events_mechanical.csv`); the mechanical results are",
         "the ones that matter, because the hand list was selected with hindsight.", "",
         "Conventions: day 0 is the event date (or the next trading day). The net 5-day move is the event study's",
         "net return from the close of day -1 to the close of day 4, after removing what the dollar and the real",
         "yield explain (betas from the 250 days before day 0); the net 20-day move ends at the close of day 19.",
         "Trades enter at the close that ends the 5-day window and exit at the close that ends the 20-day window:",
         "15 trading days of the metal's own return, 0.1% round-trip cost, no overlapping positions (an event",
         "whose entry falls on or before the open position's exit is skipped). Sharpe ratios are annualised from",
         "the strategy's daily return series over the whole sample with flat days at zero, so they are comparable",
         "with buy-and-hold. Max drawdown is the largest fall of cumulative P&L, as a simple return. *t* is the average",
         "net trade return over its",
         "standard error. 'Ex top 5' removes the five most profitable trades.", ""]

    # ---- H3
    L += ["## H3: retention of the 5-day move at 20 days", "",
          "For events whose net 5-day move exceeded +1% (or fell below -1%): ratio = net 20-day / net 5-day.",
          "'Retains half' is the share of events with ratio >= 0.5; 'same sign' the share with ratio > 0. The mean",
          "ratio is swung by events whose 5-day move was barely over the threshold, so read the median. The last",
          "column is the metal's own (raw, gross) return from the close of day 4 to the close of day 19 on those",
          "events: what a trade entered at day 5 actually earns, as opposed to the net move the ratio measures.", "",
          "| asset | list | direction | n | mean ratio | min | 25% | median | 75% | max | retains half | same sign | mean net 5d | mean net 20d | raw day 4 to 19 |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (asset, lname, d), r in res["retention"].items():
        if r["n"] == 0:
            L.append(f"| {asset} | {lname} | {d} | 0 | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        L.append(f"| {asset} | {lname} | {d} | {r['n']} | {_f(r['mean'])} | {_f(r['min'], 1)} | {_f(r['q25'])} | "
                 f"{_f(r['median'])} | {_f(r['q75'])} | {_f(r['max'], 1)} | {r['share_half']:.0%} | {r['share_sign']:.0%} | "
                 f"{_pct(r['mean_net5'])} | {_pct(r['mean_net20'])} | {_pct(r['mean_hold'])} |")
    L.append("")

    # ---- rules
    for asset in ("gold", "silver"):
        u = res["uncond"][asset]
        L += [f"## Rules on {asset}", "",
              f"Benchmarks: no position = 0. Buy-and-hold {asset} over the full sample: {_pct(u['bh_annual'], 1)} a "
              f"year (log), Sharpe {_f(u['bh_sharpe'])}, max drawdown {_pct(u['bh_maxdd'], 1)}. The unconditional "
              f"{HOLD_DAYS}-day log return (every window in the sample) averages {_pct(u['mean'])} with SD "
              f"{_pct(u['sd'])}, positive {u['share_pos']:.0%} of the time: that is what 'buy and hold over the same "
              f"holding periods' earns on average with no signal, before cost.", ""]
        L += ["### All trades", ""] + METRICS_HEAD
        for lname in LISTS:
            for rname in RULES:
                tr = res["trades"][(asset, lname, rname)]
                L.append(_metrics_row(f"{rname} [{lname}]", metrics(tr, df, asset)))
        td = res["tension"][asset]
        L.append(_metrics_row(f"d: tension rule (weekly GPR change > 1 SD = {td['sd']:.1f}) [independent]",
                              metrics(td["trades"], df, asset)))
        L += ["", "### Same trades, five most profitable removed", ""] + METRICS_HEAD
        for lname in LISTS:
            for rname in RULES:
                tr = res["trades"][(asset, lname, rname)]
                L.append(_metrics_row(f"{rname} [{lname}]", metrics(without_largest(tr), df, asset)))
        L.append(_metrics_row("d: tension rule [independent]", metrics(without_largest(td["trades"]), df, asset)))
        L += ["", "### Buy-and-hold over the same holding windows", "",
              "Average gross long return of the metal over each rule's own windows, no cost (for buy rules this is",
              "the rule's return before cost; for the sell rule it is what being long instead would have made).", "",
              "| rule | list | trades | long over same windows, avg | rule avg after cost | unconditional avg |",
              "|---|---|---:|---:|---:|---:|"]
        for lname in LISTS:
            for rname in RULES:
                tr = res["trades"][(asset, lname, rname)]
                m = metrics(tr, df, asset)
                L.append(f"| {rname} | {lname} | {m['n']} | {_pct(same_windows_long(tr))} | {_pct(m['avg'])} | {_pct(u['mean'])} |")
        m = metrics(td["trades"], df, asset)
        u20 = res["uncond20"][asset]
        L.append(f"| d: tension rule (20-day hold) | independent | {m['n']} | {_pct(same_windows_long(td['trades']))} | "
                 f"{_pct(m['avg'])} | {_pct(u20['mean'])} (20-day) |")

        L += ["", "### Stability by year", "",
              "Fixed rules have nothing to fit, so blocked cross-validation by year reduces to asking whether each",
              "year's trades, taken alone, make money. For the tension rule the 1-SD threshold is re-estimated on",
              "the other years (leave-one-year-out). 'Years +' is the number of years with a positive average trade",
              "out of the years that had trades.", "",
              "| rule | list | years with trades | years + | mean of yearly avgs | worst year | best year |",
              "|---|---|---:|---:|---:|---|---|"]

        def year_row(label, lname, y):
            if y.empty:
                return f"| {label} | {lname} | 0 | - | - | - | - |"
            return (f"| {label} | {lname} | {len(y)} | {int((y.avg > 0).sum())} | {_pct(y.avg.mean())} | "
                    f"{y.avg.idxmin()} ({_pct(y.avg.min(), 1)}, n={int(y.loc[y.avg.idxmin(), 'n'])}) | "
                    f"{y.avg.idxmax()} ({_pct(y.avg.max(), 1)}, n={int(y.loc[y.avg.idxmax(), 'n'])}) |")

        for lname in LISTS:
            for rname in RULES:
                L.append(year_row(rname, lname, yearly(res["trades"][(asset, lname, rname)])))
        L.append(year_row("d: tension rule (LOYO threshold)", "independent", td["loyo"]))
        L.append("")

    L += ["## Verdicts", ""] + verdicts(res, df) + [""]
    return "\n".join(L)


def verdicts(res: dict, df: pd.DataFrame) -> list[str]:
    asset = "gold"
    u = res["uncond"][asset]
    out = []

    def pack(trades):
        m, mx = metrics(trades, df, asset), metrics(without_largest(trades), df, asset)
        y = yearly(trades)
        return m, mx, y

    def judge(m, mx, y, label_mech=None):
        """Three gates: beats cost (avg > 0 and > unconditional), survives ex-top-5, survives mechanical list."""
        costs = (m["avg"] > 0) and (m["t"] > 1) if m["n"] else False
        beats_uncond = m["avg"] > u["mean"] if m["n"] else False
        ex = (mx["avg"] > 0) if mx["n"] else False
        years = f"{int((y.avg > 0).sum())} of {len(y)} years positive" if len(y) else "no years with trades"
        return costs, beats_uncond, ex, years

    for rname, (code, desc) in RULES.items():
        h = pack(res["trades"][(asset, "hand", rname)])
        mch = pack(res["trades"][(asset, "mechanical", rname)])
        hc, hb, hx, hy = judge(*h)
        mc, mb, mx_, my = judge(*mch)
        hm, hmx, _ = h
        mm, mmx, _ = mch
        text = (f"**Rule {rname}.** {desc} Hand list: {hm['n']} trades, {_pct(hm['avg'])} per trade after cost "
                f"(t = {_f(hm['t'], 1)}), Sharpe {_f(hm['sharpe'])}, {_pct(hmx['avg'])} ex top 5, {hy}. "
                f"Mechanical list: {mm['n']} trades, {_pct(mm['avg'])} per trade (t = {_f(mm['t'], 1)}), Sharpe "
                f"{_f(mm['sharpe'])}, {_pct(mmx['avg'])} ex top 5, {my}. ")
        if mm["n"] == 0:
            text += "No trades on the mechanical list, so there is nothing to survive. "
        elif not mc and mm["avg"] > 0:
            text += (f"On the mechanical list its average after cost is positive but indistinguishable from zero "
                     f"(t = {_f(mm['t'], 1)}). ")
        elif mc and mb and mx_:
            text += (f"On the mechanical list it covers its costs, beats the unconditional {_pct(u['mean'])} per "
                     f"{HOLD_DAYS}-day window, and survives removing the five best trades. ")
        elif mc and mx_:
            text += (f"On the mechanical list it covers its costs and survives removing the five best trades, but it "
                     f"does not beat simply being long over random {HOLD_DAYS}-day windows ({_pct(u['mean'])}): it is "
                     f"gold's drift, not the signal. ")
        elif mc:
            text += "On the mechanical list it covers its costs only until the five best trades are removed. "
        else:
            text += "On the mechanical list it loses money after cost. "
        if hc and not mc:
            text += "That it looks better on the hand list is what hindsight selection does. "
        text += "Verdict: " + ("survives all three checks." if (mc and mb and mx_) else
                               "does not survive." if not mc else
                               "survives costs but not the drift benchmark." if (mc and mx_) else
                               "survives costs but not largest-trade removal.")
        out.append(text)
        out.append("")

    td = res["tension"][asset]
    m, mx, _ = pack(td["trades"])
    y = td["loyo"]
    years = f"{int((y.avg > 0).sum())} of {len(y)} years positive with the threshold fitted on the other years" if len(y) else "no years"
    ok_cost = m["n"] > 0 and m["avg"] > 0 and m["t"] > 1
    ok_ex = mx["n"] > 0 and mx["avg"] > 0
    u20 = res["uncond20"][asset]
    ok_b = m["n"] > 0 and m["avg"] > u20["mean"]
    text = (f"**Rule d: tension.** Buy at the close of any week whose mean GPR rose by more than one SD of weekly "
            f"changes ({td['sd']:.1f} index points), hold 20 trading days. Independent of both lists. {m['n']} trades, "
            f"{_pct(m['avg'])} per trade after cost (t = {_f(m['t'], 1)}), hit rate {_f(m['hit'])}, Sharpe "
            f"{_f(m['sharpe'])}, max drawdown {_pct(m['maxdd'], 1)}; {_pct(mx['avg'])} ex top 5; {years}. The "
            f"unconditional 20-day return is {_pct(u20['mean'])}. ")
    text += "Verdict: " + ("survives costs, largest-trade removal and the drift benchmark." if ok_cost and ok_ex and ok_b
                           else "covers costs but not the unconditional drift." if ok_cost and ok_ex
                           else "covers costs only with its best trades." if ok_cost
                           else "positive on average but indistinguishable from zero, and negative without its five best trades." if m["n"] and m["avg"] > 0 and not ok_ex
                           else "positive on average but indistinguishable from zero." if m["n"] and m["avg"] > 0
                           else "loses money after cost.")
    out.append(text)
    out.append("")
    ret = res["retention"]
    gh, gm = ret[("gold", "hand", "up")], ret[("gold", "mechanical", "up")]
    out.append(f"**Overall.** Buy-and-hold gold made {_pct(u['bh_annual'], 1)} a year over this sample with a Sharpe of "
               f"{_f(u['bh_sharpe'])}; any rule that is long gold for a fraction of the time inherits a slice of that "
               f"drift, so the bar is the unconditional {HOLD_DAYS}-day return of {_pct(u['mean'])}, not zero. "
               f"The retention numbers (H3) look strong on their own: after a net 5-day move above +1%, the median gold "
               f"event keeps {gh['median']:.1f}x of it at 20 days on the hand list and {gm['median']:.1f}x on the "
               f"mechanical list. But that is the *net* move, after the dollar and real yields are removed, and a "
               f"position in the metal earns the raw move: from day 4 to day 19 those same events returned "
               f"{_pct(gm['mean_hold'])} gross per event on the mechanical list ({_pct(gh['mean_hold'])} on the hand "
               f"list, falling to {_pct(same_windows_long(res['trades'][('gold', 'hand', 'a: buy if net 5d > +1%')]))} "
               f"once overlapping events are skipped), against {_pct(u['mean'])} for any random window. The "
               f"event-specific component persists; the part you can trade does not beat drift.")
    return out


def main() -> None:
    df = data.load().loc[:SAMPLE_END]
    lists = {"hand": load_events(), "mechanical": load_mechanical()}
    res = run(df, lists)
    text = report(res, df)
    REPORT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
