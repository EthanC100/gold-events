"""Tests for the strategy test: trade construction, metrics and retention."""

import numpy as np
import pandas as pd
import pytest

from gold_events import strategies as st
from gold_events.strategies import Trade, _non_overlapping, max_drawdown, metrics, retention, without_largest


def make_frame(n=800, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({
        "gold": 300 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
        "silver": 5 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n))),
        "dxy": 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))),
        "real_yield": 1 + np.cumsum(rng.normal(0, 0.03, n)),
        "gpr": 100 + rng.normal(0, 20, n),
    }, index=idx)


def test_event_table_windows_and_hold_return(monkeypatch):
    df = make_frame()
    monkeypatch.setattr(st, "SAMPLE_START", df.index[0])
    monkeypatch.setattr(st, "SAMPLE_END", df.index[-1])
    events = pd.DataFrame({"date": [df.index[300], df.index[-10]], "event": ["x", "late"]})
    tab = st.event_table(df, events, "gold")
    assert len(tab) == 1  # the late event's 20-day window runs past the end
    r = tab.iloc[0]
    assert r.entry_pos == 304 and r.exit_pos == 319
    assert r.hold_ret == pytest.approx(np.log(df.gold.iloc[319] / df.gold.iloc[304]))
    assert r.raw20 == pytest.approx(np.log(df.gold.iloc[319] / df.gold.iloc[299]))


def test_non_overlapping_skips_events_inside_open_position():
    idx = pd.bdate_range("2010-01-01", periods=100)
    cand = pd.DataFrame({
        "entry_pos": [10, 20, 25, 26, 60], "exit_pos": [25, 35, 40, 41, 75],
        "entry_date": idx[[10, 20, 25, 26, 60]], "exit_date": idx[[25, 35, 40, 41, 60 + 15]],
        "hold_ret": [0.02, 0.05, -0.01, 0.03, 0.01],
    })
    trades = _non_overlapping(cand, +1)
    # 10..25 taken; 20 skipped (inside); 25 skipped (entry == open exit); 26 taken; 60 taken
    assert [t.entry_pos for t in trades] == [10, 26, 60]
    assert trades[0].net == pytest.approx(0.02 - st.COST)
    short = _non_overlapping(cand.iloc[[0]], -1)
    assert short[0].gross == pytest.approx(-0.02) and short[0].direction == -1


def test_retention_ratio_and_shares():
    tab = pd.DataFrame({"net5": [0.02, 0.03, -0.02, 0.005, np.nan], "net20": [0.02, 0.006, -0.03, 0.05, 0.01],
                        "hold_ret": [0.0, 0.0, 0.0, 0.0, 0.0]})
    up = retention(tab, "up")
    assert up["n"] == 2 and up["median"] == pytest.approx((1.0 + 0.2) / 2)
    assert up["share_half"] == 0.5 and up["share_sign"] == 1.0
    down = retention(tab, "down")
    assert down["n"] == 1 and down["mean"] == pytest.approx(1.5) and down["share_half"] == 1.0


def test_metrics_and_drawdown(monkeypatch):
    df = make_frame()
    monkeypatch.setattr(st, "SAMPLE_START", df.index[0])
    monkeypatch.setattr(st, "SAMPLE_END", df.index[-1])
    p = np.log(df.gold.to_numpy())
    trades = [Trade(100, 115, df.index[100], df.index[115], +1, p[115] - p[100], p[115] - p[100] - st.COST),
              Trade(200, 215, df.index[200], df.index[215], -1, -(p[215] - p[200]), -(p[215] - p[200]) - st.COST)]
    m = metrics(trades, df, "gold")
    assert m["n"] == 2
    assert m["avg"] == pytest.approx(np.mean([t.net for t in trades]))
    assert m["total"] == pytest.approx(sum(t.net for t in trades))
    # the daily P&L reproduces the trade returns exactly
    pnl = st.daily_pnl(trades, df, "gold")
    assert pnl.sum() == pytest.approx(m["total"])
    assert (pnl.loc[df.index[116]:df.index[199]] == 0).all()
    assert max_drawdown(pd.Series([0.1, -0.2, 0.05])) == pytest.approx(np.expm1(-0.2))
    assert metrics([], df, "gold")["n"] == 0


def test_without_largest_drops_most_profitable():
    trades = [Trade(i, i + 1, None, None, 1, r, r) for i, r in enumerate([0.05, -0.01, 0.02, 0.0, 0.03, -0.02, 0.01])]
    kept = without_largest(trades, k=5)
    assert sorted(t.net for t in kept) == [-0.02, -0.01]
    assert without_largest(trades[:3], k=5) == []


def test_tension_rule_threshold_and_hold(monkeypatch):
    df = make_frame()
    monkeypatch.setattr(st, "SAMPLE_START", df.index[0])
    monkeypatch.setattr(st, "SAMPLE_END", df.index[-1])
    df.loc[df.index[250]:df.index[254], "gpr"] = 1000  # one huge week
    trades, sd = st.tension_trades(df, "gold")
    assert sd > 0 and len(trades) >= 1
    big = [t for t in trades if df.index[250] <= t.entry_date <= df.index[256]]
    assert len(big) == 1 and big[0].exit_pos - big[0].entry_pos == 20
    assert all(b.entry_pos > a.exit_pos for a, b in zip(trades, trades[1:]))
