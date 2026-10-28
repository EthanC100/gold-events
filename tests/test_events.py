"""Tests for the measured-severity scoring."""

import numpy as np
import pandas as pd

from gold_events.events import jump_series, quintile_edges, quintile_score, score_events


def test_jump_series_windows_are_five_before_and_five_from_day0():
    idx = pd.date_range("2020-01-01", periods=20)
    s = pd.Series(np.arange(20, dtype=float), index=idx)  # 0, 1, 2, ...
    j = jump_series(s)
    d = idx[10]
    assert j.loc[d, "before"] == np.mean([5, 6, 7, 8, 9])
    assert j.loc[d, "after"] == np.mean([10, 11, 12, 13, 14])
    assert j.loc[d, "change"] == 5.0
    assert np.isnan(j.loc[idx[3], "before"])   # not enough history
    assert np.isnan(j.loc[idx[17], "after"])   # not enough future


def test_quintile_score_is_1_to_5():
    values = pd.Series(np.arange(1000, dtype=float))
    edges = quintile_edges(values)
    assert quintile_score(-5, edges) == 1
    assert quintile_score(500, edges) == 3
    assert quintile_score(999, edges) == 5
    assert np.isnan(quintile_score(float("nan"), edges))


def test_score_events_end_to_end():
    days = pd.date_range("1989-01-01", "2022-12-31")
    rng = np.random.default_rng(1)
    gpr = pd.Series(100 + rng.normal(0, 10, len(days)), index=days)
    epu = pd.Series(100 + rng.normal(0, 10, len(days)), index=days)
    # plant a big GPR jump from 2010-06-15 and a big EPU jump from 2015-03-02
    gpr.loc["2010-06-15":"2010-06-19"] += 500
    epu.loc["2015-03-02":"2015-03-06"] += 500
    months = pd.date_range("1989-01-01", "2022-12-01", freq="MS")
    umcsent = pd.Series(np.arange(len(months), dtype=float), index=months)
    calendar = pd.bdate_range("1990-01-01", "2022-12-31")
    events = pd.DataFrame({
        "date": pd.to_datetime(["2010-06-15", "2010-06-25", "2015-03-01"]),  # 2015-03-01 is a Sunday
        "category": ["a", "b", "c"], "event": ["x", "y", "z"], "countries": ["", "", ""],
        "severity_hand": [3, 2, 4], "source": ["", "", ""],
    })
    out = score_events(events, gpr, epu, umcsent, calendar)
    assert out.loc[0, "gpr_score"] == 5 and out.loc[0, "severity_measured"] == 5
    assert out.loc[2, "epu_score"] == 5  # Sunday event: uses the next available index day (Monday)
    assert out.loc[1, "gpr_score"] == 1  # the jump is in its "before" window, so change is very negative
    # sentiment: month after minus month before = 2 (series rises by 1 per month)
    assert (out["sentiment_change"] == 2).all()
    # first two events are 8 trading days apart; the third is years away
    assert out.loc[0, "nearest_event_days"] == 8 and out.loc[0, "near_other_event"]
    assert not out.loc[2, "near_other_event"]
    assert list(out.columns[:6]) == ["date", "category", "event", "countries", "severity_hand", "source"]
