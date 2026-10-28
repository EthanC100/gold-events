"""Tests for the v2 factors and the rolling-window rule."""

import numpy as np
import pandas as pd
import pytest

from gold_events import rolling, v2
from gold_events.model import CONTROLS, build_design


def make_frame(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-01-01", periods=n)
    return pd.DataFrame({
        "gold": 400 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))),
        "silver": 7 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n))),
        "dxy": 90 * np.exp(np.cumsum(rng.normal(0, 0.005, n))),
        "real_yield": 1 + np.cumsum(rng.normal(0, 0.03, n)),
        "vix": 18 * np.exp(np.cumsum(rng.normal(0, 0.05, n))),
        "gpr": 100 + rng.normal(0, 20, n),
        "gld_tonnes": 500 + np.cumsum(rng.normal(0, 2, n)),
        "cb_net_purchases": 50 + 10 * np.sin(np.arange(n) / 60),
    }, index=idx)


def test_add_factors_uses_window_endpoints_and_lagged_pace():
    df = make_frame()
    X = build_design(df, rolling.EMPTY_EVENTS, "gold", 5, "severity_hand", start=df.index[100], end=df.index[-1])
    X2 = v2.add_factors(X, df, 5)
    t = X2.index[10]
    pos = df.index.get_loc(t)
    assert X2.loc[t, "d_gld"] == pytest.approx(df.gld_tonnes.iloc[pos + 4] - df.gld_tonnes.iloc[pos - 1])
    assert X2.loc[t, "cb_flow"] == pytest.approx(df.cb_net_purchases.iloc[pos + 4] * 5 / 63)
    assert list(X2.columns) == ["y"] + CONTROLS + ["d_gld", "cb_flow"]
    # a NaN in gld drops only the windows that touch it
    df2 = df.copy()
    df2.loc[df2.index[500], "gld_tonnes"] = np.nan
    X3 = v2.add_factors(X, df2, 5)
    assert len(X) - len(X3) == 2  # base = 500 (start 501) and last = 500 (start 496)


def test_frozen_v2_predicts_with_all_six_controls():
    df = make_frame()
    X = v2.design_v2(df, rolling.EMPTY_EVENTS, "gold", 5, df.index[100], df.index[-1])
    y = X.pop("y")
    f = v2.fit_frozen(X, y, v2.V2_CONTROLS, "gold", 5)
    assert set(f.coef) == set(v2.V2_CONTROLS) and f.n == len(y)
    pred = f.predict(X)
    manual = f.intercept + sum(f.coef[c] * X[c].to_numpy() for c in v2.V2_CONTROLS)
    assert pred == pytest.approx(manual)
    table = v2.coefficient_table({("gold", 5): f})
    assert "d_gld_pct_per_unit" in table.columns and table.loc[0, "d_gld_pct_per_unit"] == pytest.approx(1000 * f.coef["d_gld"])


def test_window_ends_quarterly_plus_latest():
    idx = pd.bdate_range("2000-08-30", "2012-02-14")
    ends = rolling.window_ends(idx, start=pd.Timestamp("2000-08-30"))
    assert ends[0] >= pd.Timestamp("2005-08-30") and ends[-1] == idx[-1]
    assert all(e in idx for e in ends)
    gaps = np.diff([e.to_period("Q").ordinal for e in ends[:-1]])
    assert (gaps == 1).all()  # one window per quarter


def test_rule_fires_only_when_a_coefficient_leaves_the_iqr():
    ends = pd.date_range("2010-03-31", periods=40, freq="QE")
    base = {"intercept": 0.1, "d_real_yield": -3.0, "dlog_dxy": -1.0, "dlog_vix": 0.0, "d_gpr": 0.2}
    rng = np.random.default_rng(0)
    rows = {}
    for asset in ("gold", "silver"):
        r = pd.DataFrame({k: v + rng.normal(0, 0.01, len(ends)) for k, v in base.items()}, index=ends)
        for c in CONTROLS:
            r[f"{c}_se"] = 0.05
        r["window_start"] = ends - pd.DateOffset(years=5)
        rows[(asset, 5)] = r
    v1 = {(a, 5): dict(base) for a in ("gold", "silver")}
    checks, keep = rolling.apply_rule(rows, v1)
    assert keep and len(checks) == 10
    v1[("gold", 5)]["d_real_yield"] = -6.0
    checks, keep = rolling.apply_rule(rows, v1)
    assert not keep and sum(not c.inside for c in checks) == 1
    latest = rolling.latest_window_table(rows)
    assert set(latest.asset) == {"gold", "silver"} and latest.loc[0, "window_end"] == ends[-1].strftime("%Y-%m-%d")
