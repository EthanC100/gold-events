"""Tests for the v1 model design and estimator."""

import numpy as np
import pandas as pd
import pytest

from gold_events.event_study import event_study
from gold_events.model import build_design, hac_cov, ols_fit, ridge_fit


def make_frame(n=600, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2001-01-01", periods=n)
    return pd.DataFrame({
        "gold": 300 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "silver": 5 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
        "dxy": 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))),
        "real_yield": 1 + np.cumsum(rng.normal(0, 0.03, n)),
        "vix": 20 * np.exp(np.cumsum(rng.normal(0, 0.05, n))),
        "gpr": 100 + rng.normal(0, 20, n),
    }, index=idx)


def make_events(df):
    return pd.DataFrame({
        "date": [df.index[100], df.index[300] - pd.Timedelta(days=1), df.index[598]],
        "category": ["War", "Bank", "War"],
        "severity_hand": [4, 2, 3],
        "severity_measured": [5, 5, 1],
    })


def test_design_matches_event_study_and_aligns_events():
    df = make_frame()
    events = make_events(df)
    start, end = df.index[50], df.index[-1]
    X = build_design(df, events, "gold", 5, "severity_hand", start=start, end=end)
    # y on an event's day 0 equals the event-study raw return
    day0 = df.index[100]
    res = event_study(df, day0, horizons=(5,), min_obs=1)
    assert X.loc[day0, "y"] == pytest.approx(res.get("gold", 5, "raw"))
    assert X.loc[day0, "dlog_dxy"] == pytest.approx(res.get("gold", 5, "d_dxy"))
    assert X.loc[day0, "d_real_yield"] == pytest.approx(res.get("gold", 5, "d_real_yield"))
    # event columns: severity at day 0 only, elsewhere 0; an event dated on a non-trading day
    # (the day before index[300], a Sunday) lands on index[300]
    assert X.loc[day0, "War"] == 4 and X.loc[df.index[300], "Bank"] == 2
    assert (X["War"] != 0).sum() == 1 and (X["Bank"] != 0).sum() == 1
    # the third event's 5-day window runs past the end of the data, so it is not a row
    assert df.index[598] not in X.index
    # window starts begin where the base day is >= start, and the last row ends at `end`
    assert X.index[0] == df.index[51]
    assert X.index[-1] == df.index[-5]


def test_sample_end_is_a_hard_cut():
    df = make_frame()
    events = make_events(df)
    end = df.index[400]
    X = build_design(df, events, "gold", 20, "severity_hand", start=df.index[50], end=end)
    assert X.index[-1] == df.index[400 - 19]  # last window ends exactly at `end`
    # identical result when the data after `end` is removed entirely
    X2 = build_design(df.loc[:end], events, "gold", 20, "severity_hand", start=df.index[50], end=end)
    pd.testing.assert_frame_equal(X, X2)


def test_ridge_with_zero_penalty_is_ols_and_large_penalty_shrinks_events():
    df = make_frame()
    events = make_events(df)
    X = build_design(df, events, "gold", 5, "severity_hand", start=df.index[50], end=df.index[-1])
    y = X.pop("y")
    ols = ols_fit(X, y, lag=5)
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    b, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    assert ols.coef == pytest.approx(b[1:], rel=1e-6, abs=1e-12)
    assert ols.intercept == pytest.approx(b[0], rel=1e-6, abs=1e-12)
    big = ridge_fit(X, y, penalised=["War", "Bank"], lag=5, grid=np.array([1e9]))
    assert abs(big.coef[big.names.index("War")]) < 1e-6
    assert abs(big.coef[big.names.index("Bank")]) < 1e-6
    # with the event terms shrunk to zero, the controls are exactly the controls-only OLS
    controls = ["d_real_yield", "dlog_dxy", "dlog_vix", "d_gpr"]
    only = ols_fit(X[controls], y, lag=5)
    for c in controls:
        assert big.coef[big.names.index(c)] == pytest.approx(only.coef[only.names.index(c)], rel=1e-4)


def test_hac_lag0_is_white_covariance():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
    e = rng.normal(size=200)
    V = hac_cov(X, e, np.zeros(3), lag=0)
    A = np.linalg.inv(X.T @ X)
    white = A @ (X * e[:, None]).T @ (X * e[:, None]) @ A
    assert V == pytest.approx(white)
