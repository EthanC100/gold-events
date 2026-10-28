"""Tests for the event-study maths on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from gold_events.event_study import (
    Betas,
    control_changes,
    estimate_betas,
    event_study,
    first_trading_day_after,
    first_trading_day_on_or_after,
    window_change,
)

BETA_DXY = -1.5   # gold log return per unit of dlog(DXY)
BETA_RY = -0.02   # gold log return per percentage point of d(real_yield)


def make_frame(n: int = 400, seed: int = 0, shock_day: int | None = None, shock: float = 0.0) -> pd.DataFrame:
    """Synthetic daily frame where gold = exactly beta * controls + known idiosyncratic shocks.

    No noise term, so OLS must recover the betas exactly and the net return
    on a shocked day must equal the shock.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    d_dxy = rng.normal(0, 0.005, n)
    d_ry = rng.normal(0, 0.03, n)
    idio = np.zeros(n)
    if shock_day is not None:
        idio[shock_day] = shock
    gold_ret = BETA_DXY * d_dxy + BETA_RY * d_ry + idio
    silver_ret = 2 * gold_ret  # silver as a leveraged copy, with twice the betas and shock
    df = pd.DataFrame({
        "gold": 1000 * np.exp(np.cumsum(gold_ret)),
        "silver": 20 * np.exp(np.cumsum(silver_ret)),
        "dxy": 100 * np.exp(np.cumsum(d_dxy)),
        "real_yield": 1.0 + np.cumsum(d_ry),
    }, index=idx)
    return df


def test_control_changes_are_log_dxy_and_level_real_yield():
    df = make_frame(10)
    ctrl = control_changes(df)
    assert np.isclose(ctrl["dxy"].iloc[3], np.log(df.dxy.iloc[3] / df.dxy.iloc[2]))
    assert np.isclose(ctrl["real_yield"].iloc[3], df.real_yield.iloc[3] - df.real_yield.iloc[2])
    assert ctrl.iloc[0].isna().all()


def test_estimate_betas_recovers_known_coefficients():
    df = make_frame(300)
    ctrl = control_changes(df)
    rets = np.log(df.gold).diff()
    b = estimate_betas(rets, ctrl)
    assert b.n_obs == 299  # first diff is NaN
    assert b.dxy == pytest.approx(BETA_DXY, abs=1e-9)
    assert b.real_yield == pytest.approx(BETA_RY, abs=1e-9)
    assert b.alpha == pytest.approx(0.0, abs=1e-9)


def test_estimate_betas_returns_nan_when_too_few_observations():
    df = make_frame(50)
    b = estimate_betas(np.log(df.gold).diff(), control_changes(df), min_obs=200)
    assert isinstance(b, Betas)
    assert np.isnan(b.dxy) and np.isnan(b.real_yield)
    assert b.n_obs == 49


def test_window_change_log_and_level_and_bounds():
    s = pd.Series([100.0, 110.0, np.nan, 121.0])
    assert window_change(s, 0, 1, log=True) == pytest.approx(np.log(1.1))
    assert window_change(s, 0, 1, log=False) == pytest.approx(10.0)
    assert np.isnan(window_change(s, 0, 2, log=True))   # NaN endpoint
    assert np.isnan(window_change(s, 0, 10, log=True))  # beyond the end


def test_raw_return_spans_h_days_from_close_before_day0():
    df = make_frame(400)
    day0 = df.index[300]
    res = event_study(df, day0, horizons=(1, 5, 20))
    for h in (1, 5, 20):
        expected = np.log(df.gold.iloc[300 + h - 1] / df.gold.iloc[299])
        assert res.get("gold", h, "raw") == pytest.approx(expected)
        expected_s = np.log(df.silver.iloc[300 + h - 1] / df.silver.iloc[299])
        assert res.get("silver", h, "raw") == pytest.approx(expected_s)


def test_net_return_isolates_the_idiosyncratic_shock():
    # A 3% gold-specific shock on day 0 (position 300); everything else is beta * controls.
    df = make_frame(400, shock_day=300, shock=0.03)
    res = event_study(df, df.index[300], horizons=(1, 5, 20), window=250)
    for h in (1, 5, 20):
        assert res.get("gold", h, "net") == pytest.approx(0.03, abs=1e-9)
        assert res.get("silver", h, "net") == pytest.approx(0.06, abs=1e-9)
        # net = raw - explained, by construction
        assert res.get("gold", h, "net") == pytest.approx(res.get("gold", h, "raw") - res.get("gold", h, "explained"))
    # a shock *after* the 5-day window must not show up at 1 or 5 days but must at 20
    df2 = make_frame(400, shock_day=310, shock=0.03)
    res2 = event_study(df2, df2.index[300])
    assert res2.get("gold", 1, "net") == pytest.approx(0.0, abs=1e-9)
    assert res2.get("gold", 5, "net") == pytest.approx(0.0, abs=1e-9)
    assert res2.get("gold", 20, "net") == pytest.approx(0.03, abs=1e-9)


def test_betas_use_only_the_window_before_day0():
    # Put a huge idiosyncratic shock on day 0 itself: it must not contaminate the betas.
    df = make_frame(400, shock_day=300, shock=0.5)
    res = event_study(df, df.index[300])
    assert res.get("gold", 1, "beta_dxy") == pytest.approx(BETA_DXY, abs=1e-9)
    assert res.get("gold", 1, "beta_real_yield") == pytest.approx(BETA_RY, abs=1e-9)
    assert res.get("silver", 1, "beta_dxy") == pytest.approx(2 * BETA_DXY, abs=1e-9)
    assert res.table.n_obs.iloc[0] == 250
    # a shock just *inside* the window (day -1) does contaminate them: window really ends at day -1
    df2 = make_frame(400, shock_day=299, shock=0.5)
    res2 = event_study(df2, df2.index[300])
    assert res2.get("gold", 1, "beta_dxy") != pytest.approx(BETA_DXY, abs=1e-6)


def test_short_history_gives_nan_betas_but_raw_returns():
    df = make_frame(400)
    res = event_study(df, df.index[100], window=250, min_obs=200)  # only 99 usable days before
    assert np.isnan(res.get("gold", 5, "net"))
    assert not np.isnan(res.get("gold", 5, "raw"))
    assert res.table.n_obs.iloc[0] == 99


def test_horizon_beyond_data_end_is_nan():
    df = make_frame(400)
    res = event_study(df, df.index[395])
    assert not np.isnan(res.get("gold", 1, "raw"))
    assert not np.isnan(res.get("gold", 5, "raw"))   # positions 395..399 exist
    assert np.isnan(res.get("gold", 20, "raw"))


def test_day0_must_be_a_trading_day():
    df = make_frame(50)
    with pytest.raises(ValueError):
        event_study(df, "2015-01-03")  # a Saturday
    with pytest.raises(ValueError):
        event_study(df, df.index[0])  # no day -1


def test_trading_day_helpers():
    idx = pd.DatetimeIndex(["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10"])
    assert first_trading_day_after(idx, "2022-11-08") == pd.Timestamp("2022-11-09")
    assert first_trading_day_after(idx, "2022-11-05") == pd.Timestamp("2022-11-07")  # weekend election
    assert first_trading_day_on_or_after(idx, "2022-11-08") == pd.Timestamp("2022-11-08")
    assert first_trading_day_on_or_after(idx, "2022-11-06") == pd.Timestamp("2022-11-07")
    with pytest.raises(ValueError):
        first_trading_day_after(idx, "2022-11-10")


def test_nan_rows_in_window_are_dropped_not_propagated():
    df = make_frame(400)
    df.loc[df.index[150:160], "real_yield"] = np.nan  # a 10-day hole in the controls
    res = event_study(df, df.index[300])
    assert res.table.n_obs.iloc[0] == 250 - 11  # 10 NaN levels -> 11 NaN diffs
    assert res.get("gold", 1, "beta_dxy") == pytest.approx(BETA_DXY, abs=1e-9)
