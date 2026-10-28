"""The out-of-sample test must never refit v1: its coefficients come from the <=2021 frame only."""
import numpy as np
import pandas as pd

from gold_events import oos
from gold_events.model import CONTROLS, SAMPLE_END


def _frame(n=1800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    dxy = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    ry = np.cumsum(rng.normal(0, 0.03, n))
    vix = 20 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    gpr = 100 + np.cumsum(rng.normal(0, 5, n))
    r = -0.8 * np.diff(np.log(dxy), prepend=np.log(dxy[0])) - 0.02 * np.diff(ry, prepend=ry[0]) + rng.normal(0, 0.01, n)
    gold = 1500 * np.exp(np.cumsum(r))
    silver = 20 * np.exp(np.cumsum(1.5 * r))
    return pd.DataFrame({"gold": gold, "silver": silver, "dxy": dxy, "real_yield": ry, "vix": vix, "gpr": gpr}, index=idx)


def _events():
    return pd.DataFrame({"date": pd.to_datetime(["2020-03-16"]), "category": ["Pandemic or disaster"], "severity_hand": [4]})


def test_frozen_coefficients_ignore_post_sample_data():
    df, ev = _frame(), _events()
    a = oos.frozen_v1(df, ev)
    df2 = df.copy()
    df2.loc[df2.index > SAMPLE_END, ["gold", "silver"]] *= np.exp(np.linspace(0, 1, (df2.index > SAMPLE_END).sum()))[:, None]
    b = oos.frozen_v1(df2, ev)
    for k in a:
        assert a[k].intercept == b[k].intercept
        for c in CONTROLS:
            assert a[k].coef[c] == b[k].coef[c]


def test_test_windows_start_in_2022_and_metrics_are_sane():
    df, ev = _frame(), _events()
    frozen = oos.frozen_v1(df, ev)
    windows = oos.run(df, ev, frozen)
    for w in windows:
        assert w.X.index[0] > SAMPLE_END
        assert w.full["n"] == w.trimmed["n"] + oos.N_DROP
        assert 0 <= w.full["hit_v1"] <= 1
        # the planted structure is linear in the controls, so v1 should beat predicting zero
        assert w.full["rmse_v1"] < w.full["rmse_zero"]


def test_event_moves_decompose_raw():
    df, ev = _frame(), _events()
    frozen = oos.frozen_v1(df, ev)
    saved = oos.TEST_EVENTS
    oos.TEST_EVENTS = [("2023-05-02", "Military escalation", "planted")]
    try:
        em = oos.event_moves(df, frozen)
    finally:
        oos.TEST_EVENTS = saved
    assert len(em) == 4
    assert np.allclose(em.raw - em.explained_v1, em.net_v1)
