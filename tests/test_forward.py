"""Tests for the weekly forward log and its scoring."""

import numpy as np
import pandas as pd
import pytest

from gold_events import forward as fw


def make_frame(n=60):
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "gold": 2000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "silver": 25 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
        "dxy": 100 + np.arange(n) * 0.1, "real_yield": 1.0 + np.arange(n) * 0.01,
        "vix": 15 + np.arange(n) * 0.2, "gpr": 100 + np.arange(n) * 2.0,
    }, index=idx)


FROZEN = {"controls": fw.CONTROL_COLS, "horizon_days": 5, "assets": {
    "gold": {"intercept": 0.001, "coef": {"d_real_yield": -0.03, "dlog_dxy": -1.0, "dlog_vix": 0.0, "d_gpr": 0.0}},
    "silver": {"intercept": 0.0, "coef": {"d_real_yield": 0.0, "dlog_dxy": -1.6, "dlog_vix": -0.05, "d_gpr": 0.00003}},
}}
QUESTIONS = [{"id": "q1", "source": "polymarket", "url": "", "resolved": True, "outcome": 1},
             {"id": "q2", "source": "polymarket", "url": "", "resolved": False, "outcome": None}]


def test_last_week_changes_and_prediction():
    df = make_frame()
    ch = fw.last_week_changes(df)
    assert ch["as_of"] == df.index[-1]
    assert ch["d_real_yield"] == pytest.approx(0.05)
    assert ch["dlog_dxy"] == pytest.approx(np.log(df.dxy.iloc[-1] / df.dxy.iloc[-6]))
    assert ch["d_gpr"] == pytest.approx(10.0)
    pred = fw.predict(FROZEN, ch)
    assert pred["gold"] == pytest.approx(0.001 - 0.03 * 0.05 - 1.0 * ch["dlog_dxy"])
    assert pred["silver"] == pytest.approx(-1.6 * ch["dlog_dxy"] - 0.05 * ch["dlog_vix"] + 0.00003 * 10)


def test_load_frozen_converts_percent_units(tmp_path):
    csv = tmp_path / "coef.csv"
    csv.write_text("asset,h,intercept_pct,lambda,d_real_yield_pct_per_unit,dlog_dxy_pct_per_unit,"
                   "dlog_vix_pct_per_unit,d_gpr_pct_per_unit\n"
                   "gold,5,0.15,1.0,-3.0,-1.0,0.0,0.2\nsilver,5,0.1,1.0,-3.5,-1.6,-0.05,0.3\n"
                   "gold,20,0.6,1.0,-4.3,-0.85,0.01,0.3\n")
    f = fw.load_frozen(csv)
    g = f["assets"]["gold"]
    assert g["intercept"] == pytest.approx(0.0015)
    assert g["coef"]["d_real_yield"] == pytest.approx(-0.03)   # -3% per pp -> -0.03 log per pp
    assert g["coef"]["dlog_dxy"] == pytest.approx(-1.0)        # -1% per 1% -> -1.0 log per log
    assert g["coef"]["d_gpr"] == pytest.approx(0.2 / 100 / 100)  # 0.2% per 100 points
    assert set(f["assets"]) == {"gold", "silver"}
    with pytest.raises(ValueError):
        fw.load_frozen(csv, h=20)  # no silver row at 20


def test_gpr_snapshot_uses_seven_calendar_days():
    idx = pd.date_range("2024-03-01", periods=15)
    gpr = pd.Series(np.arange(15, dtype=float) * 10, index=idx)
    snap = fw.gpr_snapshot(gpr)
    assert snap["gpr_date"] == idx[-1] and snap["gpr_level"] == 140.0 and snap["gpr_change_week"] == 70.0


def test_build_append_and_score(tmp_path):
    df = make_frame()
    gpr = df["gpr"]
    path = tmp_path / "log.csv"
    row = fw.build_row(df.index[30], QUESTIONS, FROZEN, df.iloc[:31], gpr.iloc[:31], {"q1": 0.4, "q2": float("nan")})
    assert row["log_date"] == df.index[30].strftime("%Y-%m-%d")
    assert row["my_tension_score"] == "" and row["my_prob_q1"] == ""
    assert row["market_prob_q1"] == 0.4 and np.isnan(row["market_prob_q2"])
    assert "last week's control changes" in row["note"]
    fw.append_row(row, path)
    row2 = fw.build_row(df.index[-3], QUESTIONS, FROZEN, df, gpr, {"q1": 0.6, "q2": 0.5})
    logdf = fw.append_row(row2, path)
    assert len(logdf) == 2 and list(logdf.columns) == fw.columns(QUESTIONS)

    # a build failure still produces a row, without a prediction
    row3 = fw.build_row(df.index[-1], QUESTIONS, FROZEN, None, gpr, {"q1": 0.5, "q2": 0.5}, note="data build failed")
    assert np.isnan(row3["pred_gold_5d"]) and row3["note"] == "data build failed"

    # scoring: first row's week has passed, second has not
    logdf.loc[0, "my_prob_q1"] = 0.9
    logdf.loc[1, "my_prob_q1"] = 0.7
    scored = fw.fill_realised(logdf, df)
    expected = np.log(df.gold.iloc[35] / df.gold.iloc[30])
    assert scored.loc[0, "realised_gold_5d"] == pytest.approx(expected)
    assert scored.loc[0, "realised_through"] == df.index[35].strftime("%Y-%m-%d")
    assert pd.isna(scored.loc[1, "realised_gold_5d"])
    stats = fw.prediction_stats(scored)
    assert stats["gold"]["n"] == 1
    assert stats["gold"]["mean_error"] == pytest.approx(expected - float(scored.loc[0, "pred_gold_5d"]))
    b = fw.brier(scored, QUESTIONS)  # only q1 is resolved (outcome 1): my probs 0.9, 0.7; market 0.4, 0.6
    assert b["my_n"] == 2 and b["my_brier"] == pytest.approx(np.mean([(0.9 - 1) ** 2, (0.7 - 1) ** 2]))
    assert b["market_n"] == 2 and b["market_brier"] == pytest.approx(np.mean([(0.4 - 1) ** 2, (0.6 - 1) ** 2]))
    text = fw.scorecard_markdown(scored, QUESTIONS)
    assert "| gold | 1 |" in text and "| me | 2 |" in text


def test_realised_moves_boundaries():
    df = make_frame(20)
    assert fw.realised_moves(df, df.index[-3]) is None       # fewer than 5 days left
    assert fw.realised_moves(df, "2023-12-01") is None       # before the data
    friday = df.index[9]
    assert friday.weekday() == 4
    got = fw.realised_moves(df, friday + pd.Timedelta(days=1))  # a Saturday: base is the Friday close
    assert got["realised_gold_5d"] == pytest.approx(np.log(df.gold.iloc[14] / df.gold.iloc[9]))


def test_polymarket_slug_parsing(monkeypatch):
    calls = []

    class R:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params["slug"]))
        if url.endswith("/markets"):
            return R([])
        return R([{"markets": [{"outcomePrices": '["0.23", "0.77"]'}]}])

    monkeypatch.setattr(fw.requests, "get", fake_get)
    p = fw.polymarket_probability("https://polymarket.com/event/some-event-slug?tid=1")
    assert p == pytest.approx(0.23)
    assert calls[0][1] == "some-event-slug" and calls[1][0].endswith("/events")


def test_model_version_pointer_selects_coefficients(tmp_path, monkeypatch):
    v1 = tmp_path / "v1.csv"
    v2 = tmp_path / "v2r.csv"
    head = ("asset,h,intercept_pct,lambda,d_real_yield_pct_per_unit,dlog_dxy_pct_per_unit,"
            "dlog_vix_pct_per_unit,d_gpr_pct_per_unit\n")
    v1.write_text(head + "gold,5,0.1,0,-3,-1,0,0.2\nsilver,5,0.1,0,-3,-1.6,0,0.3\n")
    v2.write_text(head + "gold,5,0.4,0,-4,-0.9,0,0.3\nsilver,5,0.4,0,-3,-1.7,0,0.5\n")
    monkeypatch.setattr(fw, "FROZEN_PATHS", {"v1": v1, "v2-rolling": v2})
    pointer = tmp_path / "forward_model.txt"
    monkeypatch.setattr(fw, "POINTER_PATH", pointer)
    assert fw.model_version() == "v1"                      # no pointer file: v1
    assert fw.load_frozen()["assets"]["gold"]["intercept"] == pytest.approx(0.001)
    pointer.write_text("v2-rolling\n")
    f = fw.load_frozen()
    assert f["version"] == "v2-rolling" and f["assets"]["gold"]["intercept"] == pytest.approx(0.004)
    pointer.write_text("nonsense\n")
    assert fw.model_version() == "v1"                      # unknown value falls back to v1


def test_fetch_probabilities_inverts_and_tolerates_failures(monkeypatch):
    monkeypatch.setitem(fw.FETCHERS, "polymarket", lambda url: 0.3 if "ok" in url else (_ for _ in ()).throw(RuntimeError("down")))
    qs = [{"id": "a", "source": "polymarket", "url": "https://polymarket.com/event/ok"},
          {"id": "b", "source": "polymarket", "url": "https://polymarket.com/event/ok", "invert": True},
          {"id": "c", "source": "polymarket", "url": "https://polymarket.com/event/bad"},
          {"id": "d", "source": "polymarket", "url": ""}]
    out = fw.fetch_probabilities(qs)
    assert out["a"] == pytest.approx(0.3) and out["b"] == pytest.approx(0.7)
    assert np.isnan(out["c"]) and np.isnan(out["d"])
