"""Tests for the mechanical event list."""

import numpy as np
import pandas as pd

from gold_events.mechanical import MERGE_DAYS, flag_days, merge_runs


def make_frame():
    idx = pd.bdate_range("1990-01-01", "2021-12-31")
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"gpr": 100 + rng.normal(0, 10, len(idx)), "epu": 100 + rng.normal(0, 10, len(idx))}, index=idx)
    f.loc["2005-03-01":"2005-03-03", "gpr"] = 500          # 3 flagged days
    f.loc["2005-03-09", "gpr"] = 500                        # 4 trading days after 03-03 -> same event
    f.loc["2005-03-17", "epu"] = 600                        # 6 trading days after 03-09 -> new event, EPU-triggered
    f.loc["2010-06-15", "epu"] = 700
    f.loc["2010-06-15", "gpr"] = 400                        # both on one day
    return f


def test_flags_use_each_index_own_95th_percentile():
    f, thr = flag_days(make_frame())
    assert abs(f["gpr_flag"].mean() - 0.05) < 1e-3 and abs(f["epu_flag"].mean() - 0.05) < 1e-3
    assert thr["gpr"] > 100 and thr["epu"] > 100


def test_merge_runs_groups_within_five_trading_days():
    f, _ = flag_days(make_frame())
    f = f.loc["2005-01-01":"2010-12-31"]
    # keep only the planted spikes so random tail days don't interfere
    f.loc[:, "gpr_flag"] = f["gpr"] >= 400
    f.loc[:, "epu_flag"] = f["epu"] >= 600
    f["flag"] = f["gpr_flag"] | f["epu_flag"]
    mech = merge_runs(f)
    assert len(mech) == 3
    first, second, third = mech.iloc[0], mech.iloc[1], mech.iloc[2]
    assert first.date == pd.Timestamp("2005-03-01") and first.end_date == pd.Timestamp("2005-03-09")
    assert first.n_flagged_days == 4 and first.trigger == "gpr" and first.peak_gpr == 500
    assert second.date == pd.Timestamp("2005-03-17") and second.trigger == "epu" and second.peak_epu == 600
    assert third.date == pd.Timestamp("2010-06-15") and third.trigger == "both"
    # the gap rule is in trading days
    gap = f.index.get_loc(second.date) - f.index.get_loc(first.end_date)
    assert gap > MERGE_DAYS
