"""Event list and measured severity.

``data/events.csv`` holds the hand-curated columns (``date, category, event,
countries, severity_hand, source``).  Running ``python -m gold_events.events``
adds measured columns in place:

``gpr_before / gpr_after / gpr_change``   5-day mean of the daily Geopolitical
    Risk index before the event (days -5..-1) and from day 0 (days 0..4), and
    the difference.  Both indices exist on every calendar day, so "day" here
    is a calendar day and day 0 is the event date itself.
``epu_*``                                 same for the daily US Economic Policy
    Uncertainty index.
``gpr_z / epu_z``                         the jump as a z-score against the same
    before/after jump computed on *every* day 1990-2021.
``gpr_score / epu_score``                 the jump on a 1-5 scale (quintiles of
    the same reference distribution), kept for reference.
``severity_measured``                     max(gpr_z, epu_z): continuous, can be
    negative when both indices fell into the event.
``severity_capped``                       severity_measured winsorised at z = 3.
    This is the severity the model uses.
``severity_quintile``                     max(gpr_score, epu_score), the earlier
    1-5 version, kept for reference.
``sentiment_before / sentiment_after / sentiment_change``
    Michigan consumer sentiment (UMCSENT) in the month before and the month
    after the event, and the difference.  Blank when the month after is past
    ``DATA_END``: nothing after 2021 is stored in the repository.
``near_other_event``                      True if another event's day 0 is within
    20 trading days; ``nearest_event_days`` is the distance (trading days).
"""

from __future__ import annotations

import pandas as pd

from gold_events import data
from gold_events.event_study import first_trading_day_on_or_after

EVENTS_PATH = data.DATA_DIR / "events.csv"
HAND_COLUMNS = ["date", "category", "event", "countries", "severity_hand", "source"]
REF_START, REF_END = pd.Timestamp("1990-01-01"), pd.Timestamp("2021-12-31")  # reference distribution
SEVERITY_CAP = 3.0  # winsorise the continuous severity at this z-score
DATA_END = pd.Period("2021-12", freq="M")  # no post-2021 data is written to the repository
PRE_DAYS = POST_DAYS = 5
NEAR_DAYS = 20


def load_events(path=EVENTS_PATH) -> pd.DataFrame:
    ev = pd.read_csv(path, parse_dates=["date"])
    ev = ev.sort_values("date").reset_index(drop=True)
    return ev


def jump_series(index: pd.Series, pre: int = PRE_DAYS, post: int = POST_DAYS) -> pd.DataFrame:
    """For every day d: mean over [d, d+post-1] minus mean over [d-pre, d-1].

    Returns a frame with columns ``before``, ``after``, ``change`` indexed by d.
    """
    s = index.sort_index()
    after = s.rolling(post).mean().shift(-(post - 1))  # mean of d..d+post-1
    before = s.rolling(pre).mean().shift(1)            # mean of d-pre..d-1
    return pd.DataFrame({"before": before, "after": after, "change": after - before})


def quintile_edges(values: pd.Series) -> list[float]:
    """The four cut points that split ``values`` into quintiles."""
    return [float(values.quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]


def quintile_score(x: float, edges: list[float]) -> float:
    """1-5 by position relative to the quintile edges (NaN stays NaN)."""
    if pd.isna(x):
        return float("nan")
    return 1 + sum(x > e for e in edges)


def jump_columns(dates: pd.Series, gpr: pd.Series, epu: pd.Series) -> pd.DataFrame:
    """The GPR/EPU jump columns and both severities for a series of dates."""
    out = pd.DataFrame(index=dates.index)
    for name, series in (("gpr", gpr), ("epu", epu)):
        jumps = jump_series(series)
        ref = jumps.loc[REF_START:REF_END, "change"].dropna()
        edges = quintile_edges(ref)
        rows = dates.apply(lambda d: _at(jumps, d))
        out[f"{name}_before"] = rows["before"].round(1).to_numpy()
        out[f"{name}_after"] = rows["after"].round(1).to_numpy()
        out[f"{name}_change"] = rows["change"].round(1).to_numpy()
        out[f"{name}_z"] = ((rows["change"] - ref.mean()) / ref.std()).round(2).to_numpy()
        out[f"{name}_score"] = [quintile_score(x, edges) for x in rows["change"]]
    out["severity_measured"] = out[["gpr_z", "epu_z"]].max(axis=1)
    out["severity_capped"] = out["severity_measured"].clip(upper=SEVERITY_CAP)
    out["severity_quintile"] = out[["gpr_score", "epu_score"]].max(axis=1)
    return out


def _at(jumps: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Row of ``jumps`` on ``date`` or the first available day after it."""
    pos = jumps.index.searchsorted(date)
    if pos >= len(jumps):
        return pd.Series({"before": float("nan"), "after": float("nan"), "change": float("nan")})
    return jumps.iloc[pos]


def score_events(events: pd.DataFrame, gpr: pd.Series, epu: pd.Series, umcsent: pd.Series,
                 calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Add the measured columns to a copy of ``events`` (hand columns untouched)."""
    ev = events[HAND_COLUMNS].copy()
    ev = pd.concat([ev, jump_columns(ev["date"], gpr, epu)], axis=1)

    monthly = umcsent.copy()
    monthly.index = monthly.index.to_period("M")
    months = ev["date"].dt.to_period("M")
    ev["sentiment_before"] = [monthly.get(m - 1, float("nan")) for m in months]
    ev["sentiment_after"] = [monthly.get(m + 1, float("nan")) if m + 1 <= DATA_END else float("nan")
                             for m in months]
    ev["sentiment_change"] = (ev["sentiment_after"] - ev["sentiment_before"]).round(1)

    day0 = [first_trading_day_on_or_after(calendar, d) for d in ev["date"]]
    pos = pd.Series([int(calendar.get_loc(d)) for d in day0])
    dist = [int((pos.drop(i) - p).abs().min()) if len(pos) > 1 else 10 ** 6 for i, p in pos.items()]
    ev["nearest_event_days"] = dist
    ev["near_other_event"] = ev["nearest_event_days"] <= NEAR_DAYS
    return ev


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    events = load_events()
    gpr = data.fetch_cached("gpr", "GPRD")
    epu = data.fetch_cached("epu", "daily_policy_index")
    umcsent = data.fetch_cached("fred", "UMCSENT")
    calendar = data.load().index
    scored = score_events(events, gpr, epu, umcsent, calendar)
    scored.to_csv(EVENTS_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Scored {len(scored)} events -> {EVENTS_PATH}")
    cols = ["date", "category", "severity_hand", "gpr_change", "gpr_z", "epu_change", "epu_z",
            "severity_measured", "severity_capped", "severity_quintile", "sentiment_change", "nearest_event_days",
            "near_other_event"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(scored[cols].to_string(index=False))
    print("\nseverity_measured by severity_hand:")
    print(scored.groupby("severity_hand").severity_measured.describe()[["count", "mean", "min", "50%", "max"]])


if __name__ == "__main__":
    main()
