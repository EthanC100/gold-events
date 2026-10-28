"""Event-study maths: raw and control-adjusted ("net") returns around a date.

Conventions
-----------
* **Day 0** is the first trading day whose return reflects the event.  The
  caller chooses it (for a Tuesday election, day 0 is the Wednesday after).
  :func:`first_trading_day_after` and :func:`first_trading_day_on_or_after`
  are the two helpers.
* The **h-day raw return** is the log return from the close of day -1 (the
  last close before the event is known) to the close of day h-1, i.e. it
  spans exactly h trading days of returns, starting with day 0::

      raw_h = log(P[day h-1]) - log(P[day -1])

* **Betas** come from an OLS of the asset's daily log return on the daily
  change in log(DXY) and the daily change in the real yield (percentage
  points), with an intercept, over the ``window`` trading days ending at
  day -1.  Nothing after day -1 is used, so the betas are known before the
  event.
* The **net return** removes the move explained by the controls over the
  same h-day window (the intercept is *not* subtracted; it is the asset's
  unconditional drift, not something the controls explain)::

      net_h = raw_h - beta_dxy * dlog(DXY)_h - beta_ry * d(real_yield)_h

Everything is expressed in log-return units; multiply by 100 for percent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 20)
WINDOW = 250
MIN_OBS = 200  # minimum valid daily observations in the beta window
ASSETS = ("gold", "silver")
CONTROLS = ("dxy", "real_yield")


@dataclass(frozen=True)
class Betas:
    dxy: float
    real_yield: float
    alpha: float
    n_obs: int


@dataclass
class EventStudyResult:
    """One event: a tidy table with one row per (asset, horizon)."""

    event_date: pd.Timestamp
    day0: pd.Timestamp
    table: pd.DataFrame  # columns: asset, horizon, raw, net, explained, d_dxy, d_real_yield, beta_dxy, beta_real_yield, n_obs

    def get(self, asset: str, horizon: int, col: str = "net") -> float:
        row = self.table[(self.table.asset == asset) & (self.table.horizon == horizon)]
        return float(row[col].iloc[0]) if len(row) else float("nan")


# --------------------------------------------------------------------------
# Calendar helpers
# --------------------------------------------------------------------------


def first_trading_day_after(index: pd.DatetimeIndex, date) -> pd.Timestamp:
    """First index date strictly after ``date``."""
    date = pd.Timestamp(date)
    pos = index.searchsorted(date, side="right")
    if pos >= len(index):
        raise ValueError(f"no trading day after {date.date()} in data")
    return index[pos]


def first_trading_day_on_or_after(index: pd.DatetimeIndex, date) -> pd.Timestamp:
    """``date`` itself if it is a trading day, else the next one."""
    date = pd.Timestamp(date)
    pos = index.searchsorted(date, side="left")
    if pos >= len(index):
        raise ValueError(f"no trading day on or after {date.date()} in data")
    return index[pos]


# --------------------------------------------------------------------------
# Core maths
# --------------------------------------------------------------------------


def control_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Daily control changes: dlog(DXY) and d(real_yield) in percentage points."""
    return pd.DataFrame({
        "dxy": np.log(df["dxy"]).diff(),
        "real_yield": df["real_yield"].diff(),
    })


def estimate_betas(asset_returns: pd.Series, controls: pd.DataFrame, min_obs: int = MIN_OBS) -> Betas:
    """OLS with intercept of daily asset log returns on the control changes.

    Rows with any NaN are dropped. Returns NaN betas if fewer than ``min_obs``
    rows survive.
    """
    data = pd.concat([asset_returns.rename("y"), controls], axis=1).dropna()
    n = len(data)
    if n < min_obs:
        return Betas(float("nan"), float("nan"), float("nan"), n)
    X = np.column_stack([np.ones(n), data["dxy"].to_numpy(), data["real_yield"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, data["y"].to_numpy(), rcond=None)
    return Betas(dxy=float(coef[1]), real_yield=float(coef[2]), alpha=float(coef[0]), n_obs=n)


def window_change(series: pd.Series, start_pos: int, end_pos: int, log: bool) -> float:
    """Change in ``series`` from position ``start_pos`` to ``end_pos`` (log diff if ``log``)."""
    if end_pos >= len(series) or start_pos < 0:
        return float("nan")
    a, b = series.iloc[start_pos], series.iloc[end_pos]
    if pd.isna(a) or pd.isna(b):
        return float("nan")
    return float(np.log(b) - np.log(a)) if log else float(b - a)


def event_study(
    df: pd.DataFrame,
    day0,
    horizons: tuple[int, ...] = HORIZONS,
    window: int = WINDOW,
    assets: tuple[str, ...] = ASSETS,
    min_obs: int = MIN_OBS,
    event_date=None,
) -> EventStudyResult:
    """Raw and net returns of ``assets`` around ``day0``.

    ``df`` is the aligned daily frame (see :mod:`gold_events.data`) and must
    contain the asset columns plus ``dxy`` and ``real_yield``.  ``day0`` must
    be a date in ``df.index``.
    """
    day0 = pd.Timestamp(day0)
    if day0 not in df.index:
        raise ValueError(f"day0 {day0.date()} is not a trading day in the data")
    pos0 = int(df.index.get_loc(day0))
    if pos0 < 1:
        raise ValueError("day0 must have at least one trading day before it")
    base = pos0 - 1  # position of day -1

    ctrl = control_changes(df)
    beta_window = ctrl.iloc[max(0, base - window + 1): base + 1]  # window days ending at day -1

    rows = []
    for asset in assets:
        rets = np.log(df[asset]).diff()
        betas = estimate_betas(rets.loc[beta_window.index], beta_window, min_obs=min_obs)
        for h in horizons:
            end = pos0 + h - 1
            raw = window_change(df[asset], base, end, log=True)
            d_dxy = window_change(df["dxy"], base, end, log=True)
            d_ry = window_change(df["real_yield"], base, end, log=False)
            explained = betas.dxy * d_dxy + betas.real_yield * d_ry
            rows.append({
                "asset": asset, "horizon": h, "raw": raw, "net": raw - explained,
                "explained": explained, "d_dxy": d_dxy, "d_real_yield": d_ry,
                "beta_dxy": betas.dxy, "beta_real_yield": betas.real_yield, "n_obs": betas.n_obs,
            })
    table = pd.DataFrame(rows)
    return EventStudyResult(event_date=pd.Timestamp(event_date) if event_date is not None else day0,
                            day0=day0, table=table)
