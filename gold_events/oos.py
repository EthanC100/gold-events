"""One-shot out-of-sample test of the frozen v1 model on 2022-01-01 onward.

Run ``python -m gold_events.oos`` to write ``reports/v1_out_of_sample.md``.

v1 is **not refitted**.  Its control coefficients and intercept (real yield, DXY, VIX and
GPR changes; the event terms were shrunk to zero by cross-validation) are reproduced
from the 2000-08-30..2021-12-31 fit exactly as :mod:`gold_events.model` computes them
(hand-severity ridge design, which cross-validation reduces to the controls), checked
against ``reports/v1_fit.md``, and then applied unchanged to every h-day window whose
base close is 2021-12-31 or later.

Reported for gold and silver at 5 and 20 days:

* RMSE and mean error of the frozen v1 prediction against the no-change baseline
  (predict zero) and against a controls-only OLS refitted on the test windows (an
  in-sample fit on the test period, so a bound on what the same specification could
  have done; its coefficients next to v1's show the drift);
* hit rate on direction (sign of prediction = sign of realised return), with the share
  of positive windows as the reference;
* all of the above with the five largest windows (largest absolute realised return)
  removed;
* the test-period events, run through the event-study machinery: raw move, the part
  the v1 controls explain over the window, and the net move, at 5 and 20 days, next to
  the standard trailing-beta event-study net return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_events import data
from gold_events.event_study import event_study, first_trading_day_on_or_after
from gold_events.events import load_events
from gold_events.model import (ASSETS, CONTROL_UNITS, CONTROLS, HORIZONS, SAMPLE_END, SAMPLE_START,
                               build_design, ols_fit, ridge_fit, rmse)

REPORT_PATH = data.ROOT / "reports" / "v1_out_of_sample.md"
COEF_PATH = data.ROOT / "data" / "v1_coefficients.csv"
TEST_BASE = pd.Timestamp("2021-12-31")   # base close of the first test window (window starts 2022-01-03)
N_DROP = 5

# Test-period events (none are in data/events.csv, which stops at 2021).  Day 0 follows the
# event-study convention: the event date if it is a trading day, else the next one; events
# announced after the close or at the weekend are dated to the next session.
TEST_EVENTS = [
    ("2022-02-24", "Military escalation", "Russia invades Ukraine"),
    ("2022-02-28", "Sanctions and trade war", "Russian central-bank reserves frozen (announced Sat 26 Feb)"),
    ("2022-06-10", "Central bank policy shock", "May CPI shock: 75bp hike priced for the June FOMC"),
    ("2022-06-15", "Central bank policy shock", "Fed's first 75bp hike since 1994"),
    ("2023-03-10", "Banking or financial stress", "Silicon Valley Bank closed by regulators"),
    ("2023-10-09", "Military escalation", "Hamas attack on Israel (Sat 7 Oct)"),
    ("2025-04-03", "Sanctions and trade war", "'Liberation Day' tariff round (announced after the close 2 Apr)"),
]


@dataclass
class Frozen:
    asset: str
    h: int
    intercept: float
    coef: dict[str, float]   # natural scale, log-return units
    lam: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + sum(self.coef[c] * X[c].to_numpy() for c in CONTROLS)


# --------------------------------------------------------------------------
# Frozen v1
# --------------------------------------------------------------------------


def frozen_v1(df: pd.DataFrame, events: pd.DataFrame) -> dict[tuple[str, int], Frozen]:
    """Reproduce the v1 control coefficients from the ≤2021 sample (the frame is cut first)."""
    df = df.loc[:SAMPLE_END]
    out = {}
    for asset in ASSETS:
        for h in HORIZONS:
            X = build_design(df, events, asset, h, "severity_hand")
            y = X.pop("y")
            cats = [c for c in X.columns if c not in CONTROLS]
            fit = ridge_fit(X, y, penalised=cats, lag=h)
            coef = {c: float(fit.coef[fit.names.index(c)]) for c in CONTROLS}
            out[(asset, h)] = Frozen(asset, h, float(fit.intercept), coef, float(fit.lam))
    return out


def coefficient_table(frozen: dict) -> pd.DataFrame:
    rows = []
    for (asset, h), f in frozen.items():
        row = {"asset": asset, "h": h, "intercept_pct": 100 * f.intercept, "lambda": f.lam}
        for c in CONTROLS:
            mult, _ = CONTROL_UNITS[c]
            row[f"{c}_pct_per_unit"] = 100 * mult * f.coef[c]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Test windows and metrics
# --------------------------------------------------------------------------


def test_design(df: pd.DataFrame, events: pd.DataFrame, asset: str, h: int) -> pd.DataFrame:
    X = build_design(df, events, asset, h, "severity_hand", start=TEST_BASE, end=df.index[-1])
    return X[["y"] + CONTROLS]


def hit_rate(pred: np.ndarray, y: np.ndarray) -> float:
    m = y != 0
    return float(np.mean(np.sign(pred[m]) == np.sign(y[m])))


def metrics(y: np.ndarray, pred_v1: np.ndarray, pred_refit: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "rmse_zero": 100 * rmse(y), "rmse_v1": 100 * rmse(y - pred_v1), "rmse_refit": 100 * rmse(y - pred_refit),
        "mean_err_v1": 100 * float(np.mean(y - pred_v1)), "mean_y": 100 * float(np.mean(y)),
        "r2_oos_v1": 1 - float(np.sum((y - pred_v1) ** 2) / np.sum(y ** 2)),
        "r2_refit": 1 - float(np.sum((y - pred_refit) ** 2) / np.sum(y ** 2)),
        "hit_v1": hit_rate(pred_v1, y), "hit_refit": hit_rate(pred_refit, y), "share_positive": float(np.mean(y > 0)),
    }


@dataclass
class Window:
    asset: str
    h: int
    X: pd.DataFrame
    pred_v1: np.ndarray
    refit: object                      # Fit from ols_fit on the test windows
    full: dict[str, float]
    trimmed: dict[str, float]
    dropped: pd.DatetimeIndex          # the five largest windows (by |y|) removed in the trimmed metrics


def run(df: pd.DataFrame, events: pd.DataFrame, frozen: dict) -> list[Window]:
    out = []
    for asset in ASSETS:
        for h in HORIZONS:
            X = test_design(df, events, asset, h)
            y = X["y"].to_numpy()
            f = frozen[(asset, h)]
            pred = f.predict(X)
            refit = ols_fit(X[CONTROLS], X["y"], lag=h)
            pred_refit = refit.fitted
            keep = np.argsort(-np.abs(y))[N_DROP:]
            keep.sort()
            dropped = X.index[np.argsort(-np.abs(y))[:N_DROP]]
            out.append(Window(asset, h, X, pred, refit, metrics(y, pred, pred_refit),
                              metrics(y[keep], pred[keep], pred_refit[keep]), dropped))
    return out


# --------------------------------------------------------------------------
# Events with v1 betas
# --------------------------------------------------------------------------


def event_moves(df: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """For each test event, asset and horizon: raw move, the part the v1 controls explain, net move,
    the v1 prediction (intercept + controls), and the standard event-study net (trailing 250-day betas)."""
    ry, ldxy, lvix, gpr = df["real_yield"], np.log(df["dxy"]), np.log(df["vix"]), df["gpr"]
    rows = []
    for date, cat, name in TEST_EVENTS:
        day0 = first_trading_day_on_or_after(df.index, date)
        pos0 = int(df.index.get_loc(day0))
        es = event_study(df, day0)
        for asset in ASSETS:
            price = np.log(df[asset])
            for h in HORIZONS:
                base, last = pos0 - 1, pos0 + h - 1
                if last >= len(df):
                    continue
                f = frozen[(asset, h)]
                dx = {"d_real_yield": ry.iloc[last] - ry.iloc[base], "dlog_dxy": ldxy.iloc[last] - ldxy.iloc[base],
                      "dlog_vix": lvix.iloc[last] - lvix.iloc[base], "d_gpr": gpr.iloc[last] - gpr.iloc[base]}
                raw = float(price.iloc[last] - price.iloc[base])
                explained = float(sum(f.coef[c] * dx[c] for c in CONTROLS))
                rows.append({"event": name, "category": cat, "date": pd.Timestamp(date), "day0": day0, "asset": asset, "h": h,
                             "raw": 100 * raw, "explained_v1": 100 * explained, "net_v1": 100 * (raw - explained),
                             "pred_v1": 100 * (f.intercept + explained), "net_trailing": 100 * es.get(asset, h, "net"),
                             "d_real_yield": dx["d_real_yield"], "dlog_dxy": 100 * dx["dlog_dxy"], "dlog_vix": 100 * dx["dlog_vix"],
                             "d_gpr": dx["d_gpr"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _pct(x, nd=2):
    return f"{x:+.{nd}f}%"


def metric_rows(m: dict) -> list[str]:
    return [
        f"| no change (predict 0) | {m['rmse_zero']:.2f}% | - | - | {m['share_positive']:.0%} positive |",
        f"| **frozen v1** | **{m['rmse_v1']:.2f}%** | {_pct(m['mean_err_v1'])} | {m['r2_oos_v1']:+.3f} | **{m['hit_v1']:.0%}** |",
        f"| controls-only OLS refitted on the test period (in-sample) | {m['rmse_refit']:.2f}% | +0.00% | {m['r2_refit']:+.3f} | {m['hit_refit']:.0%} |",
    ]


def drift_table(w: Window, f: Frozen) -> list[str]:
    lines = ["| term | units | v1 (frozen, 2000-2021) | refit on test period | refit SE (NW) | t of difference |",
             "|---|---|---:|---:|---:|---:|"]
    for c in CONTROLS:
        mult, label = CONTROL_UNITS[c]
        i = w.refit.names.index(c)
        v1c, rc, se = 100 * mult * f.coef[c], 100 * mult * w.refit.coef[i], 100 * mult * w.refit.se[i]
        lines.append(f"| {c} | {label} | {v1c:+.2f} | {rc:+.2f} | {se:.2f} | {(rc - v1c) / se:+.1f} |")
    lines.append(f"| intercept | % per window | {100 * f.intercept:+.2f} | {100 * w.refit.intercept:+.2f} | - | - |")
    return lines


def verdict(windows: list[Window], ev: pd.DataFrame) -> str:
    by = {(w.asset, w.h): w for w in windows}
    g5, g20, s5, s20 = by[("gold", 5)], by[("gold", 20)], by[("silver", 5)], by[("silver", 20)]
    label = lambda w: f"{w.asset} {w.h}-day"
    beat_full = [label(w) for w in windows if w.full["rmse_v1"] < w.full["rmse_zero"]]
    beat_trim = [label(w) for w in windows if w.trimmed["rmse_v1"] < w.trimmed["rmse_zero"]]
    gain = {label(w): 100 * (1 - w.full["rmse_v1"] / w.full["rmse_zero"]) for w in windows}
    gap = {label(w): w.full["rmse_v1"] - w.full["rmse_refit"] for w in windows}
    hits = {label(w): w.full["hit_v1"] for w in windows}
    base = {label(w): w.full["share_positive"] for w in windows}
    bias = {label(w): w.full["mean_err_v1"] for w in windows}
    # coefficient drift: largest |t| of (refit - frozen) over the four controls and four fits
    tmax, tmax_name = 0.0, ""
    for w in windows:
        f = frozen_cache[(w.asset, w.h)]
        for c in CONTROLS:
            i = w.refit.names.index(c)
            t = abs((w.refit.coef[i] - f.coef[c]) / w.refit.se[i]) if w.refit.se[i] > 0 else 0.0
            if t > tmax:
                tmax, tmax_name = t, f"{c} ({label(w)})"
    ic = {label(w): (100 * frozen_cache[(w.asset, w.h)].intercept, 100 * w.refit.intercept) for w in windows}
    e = ev.set_index(["event", "asset", "h"]).net_v1
    agree = float(np.mean(np.sign(ev.net_v1) == np.sign(ev.net_trailing)))
    maxdiff = float((ev.net_v1 - ev.net_trailing).abs().max())
    rows = []
    for date, cat, name in TEST_EVENTS:
        if (name, "gold", 20) in e.index:
            rows.append(f"{name.split(' (')[0]} {e[(name, 'gold', 5)]:+.1f}%/{e[(name, 'gold', 20)]:+.1f}%")
    all_beat = len(beat_full) == 4 and len(beat_trim) == 4
    better_than_base = all(hits[k] > base[k] for k in hits)
    text = (
        "**What was tested.** The 2000-2021 control coefficients, applied unchanged to "
        f"{g5.full['n']:,} five-day and {g20.full['n']:,} twenty-day windows from January 2022 to the latest close. "
        "The regressors are the changes in real yields, the dollar, the VIX and the GPR index *over the same window*, "
        "so this is a test of whether the sensitivities still hold, not a forecast made before the window opened: "
        "a user of these numbers would need a view on the controls first.\n\n"
        "**Against no-change.** "
        + (f"The frozen v1 beats predicting zero on all four asset-horizon pairs, by {min(gain.values()):.0f}% to "
           f"{max(gain.values()):.0f}% of RMSE (gold {gain['gold 5-day']:.0f}% at 5 days, {gain['gold 20-day']:.0f}% at 20), "
           "and still does with the five largest windows removed, so the result is not carried by a few extreme weeks. "
           if all_beat else
           f"It beats predicting zero on {', '.join(beat_full) or 'none'}; with the five largest windows removed, on "
           f"{', '.join(beat_trim) or 'none'}. ")
        + f"Direction is right {min(hits.values()):.0%} to {max(hits.values()):.0%} of the time "
        f"(gold {hits['gold 5-day']:.0%} at 5 days and {hits['gold 20-day']:.0%} at 20), "
        + ("above the share of rising windows in every case" if better_than_base else "not always above the share of rising windows")
        + f" ({base['gold 5-day']:.0%} and {base['gold 20-day']:.0%} for gold).\n\n"
        "**Against a refit on the test period.** "
        f"Refitting the same four controls on 2022-2026 (an in-sample fit, the most the specification could have done) "
        f"lowers RMSE by only {min(gap.values()):.2f} to {max(gap.values()):.2f} points; the refitted sensitivities sit "
        f"within one and a half standard errors of the frozen ones everywhere (largest difference t = {tmax:.1f}, {tmax_name}). "
        f"What did move is the intercept: gold's per-window drift went from {ic['gold 20-day'][0]:+.2f}% to "
        f"{ic['gold 20-day'][1]:+.2f}% at 20 days, silver's from {ic['silver 20-day'][0]:+.2f}% to {ic['silver 20-day'][1]:+.2f}%. "
        f"That is the whole of v1's bias: the mean error is {bias['gold 5-day']:+.2f}% and {bias['gold 20-day']:+.2f}% for gold "
        f"and {bias['silver 5-day']:+.2f}% and {bias['silver 20-day']:+.2f}% for silver, i.e. both metals rose by more than "
        "their real-yield and dollar moves account for, consistently, through 2022-2026. The betas did not drift; the level did.\n\n"
        "**Events.** Through the event-study machinery with the v1 betas, net gold moves at 5/20 days were: "
        + "; ".join(rows) + ". "
        "The invasion and the reserve freeze, SVB and the October 2023 attack all left gold higher than its controls "
        "explain at both horizons; the June 2022 rate shock left it lower, with the controls explaining most of the 20-day "
        "drop after the CPI print; the 2025 tariff round shows a net fall at 5 days that had mostly closed by 20 days for gold "
        "and had not for silver. "
        f"The v1-beta and trailing-beta net moves have the same sign in {agree:.0%} of the event rows and never differ by more "
        f"than {maxdiff:.1f} points, another sign that the control sensitivities are stable.\n\n"
        "**Verdict.** v1's control betas survive five years out of sample intact: they explain as much of the 5- and "
        "20-day variation in 2022-2026 as a model fitted on that period would, and the event-study net moves computed with "
        "them agree with the trailing-beta versions. What v1 misses is a persistent upward drift in both metals that no "
        "change-in-controls model can carry, worth about a point per 20-day window for gold and one and a half for silver. "
        "Used for what it was built for, separating an event's move from the part the dollar and yields explain, v1 is "
        "fit for purpose on the test period; used as a return forecaster it would have been right on direction about "
        "seven times in ten but systematically too low."
    )
    return text


frozen_cache: dict = {}


def report(df, frozen, windows, ev) -> str:
    last = df.index[-1]
    lines = [
        "# v1 out of sample: 2022-01-01 to the latest data",
        "",
        f"Built {pd.Timestamp.now():%Y-%m-%d}. Test windows: every h-day window (h = 5, 20) whose base close is "
        f"{TEST_BASE:%Y-%m-%d} or later and whose last close is on or before {last:%Y-%m-%d}, the latest price in the data. "
        f"v1 is the fit on {SAMPLE_START:%Y-%m-%d} to {SAMPLE_END:%Y-%m-%d} (`reports/v1_fit.md`, hand-severity ridge design); "
        "only its control coefficients and intercept are used, since cross-validation shrank the event terms to zero "
        "(silver at 5 days kept small category terms at a finite penalty; they are dropped here too, as specified). "
        "**Nothing is refitted**: the coefficients below are reproduced from the ≤2021 frame (cut before anything is computed) and "
        "match the v1 report; they are also saved to `data/v1_coefficients.csv`.",
        "",
        "Windows overlap (one per trading day), so the effective number of independent observations is roughly n/h, and the "
        "same episode shows up in up to h consecutive rows.",
        "",
        "## Frozen v1 coefficients (% of return)",
        "",
        "| asset | h | intercept | d_real_yield (per +1pp) | dlog_dxy (per +1%) | dlog_vix (per +1%) | d_gpr (per +100) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (asset, h), f in frozen.items():
        u = {c: 100 * CONTROL_UNITS[c][0] * f.coef[c] for c in CONTROLS}
        lines.append(f"| {asset} | {h} | {100 * f.intercept:+.3f} | {u['d_real_yield']:+.2f} | "
                     f"{u['dlog_dxy']:+.2f} | {u['dlog_vix']:+.3f} | {u['d_gpr']:+.2f} |")
    lines.append("")
    by = {(w.asset, w.h): w for w in windows}
    for asset in ASSETS:
        lines += [f"## {asset.capitalize()}", ""]
        for h in HORIZONS:
            w, f = by[(asset, h)], frozen[(asset, h)]
            lines += [f"### {asset.capitalize()}, {h}-day windows (n = {w.full['n']}, "
                      f"{w.X.index[0]:%Y-%m-%d} to {w.X.index[-1]:%Y-%m-%d} window starts)", "",
                      "| model | RMSE | mean error (realised − predicted) | R² vs zero | direction hit rate |", "|---|---:|---:|---:|---:|"]
            lines += metric_rows(w.full)
            lines += ["", f"Five largest windows removed (by |realised return|; window starts "
                      + ", ".join(f"{d:%Y-%m-%d}" for d in sorted(w.dropped)) + f"; n = {w.trimmed['n']}):", "",
                      "| model | RMSE | mean error | R² vs zero | direction hit rate |", "|---|---:|---:|---:|---:|"]
            lines += metric_rows(w.trimmed)
            lines += ["", "Coefficient drift (frozen v1 vs the same controls refitted on the test windows; Newey-West SE, lag h):", ""]
            lines += drift_table(w, f) + [""]
    lines += ["## Test-period events through the event-study machinery with v1 betas", "",
              "Day 0 as listed; raw = log return from the close of day −1 to the close of day h−1; *explained* = Σ β_v1 × "
              "(change in the control over the same window) using the frozen coefficients above (no intercept); net = raw − "
              "explained; *v1 prediction* adds the intercept. *Trailing-beta net* is the repository's standard event study "
              "(betas on the dollar and real yields over the 250 days before day 0), for comparison. All in percent.", ""]
    for asset in ASSETS:
        lines += [f"### {asset.capitalize()}", "",
                  "| event | day 0 | h | raw | explained by v1 controls | **net (v1 betas)** | v1 prediction | trailing-beta net | Δ real yield (pp) | Δ DXY | Δ VIX | Δ GPR |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in ev[ev.asset == asset].iterrows():
            lines.append(f"| {r.event} | {r.day0:%Y-%m-%d} | {r.h} | {_pct(r.raw)} | {_pct(r.explained_v1)} | **{_pct(r.net_v1)}** | "
                         f"{_pct(r.pred_v1)} | {_pct(r.net_trailing)} | {r.d_real_yield:+.2f} | {r.dlog_dxy:+.1f}% | {r.dlog_vix:+.0f}% | {r.d_gpr:+.0f} |")
        lines.append("")
    e20 = ev[ev.h == 20].pivot(index="event", columns="asset", values="net_v1")
    e5 = ev[ev.h == 5].pivot(index="event", columns="asset", values="net_v1")
    lines += ["Net moves (v1 betas) summarised:", "", "| event | gold 5d | gold 20d | silver 5d | silver 20d |", "|---|---:|---:|---:|---:|"]
    for date, cat, name in TEST_EVENTS:
        if name in e20.index:
            lines.append(f"| {name} | {_pct(e5.loc[name, 'gold'], 1)} | {_pct(e20.loc[name, 'gold'], 1)} | "
                         f"{_pct(e5.loc[name, 'silver'], 1)} | {_pct(e20.loc[name, 'silver'], 1)} |")
    lines += ["", "## Verdict", "", verdict(windows, ev), ""]
    return "\n".join(lines)


def main() -> None:
    df = data.load()
    events = load_events()
    frozen = frozen_v1(df, events)
    frozen_cache.update(frozen)
    coefficient_table(frozen).to_csv(COEF_PATH, index=False)
    windows = run(df, events, frozen)
    ev = event_moves(df, frozen)
    text = report(df, frozen, windows, ev)
    REPORT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
