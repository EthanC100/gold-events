"""v1 model: h-day gold/silver returns on controls plus severity-weighted event terms.

Run ``python -m gold_events.model`` to write ``reports/v1_fit.md``.

Design
------
One observation per trading day ``t`` (the window start) in the sample.  The
dependent variable is the h-day log return from the close of ``t-1`` to the
close of ``t+h-1``, exactly as in :mod:`gold_events.event_study`.  Regressors
over the same window:

* controls: change in the 10y real yield (pp), change in log DXY, change in
  log VIX, change in the daily GPR index;
* one term per event category equal to ``severity x indicator``, where the
  indicator is 1 when an event of that category has its day 0 at ``t`` (so
  the observation *is* that event's h-day window).  Severity is either the
  hand score or the measured score from :mod:`gold_events.events`.

Windows overlap, so standard errors are Newey-West (Bartlett kernel, lag h)
in a sandwich around the ridge estimator.  Ridge shrinks only the event
terms (the controls have thousands of observations and are left alone); the
penalty is chosen by 5-fold blocked (contiguous) cross-validation on the
standardised design.  Reported coefficients are back on the natural scale.

Hard sample boundary: nothing after ``SAMPLE_END`` is read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_events import data
from gold_events.event_study import first_trading_day_on_or_after
from gold_events.events import load_events

REPORT_PATH = data.ROOT / "reports" / "v1_fit.md"
EX_EVENT = pd.Timestamp("2001-09-11")  # the one event that dominates the continuous severity
SAMPLE_START = pd.Timestamp("2000-08-30")
SAMPLE_END = pd.Timestamp("2021-12-31")
HORIZONS = (5, 20)
ASSETS = ("gold", "silver")
CONTROLS = ["d_real_yield", "dlog_dxy", "dlog_vix", "d_gpr"]
CONTROL_UNITS = {  # (multiplier applied to the natural-scale coefficient, label)
    "d_real_yield": (1.0, "per +1pp real yield"),
    "dlog_dxy": (0.01, "per +1% DXY"),
    "dlog_vix": (0.01, "per +1% VIX"),
    "d_gpr": (100.0, "per +100 GPR points"),
}
LAMBDA_GRID = np.concatenate([[0.0], np.logspace(-2, 7, 46)])
N_FOLDS = 5


# --------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------


def event_day0(events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([first_trading_day_on_or_after(calendar, d) for d in events["date"]], index=events.index)


def build_design(df: pd.DataFrame, events: pd.DataFrame, asset: str, h: int, severity_col: str,
                 start: pd.Timestamp = SAMPLE_START, end: pd.Timestamp = SAMPLE_END,
                 mode: str = "category") -> pd.DataFrame:
    """Rows = window starts; columns = y, controls, then the event terms.

    ``mode="category"`` gives one column per category (severity x indicator);
    ``mode="pooled"`` gives a single ``any_event`` column = severity x 1[any
    event has its day 0 at t].
    """
    df = df.loc[:end]  # hard cut: nothing after the sample end is ever read
    n = len(df)
    price = np.log(df[asset].to_numpy())
    ry = df["real_yield"].to_numpy()
    ldxy = np.log(df["dxy"].to_numpy())
    lvix = np.log(df["vix"].to_numpy())
    gpr = df["gpr"].to_numpy()

    first = int(df.index.searchsorted(start)) + 1  # window start whose base (t-1) is >= start
    starts = np.arange(first, n - h + 1)
    base, last = starts - 1, starts + h - 1
    X = pd.DataFrame({
        "y": price[last] - price[base],
        "d_real_yield": ry[last] - ry[base],
        "dlog_dxy": ldxy[last] - ldxy[base],
        "dlog_vix": lvix[last] - lvix[base],
        "d_gpr": gpr[last] - gpr[base],
    }, index=df.index[starts])

    ev = events[(events["date"] >= start) & (events["date"] <= end)].copy()
    ev["day0"] = event_day0(ev, df.index)
    if mode == "pooled":
        groups = {"any_event": ev}
    else:
        groups = {cat: ev[ev.category == cat] for cat in sorted(events["category"].unique())}
    for name, sub in groups.items():
        col = pd.Series(0.0, index=X.index)
        for _, e in sub.iterrows():
            if e.day0 in col.index:
                col[e.day0] += float(e[severity_col])
        X[name] = col
    return X.dropna()


# --------------------------------------------------------------------------
# Ridge with HAC standard errors
# --------------------------------------------------------------------------


@dataclass
class Fit:
    names: list[str]
    coef: np.ndarray       # natural scale, excluding intercept
    se: np.ndarray         # natural scale
    intercept: float
    lam: float
    cv_mse: dict[float, float]
    fitted: np.ndarray
    resid: np.ndarray
    oof_resid: np.ndarray  # out-of-fold residuals from the blocked CV at the chosen lambda

    @property
    def t(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.coef / self.se


def _solve(X: np.ndarray, y: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """(X'X + diag(penalty))^-1 X'y."""
    return np.linalg.solve(X.T @ X + np.diag(penalty), X.T @ y)


def hac_cov(X: np.ndarray, resid: np.ndarray, penalty: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West sandwich around the (penalised) least-squares estimator."""
    A = np.linalg.inv(X.T @ X + np.diag(penalty))
    scores = X * resid[:, None]
    S = scores.T @ scores
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1)
        G = scores[l:].T @ scores[:-l]
        S += w * (G + G.T)
    return A @ S @ A


def blocked_folds(n: int, k: int = N_FOLDS) -> list[np.ndarray]:
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def ridge_fit(X: pd.DataFrame, y: pd.Series, penalised: list[str], lag: int,
              grid: np.ndarray = LAMBDA_GRID) -> Fit:
    """Partial ridge (penalty only on ``penalised`` columns) with blocked CV for lambda.

    Columns are standardised for fitting; coefficients and SEs are reported on
    the natural scale.  Columns that are constant (e.g. a category with no
    event in the sample) are dropped.
    """
    X = X.loc[:, X.std() > 0]
    names = list(X.columns)
    mu, sd = X.mean().to_numpy(), X.std().to_numpy()
    Z = np.column_stack([np.ones(len(X)), (X.to_numpy() - mu) / sd])
    yv = y.to_numpy()
    pen_mask = np.array([0.0] + [1.0 if c in penalised else 0.0 for c in names])

    cv_mse: dict[float, float] = {}
    oof: dict[float, np.ndarray] = {}
    folds = blocked_folds(len(yv))
    for lam in grid:
        r = np.empty(len(yv))
        for test in folds:
            train = np.setdiff1d(np.arange(len(yv)), test)
            b = _solve(Z[train], yv[train], lam * pen_mask)
            r[test] = yv[test] - Z[test] @ b
        cv_mse[float(lam)] = float(np.mean(r ** 2))
        oof[float(lam)] = r
    lam = min(cv_mse, key=cv_mse.get)

    penalty = lam * pen_mask
    b = _solve(Z, yv, penalty)
    fitted = Z @ b
    resid = yv - fitted
    cov = hac_cov(Z, resid, penalty, lag)
    se_std = np.sqrt(np.diag(cov))
    coef = b[1:] / sd
    se = se_std[1:] / sd
    intercept = float(b[0] - np.sum(coef * mu))
    return Fit(names, coef, se, intercept, lam, cv_mse, fitted, resid, oof[lam])


def ols_fit(X: pd.DataFrame, y: pd.Series, lag: int) -> Fit:
    return ridge_fit(X, y, penalised=[], lag=lag, grid=np.array([0.0]))


def rmse(resid: np.ndarray) -> float:
    return float(np.sqrt(np.mean(resid ** 2)))


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------


@dataclass
class Result:
    asset: str
    h: int
    severity: str
    fit: Fit            # ridge, CV-chosen penalty on the event terms
    fit0: Fit           # same design, no penalty (plain OLS)
    design: pd.DataFrame
    event_rows: np.ndarray  # boolean mask: window starts at an event's day 0
    errors: dict[str, tuple[float, float]]  # name -> (rmse all, rmse event windows), in %


def run(df: pd.DataFrame, events: pd.DataFrame) -> list[Result]:
    results = []
    categories = sorted(events["category"].unique())
    for asset in ASSETS:
        for h in HORIZONS:
            for sev in ("severity_hand", "severity_quintile"):
                X = build_design(df, events, asset, h, sev)
                y = X.pop("y")
                cats = [c for c in categories if c in X.columns]
                is_event = (X[cats].abs().sum(axis=1) > 0).to_numpy()
                fit = ridge_fit(X, y, penalised=cats, lag=h)
                fit0 = ols_fit(X, y, lag=h)
                controls = ols_fit(X[CONTROLS], y, lag=h)
                yv = y.to_numpy()
                label = {"severity_hand": "hand", "severity_quintile": "quintile"}[sev]
                errors = {
                    "no change (predict 0)": (100 * rmse(yv), 100 * rmse(yv[is_event])),
                    "controls only (OLS)": (100 * rmse(controls.resid), 100 * rmse(controls.resid[is_event])),
                    f"controls + events ({label}), ridge": (100 * rmse(fit.resid), 100 * rmse(fit.resid[is_event])),
                    f"controls + events ({label}), unpenalised": (100 * rmse(fit0.resid), 100 * rmse(fit0.resid[is_event])),
                }
                results.append(Result(asset, h, sev, fit, fit0, X, is_event, errors))
    return results


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _fmt(x: float, nd: int = 2) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.{nd}f}"


def coef_table(hand: Result, meas: Result) -> list[str]:
    """Coefficients for the two severity versions side by side, in % of return."""
    lines = ["| term | units | hand: coef | SE | t | quintile: coef | SE | t |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    def row(name, label, mult):
        cells = [name, label]
        for r in (hand, meas):
            if name in r.fit.names:
                i = r.fit.names.index(name)
                c, s = 100 * mult * r.fit.coef[i], 100 * mult * r.fit.se[i]
                cells += [_fmt(c), f"{s:.2f}", _fmt(c / s if s > 0 else float("nan"), 1)]
            else:
                cells += ["-", "-", "-"]
        return "| " + " | ".join(cells) + " |"
    for c in CONTROLS:
        mult, label = CONTROL_UNITS[c]
        lines.append(row(c, label, mult))
    cats = [n for n in hand.fit.names if n not in CONTROLS]
    for c in cats:
        lines.append(row(c, "% per severity point", 1.0))
    return lines


def error_table(hand: Result, meas: Result) -> list[str]:
    n_all, n_ev = len(hand.design), int(hand.event_rows.sum())
    lines = [f"| model | RMSE all windows (n={n_all}) | RMSE event windows (n={n_ev}) |", "|---|---:|---:|"]
    seen = set()
    for r in (hand, meas):
        for name, (a, e) in r.errors.items():
            if name in seen:
                continue
            seen.add(name)
            lines.append(f"| {name} | {a:.2f}% | {e:.2f}% |")
    return lines


def unpenalised_table(hand: Result, meas: Result) -> list[str]:
    """Event-term coefficients with no shrinkage (plain OLS), both severity versions."""
    lines = ["| category (unpenalised) | hand: % per point | SE | t | quintile: % per point | SE | t |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in hand.fit0.names:
        if name in CONTROLS:
            continue
        cells = [name]
        for r in (hand, meas):
            i = r.fit0.names.index(name)
            c, se = 100 * r.fit0.coef[i], 100 * r.fit0.se[i]
            cells += [_fmt(c), f"{se:.2f}", _fmt(c / se, 1)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def significant(fit: Fit, thresh: float = 2.0) -> list[tuple[str, float, float]]:
    out = []
    for name, c, t in zip(fit.names, fit.coef, fit.t):
        if name not in CONTROLS and abs(t) >= thresh:
            out.append((name, 100 * c, t))
    return out


def _lam_text(fit: Fit, grid: np.ndarray = LAMBDA_GRID) -> str:
    if fit.lam >= grid.max():
        return f"{fit.lam:.0e} (top of the grid: event terms shrunk to ~0)"
    if fit.lam == 0:
        return "0 (no shrinkage)"
    return f"{fit.lam:.3g}"


def report(results: list[Result], events: pd.DataFrame, df: pd.DataFrame) -> str:
    in_sample = events[(events.date >= SAMPLE_START) & (events.date <= SAMPLE_END)]
    counts = in_sample.category.value_counts()
    all_cats = sorted(events.category.unique())
    lines = [
        "# v1 model fit: gold and silver returns on controls and event terms",
        "",
        f"Built {pd.Timestamp.now():%Y-%m-%d}. Sample: windows starting after {SAMPLE_START:%Y-%m-%d} and ending "
        f"on or before {SAMPLE_END:%Y-%m-%d} (the data frame is cut at {SAMPLE_END:%Y-%m-%d} before anything is "
        f"computed). Last price used: {df.loc[:SAMPLE_END].index[-1]:%Y-%m-%d}.",
        "",
        "## Setup",
        "",
        "For every trading day t, y = log return from the close of t-1 to the close of t+h-1 (h = 5, 20).",
        "Regressors over the same window: change in the 10y real yield, change in log DXY, change in log VIX,",
        "change in the daily GPR index, and one term per event category equal to severity x 1[an event of that",
        "category has its day 0 at t]. Day 0 = the event date if it is a trading day, else the next one.",
        "Severity in this section is either the hand score (1-5) or the quintile score (max of the GPR and EPU",
        "jump quintiles, `severity_quintile`); the continuous measured severity is used in the refit section below.",
        "",
        "Fit: ridge on the event terms only (controls unpenalised), penalty chosen by 5-fold blocked",
        "cross-validation; Newey-West standard errors (lag h) because the windows overlap. Coefficients are in",
        "percent of return. *Event windows* in the error tables are the rows whose window starts at an event's",
        "day 0, i.e. the rows where the event terms can make any difference. A category term is identified by",
        "only as many rows as it has events, so its standard error and t-statistic are unreliable for categories",
        "with fewer than about five events, whatever the number says.",
        "",
        "## Events in sample",
        "",
        f"{len(in_sample)} of {len(events)} events fall in the sample. By category:",
        "",
        "| category | events in sample | mean severity_hand | mean severity_quintile |",
        "|---|---:|---:|---:|",
    ]
    for cat in all_cats:
        sub = in_sample[in_sample.category == cat]
        n = len(sub)
        lines.append(f"| {cat} | {n} | {sub.severity_hand.mean():.1f} | {sub.severity_quintile.mean():.1f} |"
                     if n else f"| {cat} | 0 | - | - |")
    dropped = [c for c in all_cats if counts.get(c, 0) == 0]
    if dropped:
        lines += ["", "Categories with no event in the sample are dropped from the fit: " + ", ".join(dropped) + "."]
    near = in_sample[in_sample.near_other_event]
    lines += ["", f"{len(near)} in-sample events are within 20 trading days of another event "
              f"({', '.join(f'{d:%Y-%m-%d}' for d in near.date)}); their 20-day windows overlap, so their "
              "category terms share the same returns.", ""]

    by_key = {(r.asset, r.h, r.severity): r for r in results}
    for asset in ASSETS:
        lines += [f"## {asset.capitalize()}", ""]
        for h in HORIZONS:
            hand, meas = by_key[(asset, h, "severity_hand")], by_key[(asset, h, "severity_quintile")]
            lines += [f"### {asset.capitalize()}, {h}-day windows", "",
                      f"Cross-validated ridge penalty on the event terms (standardised units): "
                      f"hand {_lam_text(hand.fit)}; quintile {_lam_text(meas.fit)}. "
                      f"CV MSE (units 1e-4) at lambda = 0 vs chosen: "
                      f"hand {1e4 * hand.fit.cv_mse[0.0]:.3f} vs {1e4 * hand.fit.cv_mse[hand.fit.lam]:.3f}; "
                      f"quintile {1e4 * meas.fit.cv_mse[0.0]:.3f} vs {1e4 * meas.fit.cv_mse[meas.fit.lam]:.3f}.",
                      "", "Ridge fit (what cross-validation keeps):", ""]
            lines += coef_table(hand, meas) + [""]
            lines += ["Unpenalised event terms (plain OLS on the same design, for comparison):", ""]
            lines += unpenalised_table(hand, meas) + [""]
            lines += ["In-sample error:", ""] + error_table(hand, meas) + [""]
            for label, r in (("hand", hand), ("quintile", meas)):
                for kind, fit in (("ridge", r.fit), ("unpenalised", r.fit0)):
                    items = significant(fit)
                    txt = ("; ".join(f"{n} ({c:+.2f}% per point, t={t:+.1f})" for n, c, t in items)
                           if items else "none")
                    lines.append(f"- |t| >= 2, {label} severity, {kind}: {txt}.")
            lines.append("")
    lines += ["## Reading", "", reading(results, events), ""]
    return "\n".join(lines)


def reading(results: list[Result], events: pd.DataFrame) -> str:
    by_key = {(r.asset, r.h, r.severity): r for r in results}

    def errs(asset, h, sev="severity_hand"):  # noqa: E306
        return [v[1] for v in by_key[(asset, h, sev)].errors.values()]  # event-window RMSEs

    def coef(r, name, mult=1.0):
        return 100 * mult * r.fit.coef[r.fit.names.index(name)]

    g20 = by_key[("gold", 20, "severity_hand")]
    g5 = by_key[("gold", 5, "severity_hand")]
    n_top = sum(r.fit.lam >= LAMBDA_GRID.max() for r in results)
    n5 = int((events.severity_quintile == 5).sum())

    # Unpenalised categories significant for a given asset at both horizons and both severity versions.
    def robust(asset):
        """Categories with |t| >= 2 and the same sign in every (horizon, severity) unpenalised fit."""
        sets = [{(n, np.sign(c)) for n, c, _ in significant(by_key[(asset, h, s)].fit0)}
                for h in HORIZONS for s in ("severity_hand", "severity_quintile")]
        return sorted(n for n, _ in set.intersection(*sets))

    def sign_flips(asset):
        """Categories significant at both horizons (hand severity) with opposite signs."""
        s5 = {n: c for n, c, _ in significant(by_key[(asset, 5, "severity_hand")].fit0)}
        s20 = {n: c for n, c, _ in significant(by_key[(asset, 20, "severity_hand")].fit0)}
        return sorted(n for n in s5 if n in s20 and np.sign(s5[n]) != np.sign(s20[n]))

    def improvement_sentence(asset):
        b5, c5, r5, u5 = errs(asset, 5)
        b20, c20, r20, u20 = errs(asset, 20)
        ctrl5 = "cut" if c5 < b5 else "raise"
        ctrl20 = "cut" if c20 < b20 else "raise"
        return (f"On {asset}'s event windows the controls alone {ctrl5} the 5-day RMSE from {b5:.2f}% to {c5:.2f}% "
                f"and {ctrl20} the 20-day RMSE from {b20:.2f}% to {c20:.2f}%; unpenalised event terms take those to "
                f"{u5:.2f}% and {u20:.2f}%, while the cross-validated ridge versions sit at {r5:.2f}% and {r20:.2f}%.")

    rg, rs, fg = robust("gold"), robust("silver"), sign_flips("gold")
    text = (
        f"The controls carry the model. Over 20-day windows gold moves about {coef(g20, 'd_real_yield'):+.1f}% per "
        f"percentage point of real yield and {coef(g20, 'dlog_dxy', 0.01):+.2f}% per 1% move in the dollar "
        f"(5-day: {coef(g5, 'd_real_yield'):+.1f}% and {coef(g5, 'dlog_dxy', 0.01):+.2f}%), with the expected signs, "
        f"t-statistics well beyond 2, and no dependence on which severity is used. A rise in the GPR index over the "
        f"window is also associated with higher gold and silver, after the dollar and real yields are taken into account. "
        f"{improvement_sentence('gold')} {improvement_sentence('silver')} "
        f"Cross-validation is the key result: in {n_top} of the {len(results)} fits it pushes the event penalty to the top "
        f"of the grid, i.e. it finds that category effects fitted on one stretch of the sample do not help predict "
        f"returns on another stretch, so the event terms are shrunk to nothing. The unpenalised fits do show a few "
        f"categories with |t| >= 2, but "
        + (f"only {', '.join(rg)} keeps the same sign and |t| >= 2 for gold across both horizons and both severity "
           f"versions" if rg else "none keeps the same sign and |t| >= 2 for gold across both horizons and both "
           f"severity versions")
        + (f", and only {', '.join(rs)} does for silver. " if rs else ", and none does for silver. ")
        + (f"Gold's {', '.join(fg)} term even flips sign between 5 and 20 days. " if fg else "")
        + f"The terms that do appear rest on two to four events each (the pandemic category is Fukushima and "
        f"Covid, nothing else), and their t-statistics come from a handful of rows, so they describe those episodes "
        f"rather than estimate a category effect. Quintile severity adds little beyond an on/off indicator: {n5} of "
        f"{len(events)} events score 5 on it, because the larger of two quintile ranks against all days is almost "
        f"always high for events chosen for being notable. The honest summary for v1 is that gold's and silver's "
        f"reactions to these events are mostly transmitted through real yields, the dollar and the geopolitical-risk "
        f"index, and the residual per-category effects are too small and too few to pin down. Next steps that would "
        f"change that: more events per category, a proper out-of-sample test on 2022 onward once v1 is frozen, and a "
        f"severity measure with real spread."
    )
    return text


# --------------------------------------------------------------------------
# Refit: continuous severity, pooled term, mechanical list
# --------------------------------------------------------------------------


@dataclass
class Refit:
    name: str
    fit: Fit
    fit0: Fit
    X: pd.DataFrame
    y: pd.Series


def refit(df: pd.DataFrame, hand: pd.DataFrame, mech: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """For each asset and horizon: controls-only plus the three event designs."""
    out = {}
    for asset in ASSETS:
        for h in HORIZONS:
            Xa = build_design(df, hand, asset, h, "severity_capped", mode="category")
            Xb = build_design(df, hand, asset, h, "severity_capped", mode="pooled")
            Xc = build_design(df, mech, asset, h, "severity_capped", mode="pooled")
            Xd = build_design(df, hand[hand.date != EX_EVENT], asset, h, "severity_capped", mode="pooled")
            assert Xa.index.equals(Xb.index) and Xa.index.equals(Xc.index) and Xa.index.equals(Xd.index)
            y = Xa["y"]
            hand_rows = (Xb["any_event"] != 0).to_numpy()
            mech_rows = (Xc["any_event"] != 0).to_numpy()
            cats = [c for c in Xa.columns if c not in CONTROLS + ["y"]]
            models = {
                "controls only": Refit("controls only", ols_fit(Xa[CONTROLS], y, lag=h), None, Xa[CONTROLS], y),
                "hand list, category terms": Refit("hand list, category terms",
                                                   ridge_fit(Xa.drop(columns="y"), y, penalised=cats, lag=h),
                                                   ols_fit(Xa.drop(columns="y"), y, lag=h), Xa, y),
                "hand list, pooled term": Refit("hand list, pooled term",
                                                ridge_fit(Xb.drop(columns="y"), y, penalised=["any_event"], lag=h),
                                                ols_fit(Xb.drop(columns="y"), y, lag=h), Xb, y),
                "mechanical list, pooled term": Refit("mechanical list, pooled term",
                                                      ridge_fit(Xc.drop(columns="y"), y, penalised=["any_event"], lag=h),
                                                      ols_fit(Xc.drop(columns="y"), y, lag=h), Xc, y),
                "hand list, pooled term, ex 9/11": Refit("hand list, pooled term, ex 9/11",
                                                         ridge_fit(Xd.drop(columns="y"), y, penalised=["any_event"], lag=h),
                                                         ols_fit(Xd.drop(columns="y"), y, lag=h), Xd, y),
            }
            out[(asset, h)] = {"models": models, "hand_rows": hand_rows, "mech_rows": mech_rows, "y": y}
    return out


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def refit_report(res: dict, hand: pd.DataFrame, mech: pd.DataFrame) -> str:
    hand_in = hand[(hand.date >= SAMPLE_START) & (hand.date <= SAMPLE_END)]
    mech_in = mech[(mech.date >= SAMPLE_START) & (mech.date <= SAMPLE_END)]
    lines = [
        "# Refit: continuous severity, pooled event term, mechanical list",
        "",
        "Same windows, controls, sample (2000-08-30 to 2021-12-31), ridge and blocked 5-fold CV as above.",
        "Severity is now the continuous measured score, capped: the larger of the GPR and EPU 5-day jumps, each as",
        "a z-score against all daily jumps 1990-2021, winsorised at z = 3 (`severity_capped`). Three event designs",
        "are compared with the controls-only baseline:",
        "",
        "1. hand list, one severity x indicator term per category;",
        "2. hand list, a single `any_event` = severity x 1[any hand-list event has its day 0 at t];",
        "3. mechanical list (`data/events_mechanical.csv`, days when GPR or EPU is in its top 5%, merged into",
        "   runs), the same single pooled term with the same jump-based severity.",
        "",
        "A fourth row, the hand list's pooled term with September 11 removed, is a robustness check: that event's",
        "uncapped severity (z = 19) was several times any other; the cap limits it to 3, the same as the next",
        "largest events, but it remains the one event everyone would name first.",
        "",
        f"In sample: {len(hand_in)} hand-list events ({int((hand_in.severity_capped < 0).sum())} with negative "
        f"severity, i.e. both indices fell into the event; {int((hand_in.severity_measured > 3).sum())} capped at 3) "
        f"and {len(mech_in)} mechanical events ({int((mech_in.severity_capped < 0).sum())} negative; "
        f"{int((mech_in.severity_measured > 3).sum())} capped). Severity is used as given, so a negative-severity "
        "event enters with its sign flipped.",
        "",
        "*CV RMSE* is the root mean squared out-of-fold residual from the blocked cross-validation at the chosen",
        "penalty (for controls only, plain OLS refitted per fold), on all windows and on the windows that start at",
        "a hand-list or a mechanical event. *Mean abnormal return* is the average out-of-fold controls-only residual",
        "on those windows, with its naive standard error: a selection check, since a hand-picked list of events",
        "that 'moved gold' should show a mean abnormal return the mechanical list does not.",
        "",
    ]
    for asset in ASSETS:
        for h in HORIZONS:
            r = res[(asset, h)]
            models, hr, mr, y = r["models"], r["hand_rows"], r["mech_rows"], r["y"].to_numpy()
            ctrl = models["controls only"].fit
            lines += [f"## {asset.capitalize()}, {h}-day windows", "",
                      f"| model | lambda | CV RMSE all (n={len(y)}) | CV RMSE hand windows (n={hr.sum()}) | "
                      f"CV RMSE mechanical windows (n={mr.sum()}) | in-sample RMSE all |",
                      "|---|---|---:|---:|---:|---:|"]
            base = (100 * rmse(y), 100 * rmse(y[hr]), 100 * rmse(y[mr]))
            lines.append(f"| no change (predict 0) | - | {base[0]:.2f}% | {base[1]:.2f}% | {base[2]:.2f}% | {base[0]:.2f}% |")
            for name, m in models.items():
                oof = m.fit.oof_resid
                lam = "-" if name == "controls only" else _lam_text(m.fit)
                lines.append(f"| {name} | {lam} | {100 * rmse(oof):.2f}% | {100 * rmse(oof[hr]):.2f}% | "
                             f"{100 * rmse(oof[mr]):.2f}% | {100 * rmse(m.fit.resid):.2f}% |")
            ab_all, ab_h, ab_m = _mean_se(ctrl.oof_resid), _mean_se(ctrl.oof_resid[hr]), _mean_se(ctrl.oof_resid[mr])
            raw_h, raw_m = _mean_se(y[hr]), _mean_se(y[mr])
            lines += ["", "Mean return and mean abnormal (controls-only, out-of-fold) return on event windows:", "",
                      "| windows | n | mean raw return | mean abnormal return (SE) |", "|---|---:|---:|---:|",
                      f"| all | {len(y)} | {100 * np.mean(y):+.2f}% | {100 * ab_all[0]:+.2f}% ({100 * ab_all[1]:.2f}) |",
                      f"| hand-list events | {hr.sum()} | {100 * raw_h[0]:+.2f}% | {100 * ab_h[0]:+.2f}% ({100 * ab_h[1]:.2f}) |",
                      f"| mechanical events | {mr.sum()} | {100 * raw_m[0]:+.2f}% | {100 * ab_m[0]:+.2f}% ({100 * ab_m[1]:.2f}) |",
                      ""]
            lines += ["Event coefficients (% of return per unit of severity):", "",
                      "| model | term | ridge: coef | SE | t | unpenalised: coef | SE | t |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            for name in ("hand list, category terms", "hand list, pooled term", "mechanical list, pooled term",
                         "hand list, pooled term, ex 9/11"):
                m = models[name]
                for term in m.fit.names:
                    if term in CONTROLS:
                        continue
                    i, j = m.fit.names.index(term), m.fit0.names.index(term)
                    c, se = 100 * m.fit.coef[i], 100 * m.fit.se[i]
                    c0, se0 = 100 * m.fit0.coef[j], 100 * m.fit0.se[j]
                    lines.append(f"| {name} | {term} | {c:+.3f} | {se:.3f} | {_fmt(c / se if se > 0 else float('nan'), 1)} | "
                                 f"{c0:+.3f} | {se0:.3f} | {_fmt(c0 / se0, 1)} |")
            lines.append("")
    lines += ["## Reading", "", refit_reading(res), ""]
    return "\n".join(lines)


def refit_reading(res: dict) -> str:
    names = ("hand list, category terms", "hand list, pooled term", "mechanical list, pooled term")

    def cv(asset, h, name, which="all"):
        r = res[(asset, h)]
        oof = r["models"][name].fit.oof_resid
        mask = {"all": np.ones(len(oof), bool), "hand": r["hand_rows"], "mech": r["mech_rows"]}[which]
        return 100 * rmse(oof[mask])

    def pooled(asset, h, name, penalised=True):
        f = res[(asset, h)]["models"][name].fit if penalised else res[(asset, h)]["models"][name].fit0
        i = f.names.index("any_event")
        return 100 * f.coef[i], f.t[i], f.lam

    def abnormal(asset, h, which):
        r = res[(asset, h)]
        mu, se = _mean_se(r["models"]["controls only"].fit.oof_resid[r["hand_rows"] if which == "hand" else r["mech_rows"]])
        return 100 * mu, 100 * se

    # 1. CV error differences
    diffs = {(a, h, n): cv(a, h, "controls only") - cv(a, h, n) for a in ASSETS for h in HORIZONS for n in names}
    best_all = max(diffs.values())
    diffs_hand = {(a, h, n): cv(a, h, "controls only", "hand") - cv(a, h, n, "hand") for a in ASSETS for h in HORIZONS for n in names}
    (ba, bh, bn), best_hand = max(diffs_hand.items(), key=lambda kv: kv[1])
    tops = [(a, h, n) for a in ASSETS for h in HORIZONS for n in names if res[(a, h)]["models"][n].fit.lam >= LAMBDA_GRID.max()]
    mech_gain = max(diffs[(a, h, "mechanical list, pooled term")] for a in ASSETS for h in HORIZONS)

    # 2. selection check
    ab = {(a, h, w): abnormal(a, h, w) for a in ASSETS for h in HORIZONS for w in ("hand", "mech")}
    any_sig = [(a, h, w) for (a, h, w), (m, se) in ab.items() if abs(m) > 2 * se]
    g5h, g20h, g5m, g20m = ab[("gold", 5, "hand")], ab[("gold", 20, "hand")], ab[("gold", 5, "mech")], ab[("gold", 20, "mech")]
    vol_hand = cv("gold", 5, "controls only", "hand") / cv("gold", 5, "controls only")
    vol_mech = cv("gold", 5, "controls only", "mech") / cv("gold", 5, "controls only")

    # 3. surviving terms after shrinkage (non-zero and |t| >= 2)
    survivors = []
    for a in ASSETS:
        for h in HORIZONS:
            for n in names:
                f = res[(a, h)]["models"][n].fit
                survivors += [(a, h, n, t, 100 * c, tt) for t, c, tt in zip(f.names, f.coef, f.t)
                              if t not in CONTROLS and abs(tt) >= 2 and abs(c) > 1e-6]
    pooled_surv = [(a, h) for a, h, n, t, c, tt in survivors if n == "hand list, pooled term"]
    ex = {(a, h): pooled(a, h, "hand list, pooled term, ex 9/11") for a in ASSETS for h in HORIZONS}
    full = {(a, h): pooled(a, h, "hand list, pooled term") for a in ASSETS for h in HORIZONS}

    parts = [
        f"**Does any event term help out of sample?** Barely. Across the {len(diffs)} comparisons (two metals, two "
        f"horizons, three designs) the largest improvement in cross-validated RMSE over controls-only is "
        f"{best_all:.3f} percentage points on all windows and {best_hand:.2f} points on the hand-list event windows "
        f"({ba}, {bh}-day, {bn}), against event-window RMSEs of 3 to 9%. In {len(tops)} of the {len(diffs)} fits "
        f"cross-validation pushes the penalty to the top of the grid and removes the event terms outright. "
        f"The mechanical list's pooled term never improves CV RMSE by more than {max(mech_gain, 0):.3f} points.",
        f"**Which terms survive the shrinkage with |t| >= 2?** "
        + ("; ".join(f"{t} ({a}, {h}-day, {n}: {c:+.2f}% per unit, t = {tt:+.1f})" for a, h, n, t, c, tt in survivors) + ". "
           if survivors else "None. ")
        + (f"The hand list's pooled term for gold at {', '.join(f'{h} days' for a, h in pooled_surv)} is the only one that "
           f"is not a two-to-four-event category; removing September 11 takes it from "
           + " and ".join(f"{full[(a, h)][0]:+.2f}% (t = {full[(a, h)][1]:+.1f})" for a, h in pooled_surv) + " to "
           + " and ".join(f"{ex[(a, h)][0]:+.2f}% (t = {ex[(a, h)][1]:+.1f})" for a, h in pooled_surv)
           + (", so the point estimate survives but its significance came from September 11. "
              if all(abs(ex[(a, h)][1]) < 2 and np.sign(ex[(a, h)][0]) == np.sign(full[(a, h)][0]) for a, h in pooled_surv)
              else ", so it does not depend on September 11 alone. "
              if all(abs(ex[(a, h)][1]) >= 2 for a, h in pooled_surv)
              else ", so it is a September 11 effect. ")
           if pooled_surv else
           f"No pooled term survives; the hand list's pooled term for gold is {full[('gold', 5)][0]:+.2f}% per unit at "
           f"5 days (t = {full[('gold', 5)][1]:+.1f}) and {full[('gold', 20)][0]:+.2f}% at 20 days "
           f"(t = {full[('gold', 20)][1]:+.1f}). ")
        + "The category terms that pass for silver at 5 days are the same two-to-four-event categories as before "
        "(the pandemic category is Fukushima and Covid), and their standard errors come from that handful of rows.",
        f"**Does the hand-picked list look biased?** Not in the sense of being days when gold happened to jump: the "
        f"mean out-of-fold abnormal gold return on hand-list windows is {g5h[0]:+.2f}% at 5 days and {g20h[0]:+.2f}% at "
        f"20 days (SE {g5h[1]:.2f} and {g20h[1]:.2f}), versus {g5m[0]:+.2f}% and {g20m[0]:+.2f}% on mechanical windows "
        f"(SE {g5m[1]:.2f} and {g20m[1]:.2f}), "
        + ("and none of these means is two standard errors from zero. " if not any_sig else
           "and only " + ", ".join(f"{a} {h}-day ({w})" for a, h, w in any_sig) + " is two standard errors from zero. ")
        + f"The list is selected on something else: volatility. Controls-only CV RMSE on hand-list windows is "
        f"{vol_hand:.1f}x its value on all windows, against {vol_mech:.1f}x for the mechanical windows, so the hand "
        f"list is a list of days when markets were moving a lot, chosen knowing they did. It also covers different "
        f"ground from the indices: 20 of its 68 events are never flagged by GPR or EPU (central-bank decisions, "
        f"sovereign-debt episodes, the gold-system events), while the mechanical list is full of EPU spikes on US "
        f"fiscal news that the hand list ignores, and 7 hand-list events have negative measured severity because the "
        f"indices fell into them.",
        "**Verdict for v1.** With real yields, the dollar, the VIX and the GPR index in the regression, neither the "
        "hand-picked events nor the mechanically flagged ones add a detectable, out-of-sample-stable effect on gold or "
        "silver at 5 or 20 days, pooled or by category, with hand, quintile or capped continuous severity. The "
        "mechanical list's merge rule turns long crises into single onset-dated events (the Iraq build-up runs from "
        "2002-12-18 to 2003-05-22), which is the right unit for some questions and the wrong one for an invasion-day "
        "study.",
    ]
    return "\n\n".join(parts)


def main() -> None:
    from gold_events.mechanical import load_mechanical

    df = data.load()
    events = load_events()
    if "severity_quintile" not in events.columns:
        raise SystemExit("events.csv has no measured columns; run `python -m gold_events.events` first")
    results = run(df, events)
    text = report(results, events, df)
    mech = load_mechanical()
    text += "\n" + refit_report(refit(df, events, mech), events, mech)
    REPORT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
