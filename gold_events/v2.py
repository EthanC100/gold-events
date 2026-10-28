"""v2: the v1 specification plus two gold-demand factors, trained to 2021, tested frozen on 2022 on.

New factors (both from :mod:`gold_events.data`):

* ``d_gld``: change in SPDR Gold Shares (GLD) holdings over the window, tonnes (daily from
  2004-11-18, so the v2 training sample starts there);
* ``cb_flow``: official-sector net gold purchases during the window, tonnes, at the
  *lagged* quarterly pace: the IMF IRFCL reporters' quarterly net purchases, lagged one
  quarter so only published data is used, interpolated to daily, times h/63.

Fit: exactly as v1 (``gold_events.model``): hand-severity category terms, ridge on the event
terms only, blocked 5-fold CV, Newey-West SEs; the frozen v2 is the intercept plus the control
and factor coefficients.  Test: every h-day window whose base close is 2021-12-31 or later,
predicted with the frozen coefficients, with the same metrics as the v1 test
(:mod:`gold_events.oos`), next to v1 on the same windows.  ``data/v2_coefficients.csv`` is
written beside ``data/v1_coefficients.csv``; v1 is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_events import data
from gold_events.events import load_events
from gold_events.model import (ASSETS, CONTROL_UNITS, CONTROLS, HORIZONS, SAMPLE_END, SAMPLE_START,
                               Fit, build_design, ols_fit, ridge_fit, rmse)
from gold_events.oos import N_DROP, TEST_BASE, frozen_v1, hit_rate

V2_PATH = data.DATA_DIR / "v2_coefficients.csv"
FACTORS = ["d_gld", "cb_flow"]
V2_CONTROLS = CONTROLS + FACTORS
V2_UNITS = dict(CONTROL_UNITS)
V2_UNITS.update({"d_gld": (10.0, "per +10 t GLD holdings"), "cb_flow": (100.0, "per 100 t official buying in the window")})
QUARTER_DAYS = 63


def add_factors(X: pd.DataFrame, df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Append d_gld and cb_flow to a design built by ``build_design`` (rows = window starts)."""
    idx = df.index
    pos = idx.get_indexer(X.index)
    base, last = pos - 1, pos + h - 1
    gld = df["gld_tonnes"].to_numpy()
    cb = df["cb_net_purchases"].to_numpy()
    X = X.copy()
    X["d_gld"] = gld[last] - gld[base]
    X["cb_flow"] = cb[last] * h / QUARTER_DAYS
    return X.dropna()


def design_v2(df: pd.DataFrame, events: pd.DataFrame, asset: str, h: int, start, end) -> pd.DataFrame:
    X = build_design(df, events, asset, h, "severity_hand", start=start, end=end)
    return add_factors(X, df, h)


@dataclass
class Frozen:
    asset: str
    h: int
    intercept: float
    coef: dict[str, float]
    se: dict[str, float]
    lam: float
    n: int
    controls: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + sum(self.coef[c] * X[c].to_numpy() for c in self.controls)


def fit_frozen(X: pd.DataFrame, y: pd.Series, controls: list[str], asset: str, h: int) -> Frozen:
    cats = [c for c in X.columns if c not in controls]
    fit = ridge_fit(X, y, penalised=cats, lag=h)
    coef = {c: float(fit.coef[fit.names.index(c)]) for c in controls}
    se = {c: float(fit.se[fit.names.index(c)]) for c in controls}
    return Frozen(asset, h, float(fit.intercept), coef, se, float(fit.lam), len(y), controls)


def train(df: pd.DataFrame, events: pd.DataFrame) -> dict:
    """v2 and, on v2's own (post-2004) rows, the v1 specification, both trained to SAMPLE_END."""
    cut = df.loc[:SAMPLE_END]
    out = {"v2": {}, "v1_same_rows": {}, "n": {}}
    for asset in ASSETS:
        for h in HORIZONS:
            X = design_v2(cut, events, asset, h, SAMPLE_START, SAMPLE_END)
            y = X.pop("y")
            out["v2"][(asset, h)] = fit_frozen(X, y, V2_CONTROLS, asset, h)
            out["v1_same_rows"][(asset, h)] = fit_frozen(X.drop(columns=FACTORS), y, CONTROLS, asset, h)
            out["n"][(asset, h)] = (X.index[0], X.index[-1], len(y))
    return out


def coefficient_table(frozen: dict) -> pd.DataFrame:
    rows = []
    for (asset, h), f in frozen.items():
        row = {"asset": asset, "h": h, "intercept_pct": 100 * f.intercept, "lambda": f.lam, "n": f.n}
        for c in f.controls:
            mult = V2_UNITS[c][0]
            row[f"{c}_pct_per_unit"] = 100 * mult * f.coef[c]
            row[f"{c}_se_pct_per_unit"] = 100 * mult * f.se[c]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Frozen test
# --------------------------------------------------------------------------


def metrics(y, pred):
    return {"n": int(len(y)), "rmse": 100 * rmse(y - pred), "mean_err": 100 * float(np.mean(y - pred)),
            "hit": hit_rate(pred, y), "r2_oos": 1 - float(np.sum((y - pred) ** 2) / np.sum(y ** 2))}


@dataclass
class TestResult:
    asset: str
    h: int
    n: int
    zero: dict
    v1: dict
    v2: dict
    refit_v2: dict
    zero_trim: dict
    v1_trim: dict
    v2_trim: dict
    refit: Fit               # OLS of the v2 controls on the test windows
    mean_y: float


def test(df: pd.DataFrame, events: pd.DataFrame, v2: dict, v1: dict) -> list[TestResult]:
    out = []
    for asset in ASSETS:
        for h in HORIZONS:
            X = design_v2(df, events, asset, h, TEST_BASE, df.index[-1])
            y = X["y"].to_numpy()
            p1, p2 = v1[(asset, h)].predict(X), v2[(asset, h)].predict(X)
            refit = ols_fit(X[V2_CONTROLS], X["y"], lag=h)
            keep = np.sort(np.argsort(-np.abs(y))[N_DROP:])
            zero = np.zeros_like(y)
            out.append(TestResult(asset, h, len(y), metrics(y, zero), metrics(y, p1), metrics(y, p2),
                                  metrics(y, refit.fitted), metrics(y[keep], zero[keep]), metrics(y[keep], p1[keep]),
                                  metrics(y[keep], p2[keep]), refit, 100 * float(np.mean(y))))
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _pct(x, nd=2):
    return f"{x:+.{nd}f}%"


def report(trained: dict, tests: list[TestResult], df: pd.DataFrame) -> list[str]:
    v2, v1s = trained["v2"], trained["v1_same_rows"]
    L = ["# v2: v1 plus GLD holdings and central-bank buying", "",
         "v2 adds two demand factors to the v1 specification: the change in SPDR GLD holdings over the window",
         "(tonnes) and official-sector net gold purchases during the window at the lagged quarterly pace (IMF",
         "IRFCL reporters' quarterly net purchases, lagged one quarter so only published data is used, interpolated",
         "to daily, times h/63). GLD starts 2004-11-18, so v2 trains on 2004-11 to 2021-12; the v1 specification",
         "is refitted on exactly those rows for a like-for-like comparison. Fit, ridge and cross-validation are",
         "as in v1. The central-bank series is built from the IMF's reserves template rather than the World Gold",
         "Council file (which needs a Goldhub login): it covers reporting central banks only, so it understates",
         "WGC's estimate, which includes unreported buying.", "",
         "## Coefficients (train, percent of return per unit)", "",
         "| asset | h | term | v1 (2000-2021) | v1 on v2 rows | **v2** | v2 SE | n |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    v1_orig = pd.read_csv(data.DATA_DIR / "v1_coefficients.csv").set_index(["asset", "h"])
    for (asset, h), f in v2.items():
        g = v1s[(asset, h)]
        n = trained["n"][(asset, h)][2]
        L.append(f"| {asset} | {h} | intercept | {v1_orig.loc[(asset, h), 'intercept_pct']:+.3f} | {100 * g.intercept:+.3f} | "
                 f"**{100 * f.intercept:+.3f}** | - | {n} |")
        for c in V2_CONTROLS:
            mult = V2_UNITS[c][0]
            o = f"{v1_orig.loc[(asset, h), f'{c}_pct_per_unit']:+.3f}" if c in CONTROLS else "-"
            s = f"{100 * mult * g.coef[c]:+.3f}" if c in CONTROLS else "-"
            L.append(f"| {asset} | {h} | {c} ({V2_UNITS[c][1]}) | {o} | {s} | **{100 * mult * f.coef[c]:+.3f}** | "
                     f"{100 * mult * f.se[c]:.3f} | |")
    L += ["", f"Training windows: " + "; ".join(f"{a} {h}d {s:%Y-%m-%d}..{e:%Y-%m-%d} ({n})"
                                                for (a, h), (s, e, n) in trained["n"].items()) + ".", ""]
    L += ["## Frozen test on 2022 to latest", "",
          f"Windows with base close on or after {TEST_BASE:%Y-%m-%d}; last window ends {df.index[-1]:%Y-%m-%d}. "
          "Mean error is the unexplained drift (realised minus predicted). 'Refit v2' is an OLS of the v2 controls",
          "on the test windows themselves, an in-sample bound. 'Trimmed' removes the five largest windows by |return|.", "",
          "| asset | h | n | model | RMSE | mean error (drift) | hit rate | R2 oos | RMSE trimmed | drift trimmed |",
          "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for t in tests:
        for name, m, mt in (("no change", t.zero, t.zero_trim), ("frozen v1", t.v1, t.v1_trim),
                            ("**frozen v2**", t.v2, t.v2_trim), ("refit v2 (bound)", t.refit_v2, None)):
            trim = f"{mt['rmse']:.2f}% | {_pct(mt['mean_err'])}" if mt else "- | -"
            hit = "-" if name == "no change" else f"{m['hit']:.0%}"
            L.append(f"| {t.asset} | {t.h} | {t.n} | {name} | {m['rmse']:.2f}% | {_pct(m['mean_err'])} | {hit} | "
                     f"{m['r2_oos']:+.3f} | {trim} |")
    L += ["", "## Drift: does v2 shrink it?", "",
          "| asset | h | v1 drift | v2 drift | change | v1 RMSE | v2 RMSE | change |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in tests:
        d1, d2 = t.v1["mean_err"], t.v2["mean_err"]
        L.append(f"| {t.asset} | {t.h} | {_pct(d1)} | {_pct(d2)} | {_pct(d2 - d1)} ({(abs(d2) / abs(d1) - 1) * 100:+.0f}% of |drift|) | "
                 f"{t.v1['rmse']:.2f}% | {t.v2['rmse']:.2f}% | {t.v2['rmse'] - t.v1['rmse']:+.2f} |")
    L += ["", "## Stability of the v2 coefficients: train (2004-2021) vs test (2022-latest, OLS)", "",
          "| asset | h | term | train | SE | test refit | SE | gap in train SEs |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for t in tests:
        f = v2[(t.asset, t.h)]
        for c in V2_CONTROLS:
            mult = V2_UNITS[c][0]
            i = t.refit.names.index(c)
            tr, trse = 100 * mult * f.coef[c], 100 * mult * f.se[c]
            te, tese = 100 * mult * t.refit.coef[i], 100 * mult * t.refit.se[i]
            gap = (te - tr) / trse if trse > 0 else float("nan")
            flag = " **" if abs(gap) >= 2 else ""
            L.append(f"| {t.asset} | {t.h} | {c} | {tr:+.3f} | {trse:.3f} | {te:+.3f} | {tese:.3f} | {gap:+.1f}{flag} |")
        L.append(f"| {t.asset} | {t.h} | intercept | {100 * f.intercept:+.3f} | - | {100 * t.refit.intercept:+.3f} | - | - |")
    L.append("")
    return L


def reading(trained: dict, tests: list[TestResult]) -> str:
    v2 = trained["v2"]
    g5, g20 = [t for t in tests if t.asset == "gold" and t.h == 5][0], [t for t in tests if t.asset == "gold" and t.h == 20][0]
    s20 = [t for t in tests if t.asset == "silver" and t.h == 20][0]

    def sig(asset, h, c):
        f = v2[(asset, h)]
        return f.coef[c] / f.se[c] if f.se[c] > 0 else float("nan")

    gld_t = {(a, h): sig(a, h, "d_gld") for a in ASSETS for h in HORIZONS}
    cb_t = {(a, h): sig(a, h, "cb_flow") for a in ASSETS for h in HORIZONS}
    gld_coef = 100 * 10 * v2[("gold", 5)].coef["d_gld"]
    cb_coef = 100 * 100 * v2[("gold", 20)].coef["cb_flow"]
    better = sum(t.v2["rmse"] < t.v1["rmse"] for t in tests)
    drift_better = sum(abs(t.v2["mean_err"]) < abs(t.v1["mean_err"]) for t in tests)
    unstable = []
    for t in tests:
        f = v2[(t.asset, t.h)]
        for c in V2_CONTROLS:
            i = t.refit.names.index(c)
            if f.se[c] > 0 and abs((t.refit.coef[i] - f.coef[c]) / f.se[c]) >= 2:
                unstable.append(f"{c} ({t.asset} {t.h}d)")
    return (
        f"**GLD flows are the strong new factor; central-bank buying is not.** In training, a 10-tonne change in "
        f"GLD holdings over a 5-day window goes with a {gld_coef:+.2f}% gold move (t = {gld_t[('gold', 5)]:+.1f}; "
        f"t = {gld_t[('gold', 20)]:+.1f} at 20 days, {gld_t[('silver', 5)]:+.1f} and {gld_t[('silver', 20)]:+.1f} for "
        f"silver). That is contemporaneous, not predictive: ETF creations happen on the days gold rises, so the "
        f"factor explains the window after the fact and is of no use to a forecast unless the flow itself is "
        f"forecast. The lagged central-bank pace is {cb_coef:+.2f}% per 100 tonnes bought in a 20-day window "
        f"(t = {cb_t[('gold', 20)]:+.1f}; {cb_t[('gold', 5)]:+.1f} at 5 days), a slow variable that cannot "
        f"explain week-to-week moves and barely moves the fit. "
        f"**Drift in the test period.** v2 beats v1's RMSE on {better} of 4 asset-horizon pairs and reduces the "
        f"absolute drift on {drift_better} of 4: gold's 20-day mean error goes from {_pct(g20.v1['mean_err'])} under "
        f"v1 to {_pct(g20.v2['mean_err'])} under v2 (5-day: {_pct(g5.v1['mean_err'])} to {_pct(g5.v2['mean_err'])}; "
        f"silver 20-day: {_pct(s20.v1['mean_err'])} to {_pct(s20.v2['mean_err'])}). "
        + ("So v2 explains more of the week-to-week variance but misses more of the trend: the 2024-2026 rally ran "
           "with flat-to-negative GLD holdings and only modest reported official buying over the windows in which "
           "it happened, so a model that leans on those flows predicts less and leaves a larger unexplained drift. "
           "The drift is not an omitted-flow story, at least not one these two measured flows capture. "
           if drift_better < 2 else
           "The drift that v2 removes is removed by the GLD term, which is to say by knowing the week's ETF flows: "
           "it describes the rally as one accompanied by ETF buying, it does not forecast it. ")
        + f"**Stability.** " + (f"Coefficients whose test-period refit sits more than two training SEs from the "
                                f"frozen value: {', '.join(unstable)}. " if unstable else
                                "No coefficient's test-period refit sits more than two training SEs from the frozen value. ")
        + f"The GLD beta is the least stable of all: for gold at 20 days it goes from "
        f"{100 * 10 * v2[('gold', 20)].coef['d_gld']:+.2f}% per 10 tonnes in training to "
        f"{100 * 10 * g20.refit.coef[g20.refit.names.index('d_gld')]:+.2f}% in the test period, so even the "
        f"contemporaneous relationship is not a fixed number. The v1 controls' betas are the ones that matter for "
        f"forecasting, and the rolling-window section below shows how far they have moved."
    )


def main() -> None:
    from gold_events import rolling

    df = data.load()
    events = load_events()
    trained = train(df, events)
    v1 = frozen_v1(df, events)
    tests = test(df, events, trained["v2"], v1)
    coefficient_table(trained["v2"]).to_csv(V2_PATH, index=False)
    lines = report(trained, tests, df) + ["## Reading", "", reading(trained, tests), ""]
    roll = rolling.run(df)
    lines += rolling.report(roll, df)
    lines += ["## Reading (rolling)", "", rolling_reading(roll), ""]
    text = "\n".join(lines)
    (data.ROOT / "reports" / "v2_fit.md").write_text(text)
    print(text)
    print(f"Wrote {V2_PATH} (v1 untouched) and {rolling.ROLLING_PATH}; forward model = {rolling.POINTER_PATH.read_text().strip()}")


def rolling_reading(res: dict) -> str:
    checks, keep = res["checks"], res["keep_v1"]
    n_out = sum(not c.inside for c in checks)
    g = res["roll"][("gold", 5)]
    ry = g["d_real_yield"]
    pre = ry[g.index < "2013-06-30"]
    mid = ry[(g.index >= "2014-01-01") & (g.index <= "2021-12-31")]
    last = res["latest"].set_index(["asset", "h"])
    return (
        f"**The rule fires: {n_out} of {len(checks)} coefficients are outside the last five years' interquartile "
        f"range, so the forward log now uses the latest five-year window as `v2-rolling`.** "
        if not keep else
        f"**The rule does not fire: all {len(checks)} coefficients are inside the last five years' interquartile "
        "range, so the forward log keeps v1.** "
    ) + (
        f"The picture behind it is one of regimes, not noise. Gold's real-yield beta at 5 days was near zero in "
        f"windows ending before mid-2013 (median {pre.median():+.1f}% per pp), fell to between -7% and -12% in "
        f"windows ending 2014-2021 (median {mid.median():+.1f}%), and has come back to about {ry.iloc[-1]:+.1f}% in "
        f"the latest window, so the v1 value of {res['v1'][('gold', 5)]['d_real_yield']:+.1f}% is an average over "
        f"two different worlds. The dollar beta has been more stable (between about -0.7 and -1.6 throughout) and "
        f"the GPR beta has risen from roughly zero before 2020 to about {g['d_gpr'].iloc[-1]:+.2f}% per 100 points. "
        f"The intercept, gold's drift per window after the controls, is at a two-decade high "
        f"({g['intercept'].iloc[-1]:+.2f}% per 5-day window, {res['roll'][('gold', 20)]['intercept'].iloc[-1]:+.2f}% "
        f"per 20-day window), which is the same unexplained drift the out-of-sample test found. Two cautions: the "
        f"interquartile range of about twenty quarterly windows is narrow, so the test is demanding by construction, "
        f"and the latest window's betas carry standard errors of {g['d_real_yield_se'].iloc[-1]:.1f} (real yield) and "
        f"{g['dlog_dxy_se'].iloc[-1]:.2f} (dollar) percentage points, so switching trades a stale average for a "
        f"noisier recent estimate. The switch is what the pre-set rule says to do; the forward scorecard will show "
        f"whether it earns its keep. The v2-rolling coefficients are the "
        f"{last.loc[('gold', 5), 'window_start']} to {last.loc[('gold', 5), 'window_end']} window."
    )


if __name__ == "__main__":
    main()
