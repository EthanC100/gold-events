"""Rolling five-year refits of the v1 controls-only specification, and the switching rule.

Windows are five calendar years of window starts, stepped at every quarter end from 2000-08-30
plus five years to the latest date (the last window ends at the latest trading day).  Each
window is a plain OLS of the h-day log return on the four v1 controls with Newey-West standard
errors, exactly the frozen v1 specification without event terms.

Pre-set rule (applied at h = 5, the forward log's horizon, to both metals, intercept
included): if every full-sample v1 coefficient (``data/v1_coefficients.csv``, fitted on
2000-2021) lies inside the interquartile range of the rolling estimates whose window ends in the
last five years, the forward log keeps v1.  Otherwise it switches to the most recent five-year
window's coefficients, saved as ``data/v2_rolling_coefficients.csv`` and versioned
``v2-rolling``.  The decision is written to ``data/forward_model.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_events import data
from gold_events.model import ASSETS, CONTROL_UNITS, CONTROLS, HORIZONS, SAMPLE_START, build_design, ols_fit

FIG_DIR = data.ROOT / "reports" / "figures"
V1_PATH = data.DATA_DIR / "v1_coefficients.csv"
ROLLING_PATH = data.DATA_DIR / "v2_rolling_coefficients.csv"
POINTER_PATH = data.DATA_DIR / "forward_model.txt"
WINDOW_YEARS = 5
RULE_H = 5
TERMS = ["intercept"] + CONTROLS
EMPTY_EVENTS = pd.DataFrame({"date": pd.to_datetime([]), "category": pd.Series([], dtype=str),
                             "severity_hand": pd.Series([], dtype=float)})


def _pct_units(term: str) -> float:
    return 1.0 if term == "intercept" else CONTROL_UNITS[term][0]


def window_ends(index: pd.DatetimeIndex, start: pd.Timestamp = SAMPLE_START) -> list[pd.Timestamp]:
    first = start + pd.DateOffset(years=WINDOW_YEARS)
    ends = [q for q in pd.date_range(first, index[-1], freq="QE") if q >= first]
    ends = [index[index.searchsorted(q, side="right") - 1] for q in ends]  # last trading day on or before
    if not ends or ends[-1] != index[-1]:
        ends.append(index[-1])
    return sorted(set(ends))


def rolling_fits(df: pd.DataFrame, asset: str, h: int) -> pd.DataFrame:
    """One row per window: end date, n, each term's coefficient and SE in percent-per-unit units."""
    rows = []
    for end in window_ends(df.index):
        start = end - pd.DateOffset(years=WINDOW_YEARS)
        X = build_design(df, EMPTY_EVENTS, asset, h, "severity_hand", start=start, end=end)
        y = X.pop("y")
        fit = ols_fit(X[CONTROLS], y, lag=h)
        row = {"window_end": end, "window_start": X.index[0], "n": len(y),
               "intercept": 100 * fit.intercept, "intercept_se": np.nan}
        for c in CONTROLS:
            i = fit.names.index(c)
            row[c] = 100 * CONTROL_UNITS[c][0] * fit.coef[i]
            row[f"{c}_se"] = 100 * CONTROL_UNITS[c][0] * fit.se[i]
        rows.append(row)
    return pd.DataFrame(rows).set_index("window_end")


def full_sample_fit(df: pd.DataFrame, asset: str, h: int) -> dict[str, float]:
    """OLS on everything from SAMPLE_START to the latest date, for reference (not used by the rule)."""
    X = build_design(df, EMPTY_EVENTS, asset, h, "severity_hand", start=SAMPLE_START, end=df.index[-1])
    y = X.pop("y")
    fit = ols_fit(X[CONTROLS], y, lag=h)
    out = {"intercept": 100 * fit.intercept}
    out.update({c: 100 * CONTROL_UNITS[c][0] * fit.coef[fit.names.index(c)] for c in CONTROLS})
    return out


def load_v1(path=V1_PATH) -> dict[tuple[str, int], dict[str, float]]:
    t = pd.read_csv(path)
    out = {}
    for r in t.itertuples():
        d = {"intercept": float(r.intercept_pct)}
        d.update({c: float(getattr(r, f"{c}_pct_per_unit")) for c in CONTROLS})
        out[(r.asset, int(r.h))] = d
    return out


@dataclass
class RuleCheck:
    asset: str
    h: int
    term: str
    v1: float
    q25: float
    q75: float
    latest: float
    latest_se: float
    inside: bool


def apply_rule(roll: dict[tuple[str, int], pd.DataFrame], v1: dict, h: int = RULE_H) -> tuple[list[RuleCheck], bool]:
    checks = []
    for asset in ASSETS:
        r = roll[(asset, h)]
        recent = r[r.index >= r.index[-1] - pd.DateOffset(years=5)]
        last = r.iloc[-1]
        for term in TERMS:
            q25, q75 = float(recent[term].quantile(0.25)), float(recent[term].quantile(0.75))
            val = v1[(asset, h)][term]
            checks.append(RuleCheck(asset, h, term, val, q25, q75, float(last[term]),
                                    float(last.get(f"{term}_se", np.nan)), q25 <= val <= q75))
    return checks, all(c.inside for c in checks)


def latest_window_table(roll: dict) -> pd.DataFrame:
    """Most recent window's coefficients in the v1_coefficients.csv format."""
    rows = []
    for (asset, h), r in roll.items():
        last = r.iloc[-1]
        row = {"asset": asset, "h": h, "intercept_pct": last["intercept"], "lambda": 0.0}
        row.update({f"{c}_pct_per_unit": last[c] for c in CONTROLS})
        row["window_start"] = last["window_start"].strftime("%Y-%m-%d")
        row["window_end"] = r.index[-1].strftime("%Y-%m-%d")
        rows.append(row)
    return pd.DataFrame(rows)


def plot(roll: pd.DataFrame, v1: dict[str, float], asset: str, h: int, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(TERMS), 1, figsize=(9, 2.2 * len(TERMS)), sharex=True)
    labels = {"intercept": "intercept (% per window)"}
    labels.update({c: f"{c} (% {CONTROL_UNITS[c][1]})" for c in CONTROLS})
    for ax, term in zip(axes, TERMS):
        x = roll.index
        ax.plot(x, roll[term], color="#1f4e79", lw=1.5, label="rolling 5y OLS")
        if f"{term}_se" in roll and roll[f"{term}_se"].notna().any():
            ax.fill_between(x, roll[term] - roll[f"{term}_se"], roll[term] + roll[f"{term}_se"],
                            color="#1f4e79", alpha=0.18, lw=0, label="+/- 1 SE")
        ax.axhline(v1[term], color="#c0504d", ls="--", lw=1.2, label="v1 (2000-2021)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_ylabel(labels[term], fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)
    axes[0].set_title(f"{asset}: rolling five-year coefficients, {h}-day windows (window end date)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(df: pd.DataFrame) -> dict:
    v1 = load_v1()
    roll, full = {}, {}
    for asset in ASSETS:
        for h in HORIZONS:
            roll[(asset, h)] = rolling_fits(df, asset, h)
            full[(asset, h)] = full_sample_fit(df, asset, h)
            plot(roll[(asset, h)], v1[(asset, h)], asset, h, FIG_DIR / f"rolling_{asset}_h{h}.png")
    checks, keep_v1 = apply_rule(roll, v1)
    latest = latest_window_table(roll)
    latest.to_csv(ROLLING_PATH, index=False)
    POINTER_PATH.write_text(("v1" if keep_v1 else "v2-rolling") + "\n")
    return {"v1": v1, "roll": roll, "full": full, "checks": checks, "keep_v1": keep_v1, "latest": latest}


def report(res: dict, df: pd.DataFrame) -> list[str]:
    checks, keep = res["checks"], res["keep_v1"]
    L = ["# Rolling coefficients and the switching rule", "",
         f"The v1 controls-only specification (h-day log return on changes in the real yield, log DXY, log VIX "
         f"and GPR) refitted by OLS on five-year windows stepped quarterly, window ends from "
         f"{res['roll'][('gold', 5)].index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} "
         f"({len(res['roll'][('gold', 5)])} windows). Figures: `reports/figures/rolling_<asset>_h<h>.png`, "
         "one panel per coefficient with a one-standard-error band and the v1 value as a dashed line.", "",
         "## Rule check (h = 5, the forward log's horizon)", "",
         "A coefficient passes when the v1 (2000-2021) value lies inside the interquartile range of the rolling "
         "estimates whose window ends in the last five years.", "",
         "| asset | term | v1 value | rolling IQR, last 5y | latest window | inside? |", "|---|---|---:|---:|---:|:---:|"]
    for c in checks:
        se = f" (SE {c.latest_se:.2f})" if not np.isnan(c.latest_se) else ""
        L.append(f"| {c.asset} | {c.term} | {c.v1:+.3f} | [{c.q25:+.3f}, {c.q75:+.3f}] | {c.latest:+.3f}{se} | "
                 f"{'yes' if c.inside else '**no**'} |")
    n_out = sum(not c.inside for c in checks)
    L += ["", f"**Decision:** {n_out} of {len(checks)} coefficients fall outside the recent interquartile range, so "
          + ("the forward log keeps the full-sample v1 coefficients." if keep else
             "the forward log switches to the most recent five-year window's coefficients, versioned `v2-rolling` "
             "(`data/v2_rolling_coefficients.csv`; `data/forward_model.txt` now reads `v2-rolling`)."), ""]
    L += ["## Coefficients: v1, latest five-year window, full 2000-2026 sample", "",
          "| asset | h | term | v1 (2000-2021) | latest 5y window | full sample to latest |", "|---|---:|---|---:|---:|---:|"]
    for (asset, h), r in res["roll"].items():
        last = r.iloc[-1]
        for term in TERMS:
            L.append(f"| {asset} | {h} | {term} | {res['v1'][(asset, h)][term]:+.3f} | {last[term]:+.3f} | "
                     f"{res['full'][(asset, h)][term]:+.3f} |")
    L += ["", "## Range of the rolling estimates", "",
          "| asset | h | term | min | 25% | median | 75% | max | windows where |t| >= 2 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for (asset, h), r in res["roll"].items():
        for term in TERMS:
            q = r[term].quantile([0, 0.25, 0.5, 0.75, 1])
            if term == "intercept":
                sig = "-"
            else:
                sig = f"{int((r[term].abs() / r[f'{term}_se'] >= 2).sum())} of {len(r)}"
            L.append(f"| {asset} | {h} | {term} | {q[0]:+.3f} | {q[0.25]:+.3f} | {q[0.5]:+.3f} | {q[0.75]:+.3f} | {q[1]:+.3f} | {sig} |")
    L.append("")
    return L


def main() -> None:
    df = data.load()
    res = run(df)
    print("\n".join(report(res, df)))


if __name__ == "__main__":
    main()
