"""Weekly forward log: a Monday routine for out-of-sample tracking.

``log_week.py`` (``log_week()``) appends one row to ``data/forward_log.csv``:

* the latest daily GPR value and its change over the past week;
* the current probability of each question in ``data/tension_questions.yaml``
  (Polymarket or Metaculus);
* the frozen v1 model's expected 5-day gold and silver move, computed from
  the *last* week's control changes (``data/v1_coefficients.csv``, the v1
  control coefficients fitted to 2021-12-31 and written by
  :mod:`gold_events.oos`; the event terms were shrunk to zero).  This is a placeholder for a real
  forecast of the controls: it assumes the coming week's changes in the real
  yield, the dollar, the VIX and GPR equal last week's;
* blank columns for your own tension score (0-10) and your own probability
  on each question, to be filled in by hand.

``score_week.py`` (``score_week()``) fills in the realised 5-day gold and
silver moves for rows whose week has passed, then writes
``reports/forward_scorecard.md`` with the running direction hit rate, mean
error, mean absolute error (against a no-change baseline) and the Brier
score of your probabilities versus the market's on questions marked
resolved in the YAML.

Realised move for a row logged on date D: log return from the last close on
or before D to the close five trading days later.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from gold_events import data

log = logging.getLogger(__name__)

QUESTIONS_PATH = data.DATA_DIR / "tension_questions.yaml"
LOG_PATH = data.DATA_DIR / "forward_log.csv"
FROZEN_PATHS = {"v1": data.DATA_DIR / "v1_coefficients.csv",
                "v2-rolling": data.DATA_DIR / "v2_rolling_coefficients.csv"}
POINTER_PATH = data.DATA_DIR / "forward_model.txt"  # written by gold_events.rolling per the switching rule
FROZEN_PATH = FROZEN_PATHS["v1"]
UNIT_MULT = {"d_real_yield": 1.0, "dlog_dxy": 0.01, "dlog_vix": 0.01, "d_gpr": 100.0}  # as in model.CONTROL_UNITS
SCORECARD_PATH = data.ROOT / "reports" / "forward_scorecard.md"
HORIZON = 5
HTTP_TIMEOUT = 30
CONTROL_COLS = ["d_real_yield", "dlog_dxy", "dlog_vix", "d_gpr"]
ASSETS = ("gold", "silver")


# --------------------------------------------------------------------------
# Questions and prediction markets
# --------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["questions"]


def polymarket_probability(url: str) -> float:
    """P(first outcome, normally Yes) for a polymarket.com event or market link."""
    slug = url.rstrip("/").split("/")[-1].split("?")[0]
    for kind in ("markets", "events"):
        r = requests.get(f"https://gamma-api.polymarket.com/{kind}", params={"slug": slug}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        items = r.json()
        if not items:
            continue
        market = items[0] if kind == "markets" else items[0]["markets"][0]
        prices = market.get("outcomePrices")
        prices = json.loads(prices) if isinstance(prices, str) else prices
        return float(prices[0])
    raise RuntimeError(f"no Polymarket market or event with slug {slug!r}")


def metaculus_probability(url: str) -> float:
    """Community median for a binary Metaculus question. Untested here (the API refused anonymous calls)."""
    m = re.search(r"/questions/(\d+)", url)
    if not m:
        raise ValueError(f"cannot find a question id in {url!r}")
    headers = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"} if os.environ.get("METACULUS_TOKEN") else {}
    r = requests.get(f"https://www.metaculus.com/api/posts/{m.group(1)}/", headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    q = r.json().get("question", {})
    latest = q.get("aggregations", {}).get("recency_weighted", {}).get("latest", {})
    centers = latest.get("centers") or latest.get("means")
    if not centers:
        raise RuntimeError("no community prediction in the response")
    return float(centers[0])


FETCHERS = {"polymarket": polymarket_probability, "metaculus": metaculus_probability}


def fetch_probabilities(questions: list[dict]) -> dict[str, float]:
    """id -> probability of the question as worded; NaN when the url is blank or the fetch fails.

    A question with ``invert: true`` logs 1 - P(Yes), for markets whose Yes is the
    opposite of the question (a "ceasefire continues" market standing in for "strikes resume").
    """
    out = {}
    for q in questions:
        url = (q.get("url") or "").strip()
        if not url:
            out[q["id"]] = float("nan")
            continue
        try:
            p = FETCHERS[q["source"]](url)
            out[q["id"]] = 1.0 - p if q.get("invert") else p
        except Exception as exc:
            log.warning("%s: could not fetch %s: %s", q["id"], url, exc)
            out[q["id"]] = float("nan")
    return out


# --------------------------------------------------------------------------
# Model inputs and prediction
# --------------------------------------------------------------------------


def model_version(pointer: Path | None = None) -> str:
    """Which frozen coefficients the forward log uses: ``v1`` unless the rolling rule switched it."""
    pointer = POINTER_PATH if pointer is None else pointer
    if Path(pointer).exists():
        v = Path(pointer).read_text().strip()
        if v in FROZEN_PATHS:
            return v
    return "v1"


def load_frozen(path: Path | None = None, h: int = HORIZON, version: str | None = None) -> dict:
    """The frozen coefficients for horizon ``h``, converted from percent-per-unit columns to log units.

    ``path`` wins if given; otherwise the file for ``version`` (default: the pointer file's choice).
    """
    if path is None:
        version = version or model_version()
        path = FROZEN_PATHS[version]
    table = pd.read_csv(path)
    out = {"controls": CONTROL_COLS, "horizon_days": h, "assets": {}, "version": version or "custom"}
    for r in table[table.h == h].itertuples():
        out["assets"][r.asset] = {
            "intercept": float(r.intercept_pct) / 100.0,
            "coef": {c: float(getattr(r, f"{c}_pct_per_unit")) / (100.0 * UNIT_MULT[c]) for c in CONTROL_COLS},
        }
    if set(out["assets"]) != set(ASSETS):
        raise ValueError(f"{path} has no h={h} rows for {ASSETS}")
    return out


def last_week_changes(df: pd.DataFrame, h: int = HORIZON) -> dict:
    """Control changes over the last ``h`` trading days of ``df`` (the frozen model's inputs)."""
    tail = df.dropna(subset=["real_yield", "dxy", "vix", "gpr"]).iloc[-(h + 1):]
    if len(tail) < h + 1:
        raise ValueError("not enough data for a week of control changes")
    a, b = tail.iloc[0], tail.iloc[-1]
    return {
        "as_of": tail.index[-1],
        "d_real_yield": float(b.real_yield - a.real_yield),
        "dlog_dxy": float(np.log(b.dxy / a.dxy)),
        "dlog_vix": float(np.log(b.vix / a.vix)),
        "d_gpr": float(b.gpr - a.gpr),
    }


def predict(frozen: dict, changes: dict) -> dict[str, float]:
    """Expected h-day log return per asset = intercept + sum(coef x change)."""
    out = {}
    for asset, spec in frozen["assets"].items():
        out[asset] = spec["intercept"] + sum(spec["coef"][c] * changes[c] for c in frozen["controls"])
    return out


def gpr_snapshot(gpr: pd.Series) -> dict:
    """Latest daily GPR, its date, and the change from the value seven calendar days earlier."""
    gpr = gpr.dropna().sort_index()
    latest_date = gpr.index[-1]
    prior = gpr[gpr.index <= latest_date - pd.Timedelta(days=7)]
    change = float(gpr.iloc[-1] - prior.iloc[-1]) if len(prior) else float("nan")
    return {"gpr_date": latest_date, "gpr_level": float(gpr.iloc[-1]), "gpr_change_week": change}


# --------------------------------------------------------------------------
# The log
# --------------------------------------------------------------------------


def columns(questions: list[dict]) -> list[str]:
    ids = [q["id"] for q in questions]
    return (["log_date", "model_version", "data_as_of", "gpr_date", "gpr_level", "gpr_change_week"]
            + [f"{c}_lastweek" for c in CONTROL_COLS]
            + [f"pred_{a}_5d" for a in ASSETS]
            + [f"market_prob_{i}" for i in ids]
            + ["my_tension_score"] + [f"my_prob_{i}" for i in ids]
            + [f"realised_{a}_5d" for a in ASSETS] + ["realised_through", "note"])


def build_row(log_date, questions: list[dict], frozen: dict, df: pd.DataFrame | None, gpr: pd.Series,
              probs: dict[str, float], note: str = "") -> dict:
    row = {c: np.nan for c in columns(questions)}
    row["log_date"] = pd.Timestamp(log_date).strftime("%Y-%m-%d")
    row["model_version"] = frozen.get("version", "v1")
    row.update({k: (v.strftime("%Y-%m-%d") if k == "gpr_date" else v) for k, v in gpr_snapshot(gpr).items()})
    if df is not None:
        ch = last_week_changes(df)
        row["data_as_of"] = ch["as_of"].strftime("%Y-%m-%d")
        for c in CONTROL_COLS:
            row[f"{c}_lastweek"] = ch[c]
        for a, p in predict(frozen, ch).items():
            row[f"pred_{a}_5d"] = p
    for i, p in probs.items():
        row[f"market_prob_{i}"] = p
    row["my_tension_score"] = ""
    for q in questions:
        row[f"my_prob_{q['id']}"] = ""
    row["note"] = note or "prediction uses last week's control changes as the coming week's input"
    return row


def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"my_tension_score": "string"} | {})


def append_row(row: dict, path: Path = LOG_PATH) -> pd.DataFrame:
    existing = load_log(path)
    new = pd.DataFrame([row])
    out = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    out.to_csv(path, index=False)
    return out


def log_week(log_date=None, path: Path = LOG_PATH, refresh: bool = True) -> dict:
    """Build this week's row and append it. Data problems never prevent the row; they go in ``note``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log_date = pd.Timestamp(log_date) if log_date is not None else pd.Timestamp.now("UTC").normalize().tz_localize(None)
    questions = load_questions()
    frozen = load_frozen()
    note = ""
    df = None
    try:
        df, _ = data.build(refresh=refresh)
        data.PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data.PARQUET_PATH)
        df = df.loc[:log_date]  # a backdated run sees only what was known on log_date
    except Exception as exc:  # keep logging even if a source is down
        log.warning("data build failed: %s", exc)
        note = f"data build failed ({type(exc).__name__}); no model prediction"
    gpr = data.fetch_cached("gpr", "GPRD")
    gpr = gpr[gpr.index <= log_date]
    probs = fetch_probabilities(questions)
    row = build_row(log_date, questions, frozen, df, gpr, probs, note)
    append_row(row, path)
    print(f"Logged {row['log_date']} ({row['model_version']}): pred gold {row['pred_gold_5d']}, silver {row['pred_silver_5d']}, "
          f"GPR {row['gpr_level']:.1f} ({row['gpr_change_week']:+.1f} on the week)")
    return row


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def realised_moves(df: pd.DataFrame, log_date, h: int = HORIZON) -> dict | None:
    """Log returns from the last close on or before ``log_date`` to ``h`` trading days later, if available."""
    idx = df.index
    pos = int(idx.searchsorted(pd.Timestamp(log_date), side="right")) - 1
    if pos < 0 or pos + h >= len(idx):
        return None
    out = {f"realised_{a}_5d": float(np.log(df[a].iloc[pos + h] / df[a].iloc[pos])) for a in ASSETS}
    out["realised_through"] = idx[pos + h].strftime("%Y-%m-%d")
    return out


def fill_realised(logdf: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    logdf = logdf.copy()
    for a in ASSETS:
        logdf[f"realised_{a}_5d"] = pd.to_numeric(logdf.get(f"realised_{a}_5d"), errors="coerce").astype(float)
    logdf["realised_through"] = logdf.get("realised_through", pd.Series(index=logdf.index, dtype=object)).astype(object)
    for i, r in logdf.iterrows():
        if pd.notna(r.get("realised_gold_5d")):
            continue
        got = realised_moves(df, r["log_date"])
        if got:
            for k, v in got.items():
                logdf.loc[i, k] = v
    return logdf


def prediction_stats(logdf: pd.DataFrame) -> dict:
    out = {}
    for a in ASSETS:
        sub = logdf.dropna(subset=[f"pred_{a}_5d", f"realised_{a}_5d"])
        pred, real = sub[f"pred_{a}_5d"].astype(float), sub[f"realised_{a}_5d"].astype(float)
        out[a] = {
            "n": len(sub),
            "hit_rate": float((np.sign(pred) == np.sign(real)).mean()) if len(sub) else float("nan"),
            "mean_error": float((real - pred).mean()) if len(sub) else float("nan"),
            "mae": float((real - pred).abs().mean()) if len(sub) else float("nan"),
            "mae_no_change": float(real.abs().mean()) if len(sub) else float("nan"),
        }
    return out


def brier(logdf: pd.DataFrame, questions: list[dict]) -> dict:
    """Brier scores of my and the market's probabilities over resolved questions, pooled across rows."""
    mine, market = [], []
    for q in questions:
        if not q.get("resolved") or q.get("outcome") is None:
            continue
        o = float(q["outcome"])
        for col, bucket in ((f"my_prob_{q['id']}", mine), (f"market_prob_{q['id']}", market)):
            if col in logdf.columns:
                p = pd.to_numeric(logdf[col], errors="coerce").dropna()
                bucket += [(float(x) - o) ** 2 for x in p]
    return {"n_resolved": sum(1 for q in questions if q.get("resolved") and q.get("outcome") is not None),
            "my_n": len(mine), "my_brier": float(np.mean(mine)) if mine else float("nan"),
            "market_n": len(market), "market_brier": float(np.mean(market)) if market else float("nan")}


def scorecard_markdown(logdf: pd.DataFrame, questions: list[dict]) -> str:
    ps, b = prediction_stats(logdf), brier(logdf, questions)
    L = ["# Forward scorecard", "",
         f"Updated {pd.Timestamp.now('UTC'):%Y-%m-%d}. {len(logdf)} weeks logged, "
         f"{int(logdf['realised_gold_5d'].notna().sum()) if 'realised_gold_5d' in logdf else 0} with realised moves.", "",
         "## Frozen model, 5-day prediction vs realised", "",
         "Prediction = the frozen coefficients named in each row's `model_version` (v1 fitted to 2021-12-31, or",
         "v2-rolling, the latest five-year window) applied to last week's control changes.",
         "Hit rate = share of weeks with the right sign. MAE is compared with predicting no change.", "",
         "| asset | weeks | direction hit rate | mean error (realised - pred) | MAE | MAE of no change |",
         "|---|---:|---:|---:|---:|---:|"]
    for a, s in ps.items():
        L.append(f"| {a} | {s['n']} | {s['hit_rate']:.2f} | {100 * s['mean_error']:+.2f}% | {100 * s['mae']:.2f}% | "
                 f"{100 * s['mae_no_change']:.2f}% |" if s["n"] else f"| {a} | 0 | - | - | - | - |")
    L += ["", "## Probabilities: Brier score (lower is better)", "",
          f"{b['n_resolved']} of {len(questions)} questions resolved. Pooled over every logged probability on a "
          "resolved question.", "",
          "| forecaster | probabilities scored | Brier |", "|---|---:|---:|",
          f"| me | {b['my_n']} | {b['my_brier']:.3f} |" if b["my_n"] else "| me | 0 | - |",
          f"| market | {b['market_n']} | {b['market_brier']:.3f} |" if b["market_n"] else "| market | 0 | - |", ""]
    if len(logdf):
        L += ["## Rows", "", "| week | model | pred gold | realised gold | pred silver | realised silver | GPR | GPR change | my tension |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in logdf.iterrows():
            def pc(x):
                return "-" if pd.isna(x) else f"{100 * float(x):+.2f}%"
            L.append(f"| {r['log_date']} | {r.get('model_version', 'v1')} | {pc(r.get('pred_gold_5d'))} | {pc(r.get('realised_gold_5d'))} | "
                     f"{pc(r.get('pred_silver_5d'))} | {pc(r.get('realised_silver_5d'))} | "
                     f"{r.get('gpr_level', float('nan')):.0f} | {r.get('gpr_change_week', float('nan')):+.0f} | "
                     f"{'' if pd.isna(r.get('my_tension_score')) else r.get('my_tension_score')} |")
    return "\n".join(L) + "\n"


def score_week(path: Path = LOG_PATH, scorecard: Path = SCORECARD_PATH, df: pd.DataFrame | None = None) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logdf = load_log(path)
    if logdf.empty:
        print("no rows logged yet")
        return logdf
    if df is None:
        try:
            df = data.load()
        except FileNotFoundError:
            df, _ = data.build()
            df.to_parquet(data.PARQUET_PATH)
    logdf = fill_realised(logdf, df)
    logdf.to_csv(path, index=False)
    text = scorecard_markdown(logdf, load_questions())
    Path(scorecard).parent.mkdir(parents=True, exist_ok=True)
    Path(scorecard).write_text(text)
    print(text)
    return logdf
