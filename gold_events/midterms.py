"""First analysis: gold and silver around every US midterm election since 1994.

Run ``python -m gold_events.midterms`` to write ``reports/midterms.md``.

Day 0 is the first trading day *after* election day (results are known
overnight), so the 1-day move is the market's first reaction.
"""

from __future__ import annotations


import pandas as pd

from gold_events import data
from gold_events.event_study import ASSETS, HORIZONS, EventStudyResult, event_study, first_trading_day_after

REPORT_PATH = data.ROOT / "reports" / "midterms.md"

# (election date, did the House change hands)
MIDTERMS: list[tuple[str, bool]] = [
    ("1994-11-08", True),
    ("1998-11-03", False),
    ("2002-11-05", False),
    ("2006-11-07", True),
    ("2010-11-02", True),
    ("2014-11-04", False),
    ("2018-11-06", True),
    ("2022-11-08", True),
]


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Event study for every midterm. Long frame: one row per (election, asset, horizon)."""
    frames = []
    for date, flipped in MIDTERMS:
        day0 = first_trading_day_after(df.index, date)
        res: EventStudyResult = event_study(df, day0, event_date=date)
        t = res.table.copy()
        t.insert(0, "election", pd.Timestamp(date))
        t.insert(1, "day0", day0)
        t.insert(2, "house_flipped", flipped)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:+.2f}%"


def _wide(results: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Rows = elections, columns = raw/net per horizon, for one asset."""
    sub = results[results.asset == asset]
    wide = sub.pivot(index="election", columns="horizon", values=["raw", "net"])
    return wide


def _per_election_table(results: pd.DataFrame, asset: str) -> list[str]:
    wide = _wide(results, asset)
    meta = results.drop_duplicates("election").set_index("election")
    head = "| Midterm | Day 0 | House flipped | " + " | ".join(f"raw {h}d | net {h}d" for h in HORIZONS) + " |"
    sep = "|---|---|:---:|" + "---:|---:|" * len(HORIZONS)
    lines = [head, sep]
    for election, row in wide.iterrows():
        cells = [f"{election:%Y-%m-%d}", f"{meta.loc[election, 'day0']:%Y-%m-%d}",
                 "yes" if meta.loc[election, "house_flipped"] else "no"]
        for h in HORIZONS:
            cells += [_pct(row[("raw", h)]), _pct(row[("net", h)])]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _averages(results: pd.DataFrame) -> pd.DataFrame:
    """Mean raw/net by (group, asset, horizon); groups = all / flipped / held."""
    valid = results.dropna(subset=["raw", "net"])
    groups = {
        "All midterms": valid,
        "House changed hands": valid[valid.house_flipped],
        "House did not change": valid[~valid.house_flipped],
    }
    rows = []
    for label, g in groups.items():
        agg = g.groupby(["asset", "horizon"]).agg(raw=("raw", "mean"), net=("net", "mean"),
                                                  n=("election", "nunique")).reset_index()
        agg.insert(0, "group", label)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def _averages_table(avg: pd.DataFrame, asset: str) -> list[str]:
    head = "| Group | n | " + " | ".join(f"raw {h}d | net {h}d" for h in HORIZONS) + " |"
    sep = "|---|---:|" + "---:|---:|" * len(HORIZONS)
    lines = [head, sep]
    for label in avg.group.unique():
        g = avg[(avg.group == label) & (avg.asset == asset)].set_index("horizon")
        n = int(g["n"].iloc[0]) if len(g) else 0
        cells = [label, str(n)]
        for h in HORIZONS:
            cells += [_pct(g.loc[h, "raw"]) if h in g.index else "n/a",
                      _pct(g.loc[h, "net"]) if h in g.index else "n/a"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _attribution_table(results: pd.DataFrame, horizon: int = 20) -> list[str]:
    sub = results[results.horizon == horizon]
    lines = [
        f"| Midterm | Asset | beta DXY | beta real yield | dlog DXY {horizon}d | d real yield {horizon}d (pp) | explained | raw | net |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in sub.iterrows():
        if pd.isna(r.raw):
            continue
        lines.append(f"| {r.election:%Y} | {r.asset} | {r.beta_dxy:+.2f} | {r.beta_real_yield:+.3f} | "
                     f"{100 * r.d_dxy:+.2f}% | {r.d_real_yield:+.2f} | {_pct(r.explained)} | {_pct(r.raw)} | {_pct(r.net)} |")
    return lines


def _reading(results: pd.DataFrame, avg: pd.DataFrame, df: pd.DataFrame) -> str:
    """One plain-English paragraph built from the numbers."""
    valid = results.dropna(subset=["raw"])
    years = sorted(valid.election.dt.year.unique())
    if not years:
        return "No midterm had usable price data, so there is nothing to read yet."

    def a(group, asset, h, col):
        g = avg[(avg.group == group) & (avg.asset == asset) & (avg.horizon == h)]
        return float(g[col].iloc[0]) if len(g) else float("nan")

    def direction(x):
        return "rose" if x > 0 else "fell"

    g_raw, g_net = a("All midterms", "gold", 20, "raw"), a("All midterms", "gold", 20, "net")
    s_raw, s_net = a("All midterms", "silver", 20, "raw"), a("All midterms", "silver", 20, "net")
    g_flip, g_held = a("House changed hands", "gold", 20, "net"), a("House did not change", "gold", 20, "net")
    g1_flip, g1_held = a("House changed hands", "gold", 1, "net"), a("House did not change", "gold", 1, "net")
    n_flip = int(valid[valid.house_flipped].election.nunique())
    n_held = int(valid[~valid.house_flipped].election.nunique())
    gold20 = valid[(valid.asset == "gold") & (valid.horizon == 20)]
    lo, hi = gold20.net.min(), gold20.net.max()
    n_up, n_all = int((gold20.net > 0).sum()), len(gold20)

    text = (
        f"Across the {len(years)} midterms with price data ({years[0]}-{years[-1]}), gold on average "
        f"{direction(g_raw)} {_pct(g_raw)} over the 20 trading days after the election; once the part "
        f"explained by the dollar and real yields is removed the net move is {_pct(g_net)}, "
        f"{'so most of the raw move is a dollar/real-yield story rather than an election story' if abs(g_net) < abs(g_raw) / 2 else 'so the move is mostly not explained by the dollar or real yields'}. "
        f"Silver {direction(s_raw)} {_pct(s_raw)} raw and {_pct(s_net)} net over the same window. "
        f"Splitting by outcome, gold's net 20-day move averaged {_pct(g_flip)} when the House changed hands "
        f"(n={n_flip}) versus {_pct(g_held)} when it did not (n={n_held}); on day one the split is "
        f"{_pct(g1_flip)} versus {_pct(g1_held)}. "
        f"The individual net 20-day gold moves range from {_pct(lo)} to {_pct(hi)} ({n_up} of {n_all} positive), "
        f"so with this few events the averages are driven by one or two years and should be read as "
        f"anecdotes, not as evidence of a reliable midterm effect."
    )
    return text


def report(results: pd.DataFrame, df: pd.DataFrame) -> str:
    avg = _averages(results)
    missing = results[results.raw.isna()].drop_duplicates("election")
    lines = [
        "# Gold and silver around US midterm elections",
        "",
        f"Built {pd.Timestamp.now():%Y-%m-%d} from `data/daily.parquet` "
        f"(gold/silver source: see `reports/data_coverage.md`).",
        "",
        "Day 0 = first trading day after election day. Returns are log returns from the close of the",
        "day before day 0 to the close of day h-1 (h = 1, 5, 20 trading days). *Net* subtracts the move",
        "explained by the same-window change in the dollar index (DXY) and the 10y real yield, using",
        "betas from an OLS on the 250 trading days before day 0. Real yield is the TIPS yield from",
        "2003 and a nominal-minus-trailing-CPI proxy before that.",
        "",
    ]
    if len(missing):
        yrs = ", ".join(f"{d:%Y}" for d in missing.election)
        first_gold = df["gold"].first_valid_index()
        lines += [f"**Not computable in this build:** {yrs} (gold/silver data start {first_gold:%Y-%m-%d}; "
                  "re-run `python -m gold_events.data` from a machine where Stooq is reachable to extend "
                  "the series back to 1990).", ""]
    for asset in ASSETS:
        lines += [f"## {asset.capitalize()}", ""] + _per_election_table(results, asset) + [""]
        lines += [f"### {asset.capitalize()}: averages", ""] + _averages_table(avg, asset) + [""]
    lines += ["## Attribution at 20 days", "",
              "What the controls explain: beta x same-window change, for each event.", ""]
    lines += _attribution_table(results) + [""]
    lines += ["## Reading", "", _reading(results, avg, df), ""]
    return "\n".join(lines)


def main() -> None:
    df = data.load()
    results = run(df)
    text = report(results, df)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
