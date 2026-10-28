"""Mechanical event list: days when the GPR or EPU index is in its own top 5%.

Run ``python -m gold_events.mechanical`` to write ``data/events_mechanical.csv``
and ``reports/events_mechanical.md``.

Rules
-----
* Universe: trading days 1990-01-02 .. 2021-12-31 (nothing later is read).
* A day is *flagged* when GPR or EPU exceeds its own 95th percentile over
  that universe.
* Flagged days within ``MERGE_DAYS`` trading days of the previous flagged
  day belong to the same event.  The event is dated at its first flagged day.
* Each event keeps the peak GPR and EPU over its span, which index triggered
  it (``gpr``, ``epu`` or ``both``), and ``peak_z`` = the larger of the two
  peaks as z-scores of the daily levels, used to rank events by size.
* ``severity_measured`` is the same jump z-score used for the hand list
  (:func:`gold_events.events.jump_columns`), so the two lists are comparable
  in the model.
"""

from __future__ import annotations

import pandas as pd

from gold_events import data
from gold_events.events import EVENTS_PATH, REF_END, REF_START, jump_columns, load_events
from gold_events.event_study import first_trading_day_on_or_after

MECH_PATH = data.DATA_DIR / "events_mechanical.csv"
REPORT_PATH = data.ROOT / "reports" / "events_mechanical.md"
PERCENTILE = 0.95
MERGE_DAYS = 5
NEAR_HAND_DAYS = 5


def flag_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Trading days in the universe with flags and the thresholds used."""
    f = frame.loc[REF_START:REF_END, ["gpr", "epu"]].dropna()
    thr = {c: float(f[c].quantile(PERCENTILE)) for c in ("gpr", "epu")}
    f = f.assign(gpr_flag=f["gpr"] > thr["gpr"], epu_flag=f["epu"] > thr["epu"])
    f["flag"] = f["gpr_flag"] | f["epu_flag"]
    return f, thr


def merge_runs(f: pd.DataFrame) -> pd.DataFrame:
    """Group flagged days into events; rows = events."""
    flagged = f[f["flag"]]
    pos = pd.Series([f.index.get_loc(d) for d in flagged.index], index=flagged.index)
    new_event = pos.diff().fillna(MERGE_DAYS + 1) > MERGE_DAYS
    event_id = new_event.cumsum()
    rows = []
    for _, days in flagged.groupby(event_id):
        span = f.loc[days.index[0]: days.index[-1]]
        trig = {"gpr": days["gpr_flag"].any(), "epu": days["epu_flag"].any()}
        rows.append({
            "date": days.index[0],
            "category": "",
            "end_date": days.index[-1],
            "n_flagged_days": len(days),
            "trigger": "both" if all(trig.values()) else ("gpr" if trig["gpr"] else "epu"),
            "peak_gpr": round(float(span["gpr"].max()), 1),
            "peak_gpr_date": span["gpr"].idxmax(),
            "peak_epu": round(float(span["epu"].max()), 1),
            "peak_epu_date": span["epu"].idxmax(),
        })
    return pd.DataFrame(rows)


def build(frame: pd.DataFrame, hand: pd.DataFrame, gpr_daily: pd.Series, epu_daily: pd.Series) -> tuple[pd.DataFrame, dict]:
    f, thr = flag_days(frame)
    mech = merge_runs(f)
    z = {c: ((mech[f"peak_{c}"] - f[c].mean()) / f[c].std()).round(2) for c in ("gpr", "epu")}
    mech["peak_gpr_z"], mech["peak_epu_z"] = z["gpr"], z["epu"]
    mech["peak_z"] = mech[["peak_gpr_z", "peak_epu_z"]].max(axis=1)
    jumps = jump_columns(mech["date"], gpr_daily, epu_daily)
    mech["gpr_z"], mech["epu_z"] = jumps["gpr_z"], jumps["epu_z"]
    mech["severity_measured"] = jumps["severity_measured"]
    mech["severity_capped"] = jumps["severity_capped"]

    # distance to the nearest hand-list event (trading days), both directions
    cal = f.index
    hand_in = hand[(hand.date >= REF_START) & (hand.date <= REF_END)].copy()
    hand_pos = pd.Series([cal.get_loc(first_trading_day_on_or_after(cal, d)) for d in hand_in.date], index=hand_in.index)
    mech_pos = pd.Series([cal.get_loc(d) for d in mech.date], index=mech.index)
    mech["nearest_hand_days"] = [int((hand_pos - p).abs().min()) for p in mech_pos]
    mech["near_hand_event"] = mech["nearest_hand_days"] <= NEAR_HAND_DAYS
    mech["hand_event"] = [hand_in.loc[(hand_pos - p).abs().idxmin(), "event"] if (hand_pos - p).abs().min() <= NEAR_HAND_DAYS else ""
                          for p in mech_pos]
    # hand events falling anywhere inside a mechanical event's span (first to last flagged day)
    hand_day0 = pd.Series([cal[p] for p in hand_pos], index=hand_in.index)
    mech["hand_events_in_span"] = [
        "; ".join(hand_in.loc[(hand_day0 >= r.date) & (hand_day0 <= r.end_date), "event"]) for r in mech.itertuples()]
    hand_in["nearest_mech_days"] = [int((mech_pos - p).abs().min()) for p in hand_pos]
    hand_in["flagged"] = hand_in["nearest_mech_days"] <= NEAR_HAND_DAYS
    hand_in["in_span"] = [bool(((mech.date <= d) & (mech.end_date >= d)).any()) for d in hand_day0]
    info = {"thresholds": thr, "n_days": len(f), "first": f.index[0], "last": f.index[-1], "n_flagged": int(f["flag"].sum()),
            "n_gpr_flagged": int(f["gpr_flag"].sum()), "n_epu_flagged": int(f["epu_flag"].sum()),
            "hand": hand_in}
    return mech, info


def report(mech: pd.DataFrame, info: dict) -> str:
    hand = info["hand"]
    thr = info["thresholds"]
    missed = hand[~hand.flagged]
    never = hand[~hand.flagged & ~hand.in_span]
    top = mech[~mech.near_hand_event & (mech.hand_events_in_span == "")].sort_values("peak_z", ascending=False).head(10)
    lines = [
        "# Mechanical event list",
        "",
        f"Universe: {info['n_days']} trading days {info['first']:%Y-%m-%d} to {info['last']:%Y-%m-%d}. A day is flagged when "
        f"GPR > {thr['gpr']:.1f} or EPU > {thr['epu']:.1f} (each index's own 95th percentile over the universe). "
        f"{info['n_flagged']} days are flagged ({info['n_gpr_flagged']} by GPR, {info['n_epu_flagged']} by EPU). "
        f"Flagged days within {MERGE_DAYS} trading days of the previous flagged day are merged into one event.",
        "",
        f"**{len(mech)} mechanical events.** Triggered by GPR only: {(mech.trigger == 'gpr').sum()}, EPU only: "
        f"{(mech.trigger == 'epu').sum()}, both: {(mech.trigger == 'both').sum()}. Median span "
        f"{mech.n_flagged_days.median():.0f} flagged days; longest {mech.n_flagged_days.max()} "
        f"({mech.loc[mech.n_flagged_days.idxmax(), 'date']:%Y-%m-%d}).",
        "",
        f"**Overlap with the hand list.** {mech.near_hand_event.sum()} of {len(mech)} mechanical events start within "
        f"{NEAR_HAND_DAYS} trading days of a hand-list event, and {(mech.hand_events_in_span != '').sum()} contain a "
        f"hand-list event somewhere in their span. Of the {len(hand)} hand-list events in 1990-2021, "
        f"{len(hand) - len(missed)} are within {NEAR_HAND_DAYS} trading days of a mechanical event's start, "
        f"{hand.in_span.sum()} fall inside a mechanical event's span, and {len(never)} are neither: the indices never "
        f"flagged them.",
        "",
        f"## Hand-list events the indices never flagged ({len(never)})",
        "",
        "Not within 5 trading days of a mechanical event's start and not inside any mechanical event's span.",
        "",
        "| date | category | event | severity_hand | gpr_z | epu_z | nearest mechanical start (trading days) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in never.iterrows():
        lines.append(f"| {r.date:%Y-%m-%d} | {r.category} | {r.event} | {r.severity_hand} | {r.gpr_z:+.2f} | "
                     f"{r.epu_z:+.2f} | {r.nearest_mech_days} |")
    inside_only = hand[~hand.flagged & hand.in_span]
    if len(inside_only):
        lines += ["", f"A further {len(inside_only)} hand-list events are not within {NEAR_HAND_DAYS} days of a mechanical "
                  "start but fall inside a longer mechanical event's span: "
                  + "; ".join(f"{r.event} ({r.date:%Y-%m-%d})" for _, r in inside_only.iterrows()) + "."]
    lines += ["", "## Ten largest mechanical events not on the hand list", "",
              "Not within 5 trading days of a hand-list event and containing none in their span. Ranked by peak_z, "
              "the larger of the two peak index levels as a z-score of daily levels 1990-2021.", "",
              "| date | end | trigger | peak GPR | peak EPU | peak_z | severity (jump z) |",
              "|---|---|---|---:|---:|---:|---:|"]
    for _, r in top.iterrows():
        lines.append(f"| {r.date:%Y-%m-%d} | {r.end_date:%Y-%m-%d} | {r.trigger} | {r.peak_gpr:.0f} | {r.peak_epu:.0f} | "
                     f"{r.peak_z:.1f} | {r.severity_measured:+.2f} |")
    lines += ["", "## All mechanical events", "",
              "| date | end | days | trigger | peak GPR | peak EPU | peak_z | severity | hand-list event at start | hand-list events in span |",
              "|---|---|---:|---|---:|---:|---:|---:|---|---|"]
    for _, r in mech.iterrows():
        lines.append(f"| {r.date:%Y-%m-%d} | {r.end_date:%Y-%m-%d} | {r.n_flagged_days} | {r.trigger} | {r.peak_gpr:.0f} | "
                     f"{r.peak_epu:.0f} | {r.peak_z:.1f} | {r.severity_measured:+.2f} | {r.hand_event} | {r.hand_events_in_span} |")
    return "\n".join(lines) + "\n"


def load_mechanical(path=MECH_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date", "end_date", "peak_gpr_date", "peak_epu_date"])


def main() -> None:
    frame = data.load().loc[:REF_END]
    hand = load_events()
    gpr = data.fetch_cached("gpr", "GPRD")
    epu = data.fetch_cached("epu", "daily_policy_index")
    mech, info = build(frame, hand, gpr, epu)
    mech.to_csv(MECH_PATH, index=False, date_format="%Y-%m-%d")
    text = report(mech, info)
    REPORT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
