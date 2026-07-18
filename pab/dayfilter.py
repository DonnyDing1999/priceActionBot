"""Prospective day-level trade/no-trade grading — the anti-chop switch.

Computed each morning from bars STRICTLY BEFORE today's open (no lookahead):
yesterday's texture (bar overlap, directional efficiency, small-body ratio) and
range compression vs the 10-day average. Chop begets chop; trend begets trend —
Brooks' "yesterday tells you what kind of day to expect" heuristic, quantified.

Deliberately dumb: a handful of features, thresholds set at round quartile-ish
values, NO optimization loop (researcher-overfitting guard). Any adoption must
survive a fresh full run and eventually out-of-sample data.
"""
from __future__ import annotations

import pandas as pd

from pab.bars import ET


def day_features(cont: pd.DataFrame, session_date: str) -> dict | None:
    """Texture features of the PRIOR RTH day (+10-day range context), known at
    today's 09:30 open. None if there is no prior day in the data."""
    open_ts = pd.Timestamp(f"{session_date} 09:30", tz=ET)
    before = cont[cont.index < open_ts]
    days = sorted({d for d in before.index.date if d < open_ts.date()})
    if not days:
        return None

    def rth(d):
        day = before[before.index.date == d]
        hhmm = day.index.strftime("%H:%M")
        sl = day[(hhmm >= "09:30") & (hhmm <= "15:55")]
        return sl if len(sl) else day

    y = rth(days[-1])
    if len(y) < 10:
        return None
    hi, lo = y["high"].to_numpy(), y["low"].to_numpy()
    op, cl = y["open"].to_numpy(), y["close"].to_numpy()

    # mean overlap of consecutive bars, normalized by their average range
    ov = []
    for i in range(1, len(y)):
        inter = min(hi[i], hi[i - 1]) - max(lo[i], lo[i - 1])
        avg_rng = ((hi[i] - lo[i]) + (hi[i - 1] - lo[i - 1])) / 2 or 0.25
        ov.append(max(0.0, inter) / avg_rng)
    overlap = sum(ov) / len(ov)

    day_range = float(hi.max() - lo.min()) or 0.25
    efficiency = abs(float(cl[-1] - op[0])) / day_range          # 1 = clean trend day
    bodies = abs(cl - op)
    small_body = float((bodies < 0.3 * (hi - lo + 1e-9)).mean())  # doji-ish share

    ranges = []
    for d in days[-10:]:
        r = rth(d)
        if len(r):
            ranges.append(float(r["high"].max() - r["low"].min()))
    adr = sum(ranges) / len(ranges) if ranges else day_range
    range_vs_adr = day_range / adr if adr else 1.0

    return {"overlap": round(overlap, 3), "efficiency": round(efficiency, 3),
            "small_body": round(small_body, 3), "range_vs_adr": round(range_vs_adr, 3)}


# --- grading variants (round thresholds, chosen a priori — see module docstring) ---

def grade(feats: dict | None, variant: str = "combo") -> str:
    """'A' = trade normally, 'C' = chop-expected day (throttle or skip)."""
    if feats is None:
        return "A"                      # no information -> don't invent a filter
    f = feats
    if variant == "overlap":
        return "C" if f["overlap"] > 0.60 else "A"
    if variant == "efficiency":
        return "C" if f["efficiency"] < 0.25 else "A"
    if variant == "combo":
        chop_texture = f["overlap"] > 0.55 and f["efficiency"] < 0.30
        compressed = f["range_vs_adr"] < 0.65
        return "C" if (chop_texture or compressed) else "A"
    raise ValueError(variant)
