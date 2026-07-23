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


# ---- LLM day grader: the model judges A/C itself (user-directed). Inputs are
# STRICTLY prospective: yesterday's full RTH table + overnight/gap for the AM
# window; plus TODAY'S morning table for the PM window (already history at 13:30).
# Cached on disk so identical (inputs, model) never re-pays. ----

GRADER_SYSTEM = """You are an Al Brooks day-type analyst. Judge whether the UPCOMING \
trading window is worth trading with a with-trend price-action strategy.

Grade A = tradeable: odds favor directional structure (trend legs, breakout with \
follow-through, clean two-legged pullbacks).
Grade C = skip: odds favor a two-sided chop/barbwire session (high bar overlap, low \
directional efficiency, many small-body bars, always-in flip-flopping).

You see ONLY information available BEFORE the window opens. Weigh: yesterday's texture \
(overlap/efficiency), range vs recent average, the overnight gap (big gaps breed trend \
days more often), and — for afternoon windows — this morning's character and where price \
sits in the morning range. Be decisive; roughly half of all days are C.

Return ONLY JSON: {"grade": "A" | "C", "confidence": <0-100>, "reason": "<one or two \
sentences citing the specific evidence>"}"""


def _session_table_of(cont, start_ts, end_ts) -> str:
    sl = cont[(cont.index >= start_ts) & (cont.index <= end_ts)]
    rows = ["time|open|high|low|close"]
    for ts, r in sl.iterrows():
        rows.append(f"{ts.strftime('%H:%M')}|{r.open:g}|{r.high:g}|{r.low:g}|{r.close:g}")
    return "\n".join(rows)


def grade_llm(cont, session_date: str, *, window: str = "am", cfg=None) -> dict:
    """Opus-judged day grade. Returns {"grade","confidence","reason"} (+"_cached")."""
    import hashlib
    import json as _json
    from pathlib import Path

    from pab.agent import AgentConfig, CACHE_DIR, _loads_lenient
    from pab.review import _chat_text

    cfg = cfg or AgentConfig()
    open_ts = pd.Timestamp(f"{session_date} 09:30", tz=ET)
    feats = day_features(cont, session_date)
    if feats is None:
        return {"grade": "A", "confidence": 0, "reason": "no prior-day data"}

    before = cont[cont.index < open_ts]
    prior_days = sorted({d for d in before.index.date if d < open_ts.date()})
    d1 = prior_days[-1]
    y_tbl = _session_table_of(cont, pd.Timestamp(f"{d1} 09:30", tz=ET),
                              pd.Timestamp(f"{d1} 15:55", tz=ET))
    overnight = before[before.index > pd.Timestamp(f"{d1} 15:55", tz=ET)]
    on_line = (f"overnight: high {float(overnight['high'].max()):g} / "
               f"low {float(overnight['low'].min()):g}" if len(overnight) else
               "overnight: n/a")
    today = cont[(cont.index >= open_ts)]
    try:  # yesterday may have no RTH bars (dataset's first day) -> header-only table
        gap = f"{float(today['open'].iloc[0]) - float(y_tbl.splitlines()[-1].split('|')[-1]):+.2f}"
    except (ValueError, IndexError):
        gap = "n/a"

    parts = [f"UPCOMING WINDOW: {'09:30-11:30' if window == 'am' else '13:30-15:30'} ET "
             f"on {session_date}",
             f"yesterday features: {feats}",
             f"YESTERDAY ({d1}) 5m RTH bars:\n{y_tbl}", on_line,
             f"today's RTH open gap vs yesterday close: {gap}"]
    if window == "pm":
        am_tbl = _session_table_of(cont, open_ts,
                                   pd.Timestamp(f"{session_date} 13:25", tz=ET))
        parts.append(f"THIS MORNING ({session_date} 09:30-13:25) 5m bars:\n{am_tbl}")
    user = "\n\n".join(parts) + '\n\nReturn ONLY the JSON: {"grade","confidence","reason"}'

    key = hashlib.sha256(_json.dumps(
        {"provider": cfg.provider, "model": cfg.resolved_model(), "window": window,
         "system": GRADER_SYSTEM, "user": user}, sort_keys=True).encode()).hexdigest()
    cache = Path(CACHE_DIR).parent / "daygrades" / f"{key}.json"
    if cache.exists():
        try:
            out = _json.loads(cache.read_text("utf-8"))
            out["_cached"] = True
            return out
        except Exception:  # noqa: BLE001
            pass
    out = _loads_lenient(_chat_text(GRADER_SYSTEM, user, cfg))
    if out.get("grade") not in ("A", "C"):
        out = {"grade": "A", "confidence": 0,
               "reason": f"unparseable grade, defaulting A: {str(out)[:80]}"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(_json.dumps(out, ensure_ascii=False), "utf-8")
    return out
