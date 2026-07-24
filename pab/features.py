"""Numeric sidecar: structured features that accompany the chart image (no-lookahead).

'Vision for the setup, numbers for the price.' Only bars at/<= current_ts are used, so
this is safe to call bar-by-bar inside the event-driven backtest.
"""
from __future__ import annotations

import pandas as pd

from pab.bars import ET, add_ema, prior_rth_close

TICK = 0.25
POINT_VALUE_MES = 5.0
PER_TRADE_RISK_USD = 75.0
WINDOW_BARS = 24  # 09:30-11:30 ET


def _bar_type(o: float, h: float, l: float, c: float) -> str:
    rng = (h - l) or TICK
    body = c - o
    if abs(body) < 0.10 * rng:
        return "doji"
    return "trend_bull" if body > 0 else "trend_bear"


def build_sidecar(cont: pd.DataFrame, current_ts: pd.Timestamp, *,
                  symbol: str = "ES=F", ema_period: int = 20,
                  recent_n: int = 6, spec=None,
                  window_start: str = "09:30") -> dict:
    """Structured numeric context as of `current_ts` (inclusive), no future bars used.
    `spec` (pab.instruments.InstrumentSpec) overrides the risk-config block for
    non-MES instruments; None keeps the historical MES values byte-identical.
    `window_start` anchors the session view: for the afternoon window ("13:30") the
    bar count/table start there and the morning is compressed into magnet levels —
    the morning has already happened by then, so this is lookahead-safe."""
    hist = add_ema(cont[cont.index <= current_ts], ema_period)
    if hist.empty:
        raise ValueError("no bars at/before current_ts")
    ema_col = f"ema{ema_period}"
    cur = hist.iloc[-1]
    o, h, l, c = (float(cur["open"]), float(cur["high"]),
                  float(cur["low"]), float(cur["close"]))
    rng = (h - l) or TICK

    session_date = current_ts.date()
    open_ts = pd.Timestamp(f"{session_date} {window_start}", tz=ET)
    rth_open_ts = pd.Timestamp(f"{session_date} 09:30", tz=ET)
    session = hist[hist.index >= open_ts]
    bar_index = int(len(session))

    prior_close = prior_rth_close(cont, rth_open_ts, str(session_date))
    day_session = hist[hist.index >= rth_open_ts]
    sess_open_px = float(day_session["open"].iloc[0]) if len(day_session) else o
    gap = round(sess_open_px - prior_close, 2) if prior_close is not None else None

    # magnet levels (no-lookahead: strictly before today's RTH open): prior day's RTH
    # high/low and the overnight (post-16:00 -> pre-open) high/low. Brooks: price is
    # drawn to these; stop entries INTO a nearby magnet are low-quality.
    before = cont[cont.index < rth_open_ts]
    magnets = []
    prior_dates = sorted({d for d in before.index.date if d < session_date})
    if prior_dates:
        d1 = prior_dates[-1]
        day1 = before[before.index.date == d1]
        hhmm1 = day1.index.strftime("%H:%M")
        rth1 = day1[(hhmm1 >= "09:30") & (hhmm1 <= "16:00")]
        src1 = rth1 if len(rth1) else day1
        overnight = before[before.index > src1.index[-1]]
        levels = [("yday_high", float(src1["high"].max())),
                  ("yday_low", float(src1["low"].min()))]
        if prior_close is not None:
            levels.append(("yday_close", prior_close))
        if len(overnight):
            levels += [("overnight_high", float(overnight["high"].max())),
                       ("overnight_low", float(overnight["low"].min()))]
        magnets = [{"name": nm, "px": round(px, 2), "pts_from_close": round(px - c, 2)}
                   for nm, px in levels]

    # afternoon window: compress THIS morning (already history at window open) into
    # magnet levels instead of carrying 48 bars of table — lookahead-safe by construction
    if open_ts > rth_open_ts:
        am = hist[(hist.index >= rth_open_ts) & (hist.index < open_ts)]
        if len(am):
            for nm, px in (("am_high", float(am["high"].max())),
                           ("am_low", float(am["low"].min())),
                           ("am_close", float(am["close"].iloc[-1]))):
                magnets.append({"name": nm, "px": round(px, 2),
                                "pts_from_close": round(px - c, 2)})

    ema = float(cur[ema_col])
    ema_prev = float(hist[ema_col].iloc[-4]) if len(hist) >= 4 else ema

    recent = hist.tail(recent_n)
    ai = "neutral"
    if len(recent) >= 2:
        rising = recent["close"].iloc[-1] > recent["close"].iloc[0]
        if c > ema and rising:
            ai = "ail"
        elif c < ema and not rising:
            ai = "ais"

    sess_hi = float(session["high"].max()) if len(session) else h
    sess_lo = float(session["low"].min()) if len(session) else l

    # full developing session, open -> now, as a compact pipe table — the numeric
    # "chart" the model reads bar-by-bar (~60% fewer tokens than the old JSON list).
    ecol = session[ema_col]
    rows = ["i|time|open|high|low|close|type|cp|ema"]
    so, sh, sl, sc = (session["open"].to_numpy(), session["high"].to_numpy(),
                      session["low"].to_numpy(), session["close"].to_numpy())
    for k, ts in enumerate(session.index):
        bt = _bar_type(float(so[k]), float(sh[k]), float(sl[k]), float(sc[k]))
        cp = (float(sc[k]) - float(sl[k])) / ((float(sh[k]) - float(sl[k])) or TICK)
        rows.append(f"{k+1}|{ts.strftime('%H:%M')}|{so[k]:g}|{sh[k]:g}|{sl[k]:g}"
                    f"|{sc[k]:g}|{bt}|{cp:.2f}|{float(ecol.iloc[k]):.2f}")
    session_table = "\n".join(rows)

    # session texture — the "is this a chop day" numbers (overlap ~ barbwire)
    n_s = len(session)
    if n_s >= 2:
        ovs = []
        for k in range(1, n_s):
            inter = min(sh[k], sh[k-1]) - max(sl[k], sl[k-1])
            avg_rng = ((sh[k]-sl[k]) + (sh[k-1]-sl[k-1])) / 2 or TICK
            ovs.append(max(0.0, float(inter)) / float(avg_rng))
        overlap_sess = sum(ovs) / len(ovs)
        overlap_5 = sum(ovs[-5:]) / len(ovs[-5:])
        bodies = abs(sc - so)
        small_share = float((bodies < 0.3 * (sh - sl + 1e-9)).mean())
        eff = abs(float(sc[-1] - so[0])) / ((sess_hi - sess_lo) or TICK)
        chop = {"overlap_session": round(overlap_sess, 2),
                "overlap_last5": round(overlap_5, 2),
                "efficiency": round(eff, 2),
                "small_body_share": round(small_share, 2)}
    else:
        chop = None

    # confirmed 1-bar swing pivots -> market structure labels (HH/HL/LH/LL).
    # The last bar can never be a confirmed pivot (needs a right neighbor) — causal.
    swings = []
    last_sh = last_sl = None
    for k in range(1, n_s - 1):
        if sh[k] > sh[k-1] and sh[k] > sh[k+1]:
            lab = "HH" if (last_sh is not None and sh[k] > last_sh) else \
                  ("LH" if last_sh is not None else "SH")
            last_sh = float(sh[k])
            swings.append(f"{lab}@{sh[k]:g}(bar{k+1})")
        if sl[k] < sl[k-1] and sl[k] < sl[k+1]:
            lab = "HL" if (last_sl is not None and sl[k] > last_sl) else \
                  ("LL" if last_sl is not None else "SL")
            last_sl = float(sl[k])
            swings.append(f"{lab}@{sl[k]:g}(bar{k+1})")
    swings = swings[-6:]

    # 2-3 bar combos on the most recent bars (Brooks: ii / inside / outside)
    pattern = ""
    if n_s >= 3 and sh[-1] <= sh[-2] and sl[-1] >= sl[-2] \
            and sh[-2] <= sh[-3] and sl[-2] >= sl[-3]:
        pattern = "ii"
    elif n_s >= 2 and sh[-1] <= sh[-2] and sl[-1] >= sl[-2]:
        pattern = "inside"
    elif n_s >= 2 and sh[-1] >= sh[-2] and sl[-1] <= sl[-2]:
        pattern = "outside"

    return {
        "symbol": symbol,
        "session_date": str(session_date),
        "bar_time_et": current_ts.strftime("%Y-%m-%d %H:%M %Z"),
        "bar_index_from_open": bar_index,          # 1 = 09:30 bar
        "in_trade_window": bar_index <= WINDOW_BARS,
        "bar": {"o": round(o, 2), "h": round(h, 2), "l": round(l, 2),
                "c": round(c, 2), "type": _bar_type(o, h, l, c),
                "close_pos_in_range": round((c - l) / rng, 2),
                "range_pts": round(rng, 2), "volume": int(cur["volume"])},
        "ema20": round(ema, 2),
        "close_vs_ema_pts": round(c - ema, 2),
        "ema_slope_pts_3bar": round(ema - ema_prev, 2),
        "prior_rth_close": None if prior_close is None else round(prior_close, 2),
        "open_gap_pts": gap,
        "session_high": round(sess_hi, 2),
        "session_low": round(sess_lo, 2),
        "pts_below_session_high": round(sess_hi - c, 2),
        "pts_above_session_low": round(c - sess_lo, 2),
        "always_in_hint": ai,
        "magnets": magnets,
        "chop": chop,
        "swings": swings,
        "last_bars_pattern": pattern,
        "session_table": session_table,
        "config": ({"tick": TICK, "point_value_usd": POINT_VALUE_MES,
                    "per_trade_risk_usd": PER_TRADE_RISK_USD,
                    "per_trade_risk_pts": round(PER_TRADE_RISK_USD / POINT_VALUE_MES, 2)}
                   if spec is None else
                   {"tick": spec.tick, "point_value_usd": spec.point_value_usd,
                    "per_trade_risk_usd": spec.per_trade_risk_usd,
                    "per_trade_risk_pts": round(spec.per_trade_risk_pts, 2),
                    "qty": spec.qty}),
    }


def classify_regime(sidecar: dict, window: str = "am") -> str:
    """Coarse, deterministic regime from the sidecar (no LLM, no lookahead) — used to
    ROUTE which setup cards enter the prompt, and to retrieve regime-matched experience.
    Deliberately generous: it narrows the card set, the model still makes the read.
      open  — first ~6 bars, session structure not yet formed (AM window only: the
              afternoon window opens onto a morning that already happened, so bar 3
              of the PM window is not an "opening" bar — fall through to trend/range)
      trend — always-in one side + EMA slope/extension agree
      range — everything else (two-sided, EMA flat, price straddling)"""
    if window == "am" and sidecar["bar_index_from_open"] <= 6:
        return "open"
    if (sidecar["always_in_hint"] in ("ail", "ais")
            and (abs(sidecar["ema_slope_pts_3bar"]) >= 1.0
                 or abs(sidecar["close_vs_ema_pts"]) >= 6.0)):
        return "trend"
    return "range"
