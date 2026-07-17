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
                  recent_n: int = 6, spec=None) -> dict:
    """Structured numeric context as of `current_ts` (inclusive), no future bars used.
    `spec` (pab.instruments.InstrumentSpec) overrides the risk-config block for
    non-MES instruments; None keeps the historical MES values byte-identical."""
    hist = add_ema(cont[cont.index <= current_ts], ema_period)
    if hist.empty:
        raise ValueError("no bars at/before current_ts")
    ema_col = f"ema{ema_period}"
    cur = hist.iloc[-1]
    o, h, l, c = (float(cur["open"]), float(cur["high"]),
                  float(cur["low"]), float(cur["close"]))
    rng = (h - l) or TICK

    session_date = current_ts.date()
    open_ts = pd.Timestamp(f"{session_date} 09:30", tz=ET)
    session = hist[hist.index >= open_ts]
    bar_index = int(len(session))

    prior_close = prior_rth_close(cont, open_ts, str(session_date))
    sess_open_px = float(session["open"].iloc[0]) if len(session) else o
    gap = round(sess_open_px - prior_close, 2) if prior_close is not None else None

    # magnet levels (no-lookahead: strictly before today's open): prior day's RTH
    # high/low and the overnight (post-16:00 -> pre-open) high/low. Brooks: price is
    # drawn to these; stop entries INTO a nearby magnet are low-quality.
    before = cont[cont.index < open_ts]
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

    recent_bars = [
        {"t": ts.strftime("%H:%M"), "o": round(float(r.open), 2),
         "h": round(float(r.high), 2), "l": round(float(r.low), 2),
         "c": round(float(r.close), 2), "v": int(r.volume)}
        for ts, r in recent.iterrows()
    ]

    sess_hi = float(session["high"].max()) if len(session) else h
    sess_lo = float(session["low"].min()) if len(session) else l

    # full developing session, open -> now, with per-bar EMA — the numeric "chart"
    # the model reads bar-by-bar (replaces the rendered image in the no-vision path).
    ecol = session[ema_col]
    session_bars = [
        {"i": k + 1, "t": ts.strftime("%H:%M"),
         "o": round(float(r.open), 2), "h": round(float(r.high), 2),
         "l": round(float(r.low), 2), "c": round(float(r.close), 2),
         "type": _bar_type(float(r.open), float(r.high), float(r.low), float(r.close)),
         "cp": round((float(r.close) - float(r.low)) / ((float(r.high) - float(r.low)) or TICK), 2),
         "e": round(float(ecol.iloc[k]), 2)}
        for k, (ts, r) in enumerate(session.iterrows())
    ]

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
        "session_bars": session_bars,
        "recent_bars": recent_bars,
        "config": ({"tick": TICK, "point_value_usd": POINT_VALUE_MES,
                    "per_trade_risk_usd": PER_TRADE_RISK_USD,
                    "per_trade_risk_pts": round(PER_TRADE_RISK_USD / POINT_VALUE_MES, 2)}
                   if spec is None else
                   {"tick": spec.tick, "point_value_usd": spec.point_value_usd,
                    "per_trade_risk_usd": spec.per_trade_risk_usd,
                    "per_trade_risk_pts": round(spec.per_trade_risk_pts, 2),
                    "qty": spec.qty}),
    }


def classify_regime(sidecar: dict) -> str:
    """Coarse, deterministic regime from the sidecar (no LLM, no lookahead) — used to
    ROUTE which setup cards enter the prompt, and to retrieve regime-matched experience.
    Deliberately generous: it narrows the card set, the model still makes the read.
      open  — first ~6 bars, session structure not yet formed
      trend — always-in one side + EMA slope/extension agree
      range — everything else (two-sided, EMA flat, price straddling)"""
    if sidecar["bar_index_from_open"] <= 6:
        return "open"
    if (sidecar["always_in_hint"] in ("ail", "ais")
            and (abs(sidecar["ema_slope_pts_3bar"]) >= 1.0
                 or abs(sidecar["close_vs_ema_pts"]) >= 6.0)):
        return "trend"
    return "range"
