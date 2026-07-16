"""Event-driven, bar-by-bar backtester (strict no-lookahead) for the opening window.

Backbone of the system. For each session it walks the 24 opening-window bars in order.
At each flat bar it builds the numeric sidecar from bars <= current only and calls a
Strategy. Entry honors the signal's `entry_type`:
  - "stop" (the Brooks default): a stop order 1 tick beyond the signal bar's extreme,
    working for the NEXT bar only — if that bar never triggers it, the order is
    canceled and no trade happens (failed signals filter themselves out).
  - "market"/"limit": fill at the next bar's open (limit approximated as market for now).
Stops/targets are scanned bar-by-bar; when 1-minute bars are supplied (`m1=`), the
scan runs minute-by-minute INSIDE each 5m bar, resolving stop-vs-target ordering and
pre-fill vs post-fill price action exactly (ambiguity shrinks from 5 minutes to 1;
within a single 1m bar the stop still wins, conservatively).

ENTRIES stop at the window end, but an open position is NOT force-flattened there:
it keeps working to its stop or target through the afternoon (exit scan runs on bars
up to Config.flat_by) and is force-closed only at that bar's close ("eod") — never
held overnight, so the per-trade $ cap stays enforceable. While a position works,
no new entries happen (1 contract). Commission + 1-tick adverse slippage are charged.
The four hard-risk caps gate every entry via RiskManager; pass `stats=` a dict to
collect veto counts / signals / no-fills for observability.

`toy_strategy` here only validates the plumbing (it has no edge).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from pab.bars import ET
from pab.features import build_sidecar
from pab.risk import RiskManager

Strategy = Callable[[dict, pd.DataFrame], Optional["Signal"]]


@dataclass
class Config:
    tick: float = 0.25
    point_value: float = 5.0          # MES $/pt
    commission_rt: float = 1.5        # round-turn $
    slippage_ticks: int = 1           # adverse, per fill (entry + stop)
    per_trade_risk_usd: float = 75.0
    min_rr: float = 1.0               # reward:risk floor on the near target
    max_trades: int = 3
    max_consec_loss: int = 2
    daily_loss_cap: float = 500.0
    window_end: str = "11:25"         # last ENTRY bar (signals come from 09:30-11:30 only)
    flat_by: str = "15:55"            # open trades may run past the window to their stop/
                                      # target; force-flat at this bar's close (no overnight)


@dataclass
class Signal:
    side: str            # "long" | "short"
    stop: float
    target: float
    reason: str = ""
    entry_type: str = "market"   # "stop" (1 tick beyond signal bar) | "market" | "limit"


@dataclass
class Trade:
    session: str
    side: str
    entry_ts: str
    exit_ts: str
    entry: float
    exit: float
    stop: float
    target: float
    risk_pts: float
    pnl_pts: float
    pnl_usd: float
    r: float
    exit_reason: str     # stop | target | eod (force-flat at Config.flat_by)
    reason: str


def session_window(cont: pd.DataFrame, session_date: str,
                   window_end: str = "11:25") -> pd.DataFrame:
    open_ts = pd.Timestamp(f"{session_date} 09:30", tz=ET)
    end_ts = pd.Timestamp(f"{session_date} {window_end}", tz=ET)
    return cont[(cont.index >= open_ts) & (cont.index <= end_ts)]


def _bump(stats: Optional[dict], key: str, sub: Optional[str] = None) -> None:
    if stats is None:
        return
    if sub is None:
        stats[key] = stats.get(key, 0) + 1
    else:
        d = stats.setdefault(key, {})
        d[sub] = d.get(sub, 0) + 1


def _minutes_of(m1: Optional[pd.DataFrame], ts: pd.Timestamp) -> Optional[pd.DataFrame]:
    """1m bars inside the 5m bar starting at `ts` (None if unavailable/empty)."""
    if m1 is None:
        return None
    sl = m1[(m1.index >= ts) & (m1.index < ts + pd.Timedelta(minutes=5))]
    return sl if len(sl) else None


def _hit(side: str, stop: float, target: float, hi: float, lo: float):
    """First exit hit within one (hi, lo) range — stop wins ties, conservatively."""
    if side == "long":
        if lo <= stop:
            return "stop"
        if hi >= target:
            return "target"
    else:
        if hi >= stop:
            return "stop"
        if lo <= target:
            return "target"
    return None


def run_session(cont: pd.DataFrame, session_date: str, strategy: Strategy,
                cfg: Config = Config(), *, m1: Optional[pd.DataFrame] = None,
                stats: Optional[dict] = None) -> list[Trade]:
    win = session_window(cont, session_date, cfg.window_end)
    if len(win) < 2:
        return []
    # exit frame: same bars as `win` plus the rest of the day up to flat_by; `win`
    # is a prefix of `ext` (same 09:30 start), so bar indices line up
    ext = session_window(cont, session_date, cfg.flat_by)
    if len(ext) < len(win):
        ext = win
    slip = cfg.slippage_ticks * cfg.tick
    rm = RiskManager(cfg)              # independent code-enforced risk layer (has final say)
    trades: list[Trade] = []
    i = 0
    while i < len(win) - 1:  # need a next bar to fill on
        if rm.session_halted()[0]:     # circuit breakers: max trades / consec loss / daily loss
            break
        cur_ts = win.index[i]
        sig = strategy(build_sidecar(cont, cur_ts), win.iloc[: i + 1])
        if sig is None:
            i += 1
            continue
        _bump(stats, "signals")

        cur = win.iloc[i]
        use_stop_entry = getattr(sig, "entry_type", "market") == "stop"
        if use_stop_entry:  # Brooks-style: stop order 1 tick beyond the signal bar
            entry_ref = (float(cur["high"]) + cfg.tick if sig.side == "long"
                         else float(cur["low"]) - cfg.tick)
        else:               # market / limit (approximated as market at next open)
            entry_ref = float(cur["close"])

        # pre-trade veto: geometry + per-trade $ cap + R:R floor (LLM not trusted to self-enforce)
        rd = rm.validate_entry(sig.side, entry_ref, sig.stop, sig.target)
        if not rd.ok:
            _bump(stats, "veto", rd.reason)
            i += 1
            continue
        risk_pts = abs(entry_ref - sig.stop)

        # ---- fill on bar j (stop orders work for ONE bar, else canceled) ----
        j = i + 1
        nxt = win.iloc[j]
        mins_j = _minutes_of(m1, win.index[j])
        fill_from = 0            # minute row in bar j from which exits may trigger
        if use_stop_entry:
            n_open = float(nxt["open"])
            gapped = (n_open >= entry_ref) if sig.side == "long" else (n_open <= entry_ref)
            triggered = ((float(nxt["high"]) >= entry_ref) if sig.side == "long"
                         else (float(nxt["low"]) <= entry_ref))
            if gapped:
                entry = n_open + slip if sig.side == "long" else n_open - slip
            elif triggered:
                entry = entry_ref + slip if sig.side == "long" else entry_ref - slip
                if mins_j is not None:  # locate the trigger minute: exits can't precede the fill
                    for mi in range(len(mins_j)):
                        mrow = mins_j.iloc[mi]
                        hit_trig = ((float(mrow["high"]) >= entry_ref) if sig.side == "long"
                                    else (float(mrow["low"]) <= entry_ref))
                        if hit_trig:
                            fill_from = mi
                            break
            else:                # never triggered -> order canceled, stay flat at bar j
                _bump(stats, "no_fill")
                i = j
                continue
        else:
            raw = float(nxt["open"])
            entry = raw + slip if sig.side == "long" else raw - slip

        # ---- exit scan: past the entry window if needed, minute-resolution when m1 given ----
        exit_px = exit_reason = None
        k = j
        while k < len(ext):
            mins = mins_j if k == j else _minutes_of(m1, ext.index[k])
            if mins is not None:
                start = fill_from if k == j else 0
                for mi in range(start, len(mins)):
                    mrow = mins.iloc[mi]
                    h = _hit(sig.side, sig.stop, sig.target,
                             float(mrow["high"]), float(mrow["low"]))
                    if h:
                        exit_reason = h
                        break
            else:  # no 1m data -> whole-5m-bar range, stop-first (conservative)
                exit_reason = _hit(sig.side, sig.stop, sig.target,
                                   float(ext.iloc[k]["high"]), float(ext.iloc[k]["low"]))
            if exit_reason:
                if exit_reason == "stop":
                    exit_px = sig.stop - slip if sig.side == "long" else sig.stop + slip
                else:
                    exit_px = sig.target
                break
            k += 1
        if exit_px is None:                              # force flat at end of day (no overnight)
            k = len(ext) - 1
            exit_px, exit_reason = float(ext.iloc[k]["close"]), "eod"

        pnl_pts = (exit_px - entry) if sig.side == "long" else (entry - exit_px)
        pnl_usd = pnl_pts * cfg.point_value - cfg.commission_rt
        trades.append(Trade(
            session=session_date, side=sig.side,
            entry_ts=win.index[j].strftime("%H:%M"),
            exit_ts=ext.index[k].strftime("%H:%M"),
            entry=round(entry, 2), exit=round(exit_px, 2), stop=sig.stop,
            target=sig.target, risk_pts=round(risk_pts, 2),
            pnl_pts=round(pnl_pts, 2), pnl_usd=round(pnl_usd, 2),
            r=round(pnl_pts / risk_pts, 2), exit_reason=exit_reason, reason=sig.reason))

        rm.on_trade_closed(pnl_usd)    # update circuit-breaker state
        i = k + 1
    return trades


def run(cont: pd.DataFrame, sessions: list[str], strategy: Strategy,
        cfg: Config = Config(), *, m1: Optional[pd.DataFrame] = None,
        stats: Optional[dict] = None) -> list[Trade]:
    out: list[Trade] = []
    for s in sessions:
        out.extend(run_session(cont, s, strategy, cfg, m1=m1, stats=stats))
    return out


def summarize(trades: list[Trade]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    pnl = [t.pnl_usd for t in trades]
    wins = [p for p in pnl if p > 0]
    gross_win = sum(wins)
    gross_loss = -sum(p for p in pnl if p < 0)
    eq = peak = mdd = 0.0
    for p in pnl:
        eq += p
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    return {
        "trades": n,
        "win_rate": round(len(wins) / n, 3),
        "total_usd": round(sum(pnl), 2),
        "avg_usd": round(sum(pnl) / n, 2),
        "avg_R": round(sum(t.r for t in trades) / n, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "max_drawdown_usd": round(mdd, 2),
        "exits": {r: sum(1 for t in trades if t.exit_reason == r)
                  for r in sorted({t.exit_reason for t in trades})},
    }


def toy_strategy(sidecar: dict, bars: pd.DataFrame) -> Optional[Signal]:
    """PLUMBING ONLY — no edge. Enter on a strong trend bar closing across the EMA;
    stop 1 tick beyond the bar, fixed 2R target."""
    b = sidecar["bar"]
    if b["type"] == "trend_bull" and b["close_pos_in_range"] >= 0.65 \
            and sidecar["close_vs_ema_pts"] > 0:
        stop = b["l"] - 0.25
        risk = b["c"] - stop
        return Signal("long", stop, b["c"] + 2 * risk, "bull trend bar > EMA")
    if b["type"] == "trend_bear" and b["close_pos_in_range"] <= 0.35 \
            and sidecar["close_vs_ema_pts"] < 0:
        stop = b["h"] + 0.25
        risk = stop - b["c"]
        return Signal("short", stop, b["c"] - 2 * risk, "bear trend bar < EMA")
    return None
