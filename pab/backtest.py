"""Event-driven, bar-by-bar backtester (strict no-lookahead) for the opening window.

Backbone of the system. For each session it walks the 24 opening-window bars in order.
At each flat bar it builds the numeric sidecar from bars <= current only and calls a
Strategy. Entry honors the signal's `entry_type`:
  - "stop" (the Brooks default): a stop order 1 tick beyond the signal bar's extreme,
    working for the NEXT bar only — if that bar never triggers it, the order is
    canceled and no trade happens (failed signals filter themselves out).
  - "market": fill at the next bar's open.
  - "limit": a REST order at the signal's `entry_px`, working bars j..j+limit_ttl_bars-1.
    It fills conservatively and with NO slippage — only when a working bar/minute trades
    THROUGH the limit by >=1 tick (long: low <= entry_px-tick; short: high >= entry_px+
    tick), at entry_px; or, if a working bar OPENS through the limit, at that (better)
    open (price improvement). It is canceled if the TTL elapses unfilled ("limit_expired"),
    the session runs out of bars / halts ("limit_halted"), or the `cancel_if` hook fires
    ("limit_canceled", wired in WP2). A "limit" with no entry_px degrades to a market
    entry and counts "limit_missing_px".
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
from pab.features import build_sidecar, classify_regime
from pab.risk import RiskManager

Strategy = Callable[[dict, pd.DataFrame], Optional["Signal"]]


@dataclass
class Config:
    tick: float = 0.25
    point_value: float = 5.0          # MES $/pt
    commission_rt: float = 1.5        # round-turn $
    slippage_ticks: int = 1           # adverse, per fill (entry + stop)
    per_trade_risk_usd: float = 75.0
    min_risk_pts: float = 2.0         # reject micro-stops: commission+slippage (~0.8pt)
                                      # would dominate the R math below this
    min_rr: float = 1.0               # reward:risk floor on the near target
    magnet_veto: bool = True          # veto if a magnet sits between entry and target
                                      # capping the first-obstacle reward below min_rr
    # WP2 structural vetoes (RiskManager v2, active only when a sidecar-v2 ctx is built)
    zone_veto: bool = True            # range regime: no chase into the wrong third
    pullback_gate: bool = True        # trend regime: second-entry gate + counter-trend block
    no_chase: bool = True             # all regimes: no stop-entry beyond a climax bar
    cooldown: bool = True             # suppress re-entry churn right after a stop-out
    cooldown_bars: int = 3            # opposite-side cooldown length (bars)
    cooldown_ticks: int = 3           # opposite-side cooldown price band (ticks)
    max_trades: int = 3
    max_consec_loss: int = 2
    daily_loss_cap: float = 500.0
    window_start: str = "09:30"       # first ENTRY-window bar (afternoon session: 13:30)
    window_end: str = "11:25"         # last ENTRY bar (afternoon session: 15:25)
    flat_by: str = "15:55"            # open trades may run past the window to their stop/
                                      # target; force-flat at this bar's close (no overnight)
    decision_latency_s: float = 0.0   # live parity: seconds after the signal bar closes
                                      # before the order is WORKING (LLM think time + order
                                      # placement). Rounded UP to whole minutes; needs m1
                                      # (without m1 it degrades to 0 = instant, as before)
    limit_ttl_bars: int = 3           # a resting limit (entry_type "limit") works this many
                                      # bars before it is canceled unfilled ("limit_expired")


@dataclass
class Signal:
    side: str            # "long" | "short"
    stop: float
    target: float
    reason: str = ""
    entry_type: str = "market"   # "stop" (1 tick beyond signal bar) | "market" | "limit"
    entry_px: float | None = None  # resting price; REQUIRED for entry_type "limit", else ignored


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
                   window_end: str = "11:25",
                   window_start: str = "09:30") -> pd.DataFrame:
    open_ts = pd.Timestamp(f"{session_date} {window_start}", tz=ET)
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


def _outcome(stats: Optional[dict], ts: pd.Timestamp, stage: str, node: str) -> None:
    """WP5 terminal taxonomy: record what the engine did with the signal on this bar,
    keyed by the SIGNAL bar's HH:MM (unique per session) so the runner can stamp the
    matching journaled decision. Purely additive under stats["outcomes"] — never touches
    trade logic or the existing counters."""
    if stats is None:
        return
    stats.setdefault("outcomes", []).append(
        {"time": ts.strftime("%H:%M"), "stage": stage, "node": node})


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


def _fill_limit(ext: pd.DataFrame, m1: Optional[pd.DataFrame], j: int, side: str,
                entry_px: float, cfg: Config, lat_min: int,
                mins_j: Optional[pd.DataFrame], rm: RiskManager,
                cancel_if: Optional[Callable], stats: Optional[dict]):
    """Work a resting limit at `entry_px` over bars j..j+cfg.limit_ttl_bars-1 (bounded by
    the session frame `ext`). Conservative, NO slippage:
      - trades THROUGH by >=1 tick (long: low <= px-tick; short: high >= px+tick) -> fill
        AT entry_px;
      - OPENS through (long: open < px; short: open > px) -> fill at that (better) open.
    decision_latency delays activation to `lat_min` minutes into bar j (needs m1); if that
    is past bar j the order simply starts working on bar j+1. Cancel precedence, per
    working bar: `cancel_if(ts)` hook -> session-halt; after the loop, TTL vs short-session.
    Returns (k_fill, fill_from_minute, fill_px) on a fill, else (None, resume_i, None) on
    cancel (bumping the matching stats counter). `fill_from_minute` is the minute row in
    the fill bar from which the exit scan may start (0 when that bar has no m1)."""
    tick = cfg.tick
    ttl_last = j + cfg.limit_ttl_bars
    last = min(ttl_last, len(ext))                 # working bars are j .. last-1
    for k in range(j, last):
        ts = ext.index[k]
        if cancel_if is not None and cancel_if(ts):     # WP2 regime-flip hook (else no-op)
            _bump(stats, "limit_canceled")
            return None, k + 1, None
        if rm.session_halted()[0]:                      # circuit breaker tripped mid-work
            _bump(stats, "limit_halted")
            return None, k + 1, None
        mins = mins_j if k == j else _minutes_of(m1, ts)
        start = lat_min if k == j else 0
        if mins is not None:
            if start >= len(mins):                      # activation falls after bar j ->
                continue                                # the order works from the next bar
            a_open = float(mins.iloc[start]["open"])
            if (a_open < entry_px) if side == "long" else (a_open > entry_px):
                return k, start, a_open                 # opened through -> price improvement
            for mi in range(start, len(mins)):
                mrow = mins.iloc[mi]
                through = ((float(mrow["low"]) <= entry_px - tick) if side == "long"
                           else (float(mrow["high"]) >= entry_px + tick))
                if through:
                    return k, mi, entry_px
        else:
            o = float(ext.iloc[k]["open"])
            if (o < entry_px) if side == "long" else (o > entry_px):
                return k, 0, o
            through = ((float(ext.iloc[k]["low"]) <= entry_px - tick) if side == "long"
                       else (float(ext.iloc[k]["high"]) >= entry_px + tick))
            if through:
                return k, 0, entry_px
    # exhausted the working window without a fill
    _bump(stats, "limit_expired" if last >= ttl_last else "limit_halted")
    return None, last, None


def run_session(cont: pd.DataFrame, session_date: str, strategy: Strategy,
                cfg: Config = Config(), *, m1: Optional[pd.DataFrame] = None,
                stats: Optional[dict] = None,
                cancel_if: Optional[Callable] = None) -> list[Trade]:
    win = session_window(cont, session_date, cfg.window_end, cfg.window_start)
    if len(win) < 2:
        return []
    # exit frame: same bars as `win` plus the rest of the day up to flat_by; `win`
    # is a prefix of `ext` (same window start), so bar indices line up
    ext = session_window(cont, session_date, cfg.flat_by, cfg.window_start)
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
        sidecar = build_sidecar(cont, cur_ts, window_start=cfg.window_start)
        sig = strategy(sidecar, win.iloc[: i + 1])
        if sig is None:
            i += 1
            continue
        _bump(stats, "signals")

        cur = win.iloc[i]
        entry_type = getattr(sig, "entry_type", "market")
        entry_px = getattr(sig, "entry_px", None)
        use_stop_entry = entry_type == "stop"
        use_limit_entry = entry_type == "limit" and entry_px is not None
        if entry_type == "limit" and entry_px is None:
            _bump(stats, "limit_missing_px")   # no resting price -> degrade to market
        if use_stop_entry:  # Brooks-style: stop order 1 tick beyond the signal bar
            entry_ref = (float(cur["high"]) + cfg.tick if sig.side == "long"
                         else float(cur["low"]) - cfg.tick)
        elif use_limit_entry:   # risk-validate the limit at its resting price
            entry_ref = float(entry_px)
        else:               # market (or a limit lacking entry_px -> treated as market)
            entry_ref = float(cur["close"])

        # pre-trade veto: geometry + per-trade $ cap + R:R floor + first-obstacle R:R
        # (LLM not trusted to self-enforce). Obstacles = magnet levels + session extremes;
        # a session extreme the signal bar itself just printed is not a forward obstacle
        # (the entry fills on the NEXT bar), so it never counts against its own reward.
        obstacles = [m["px"] for m in sidecar.get("magnets", [])]
        own = sidecar.get("bar", {})
        own_extreme = {"session_high": own.get("h"), "session_low": own.get("l")}
        for k in ("session_high", "session_low", "ema20"):
            v = sidecar.get(k)
            if v is not None and v != own_extreme.get(k):
                obstacles.append(float(v))
        # WP2: structural-veto ctx from the numeric sidecar (v2 keys only). With
        # PA_SIDECAR_V2=0 the sidecar carries no "zone" -> ctx stays None -> RiskManager
        # falls back to exactly its v1 gate (legacy-replay parity).
        ctx = None
        if sidecar.get("zone") is not None:
            b = sidecar.get("bar", {})
            ctx = {
                "regime": classify_regime(
                    sidecar, window=("am" if cfg.window_start == "09:30" else "pm")),
                "always_in": sidecar.get("always_in_hint"),
                "zone": sidecar.get("zone"),
                "price_position": sidecar.get("price_position"),
                "working_range": sidecar.get("working_range"),
                "avg_range_10": sidecar.get("avg_range_10"),
                "hl_count": sidecar.get("hl_count"),
                "consec_trend_bars": sidecar.get("consec_trend_bars"),
                "signal_bar_range": (round(float(b["h"]) - float(b["l"]), 2)
                                     if "h" in b and "l" in b else None),
                "bar_index": sidecar.get("bar_index_from_open"),
                "entry_type": entry_type,           # the proposed signal's entry_type
            }
        rd = rm.validate_entry(sig.side, entry_ref, sig.stop, sig.target,
                               obstacles=obstacles, ctx=ctx)
        if not rd.ok:
            _bump(stats, "veto", rd.reason)
            _outcome(stats, cur_ts, "risk", rd.reason)   # WP5 terminal (stage risk)
            i += 1
            continue
        risk_pts = abs(entry_ref - sig.stop)

        # ---- fill (stop: works bar j only; limit: rests j..j+ttl-1; market: next open) ----
        j = i + 1
        nxt = win.iloc[j]
        mins_j = _minutes_of(m1, win.index[j])
        # live parity: the order only becomes WORKING lat_min minutes into bar j
        # (decision latency, ceil to minutes); needs m1 for sub-bar activation
        lat_min = 0
        if cfg.decision_latency_s > 0 and mins_j is not None:
            lat_min = int(-(-cfg.decision_latency_s // 60))
        fill_from = lat_min      # minute row in bar j from which fills/exits may trigger
        entry_bar_ts = win.index[j]
        if use_limit_entry:      # resting limit works bars j..j+limit_ttl_bars-1
            # WP2: cancel a resting limit if the regime flips out from under it. The
            # closure rebuilds the sidecar at each working bar and compares its regime to
            # the regime at signal time; any exception -> don't cancel. An explicit
            # `cancel_if` (WP3 hook / tests) still fires, OR-composed and checked first.
            limit_cancel = cancel_if
            if ctx is not None:
                r0 = ctx["regime"]
                win_kind = "am" if cfg.window_start == "09:30" else "pm"

                def limit_cancel(ts, _r0=r0, _wk=win_kind, _outer=cancel_if):
                    if _outer is not None and _outer(ts):
                        return True
                    try:
                        sc2 = build_sidecar(cont, ts, window_start=cfg.window_start)
                        return classify_regime(sc2, window=_wk) != _r0
                    except Exception:
                        return False
            res = _fill_limit(ext, m1, j, sig.side, float(entry_px), cfg,
                              lat_min, mins_j, rm, limit_cancel, stats)
            if res[0] is None:   # canceled (TTL / session-halt / cancel_if): resume past
                _outcome(stats, cur_ts, "engine", "no_fill")   # WP5 terminal (limit cancel)
                i = res[1]       # the working bars the fill scan just consumed
                continue
            j, fill_from, entry = res       # fill bar, its fill minute, fill price (no slip)
            mins_j = _minutes_of(m1, ext.index[j])
            entry_bar_ts = ext.index[j]
        elif use_stop_entry:
            if lat_min > 0 and mins_j is not None and lat_min >= len(mins_j):
                _bump(stats, "no_fill")   # decision arrived after bar j ended -> canceled
                _outcome(stats, cur_ts, "engine", "no_fill")   # WP5 terminal
                i = j
                continue
            if lat_min > 0 and mins_j is not None:
                act = mins_j.iloc[lat_min:]              # minutes the order is live for
                a_open = float(act.iloc[0]["open"])
                gapped = ((a_open >= entry_ref) if sig.side == "long"
                          else (a_open <= entry_ref))
                triggered = ((float(act["high"].max()) >= entry_ref) if sig.side == "long"
                             else (float(act["low"].min()) <= entry_ref))
            else:
                a_open = float(nxt["open"])
                gapped = ((a_open >= entry_ref) if sig.side == "long"
                          else (a_open <= entry_ref))
                triggered = ((float(nxt["high"]) >= entry_ref) if sig.side == "long"
                             else (float(nxt["low"]) <= entry_ref))
            if gapped:
                entry = a_open + slip if sig.side == "long" else a_open - slip
            elif triggered:
                entry = entry_ref + slip if sig.side == "long" else entry_ref - slip
                if mins_j is not None:  # locate the trigger minute: exits can't precede the fill
                    for mi in range(lat_min, len(mins_j)):
                        mrow = mins_j.iloc[mi]
                        hit_trig = ((float(mrow["high"]) >= entry_ref) if sig.side == "long"
                                    else (float(mrow["low"]) <= entry_ref))
                        if hit_trig:
                            fill_from = mi
                            break
            else:                # never triggered -> order canceled, stay flat at bar j
                _bump(stats, "no_fill")
                _outcome(stats, cur_ts, "engine", "no_fill")   # WP5 terminal
                i = j
                continue
        else:
            if lat_min > 0 and mins_j is not None:
                mi0 = min(lat_min, len(mins_j) - 1)
                raw = (float(mins_j.iloc[mi0]["open"]) if lat_min < len(mins_j)
                       else float(nxt["close"]))       # decided after bar j: ~bar close
                fill_from = mi0
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
            entry_ts=entry_bar_ts.strftime("%H:%M"),
            exit_ts=ext.index[k].strftime("%H:%M"),
            entry=round(entry, 2), exit=round(exit_px, 2), stop=sig.stop,
            target=sig.target, risk_pts=round(risk_pts, 2),
            pnl_pts=round(pnl_pts, 2), pnl_usd=round(pnl_usd, 2),
            r=round(pnl_pts / risk_pts, 2), exit_reason=exit_reason, reason=sig.reason))
        _outcome(stats, cur_ts, "engine", "filled")     # WP5 terminal (stage engine)

        # circuit-breaker state + cooldown memory. exit_bar_index uses the 1-based sidecar
        # bar_index_from_open convention: k is the 0-based position of the exit bar in
        # `ext` (which shares win's window start), so bar_index_from_open == k + 1.
        rm.on_trade_closed(pnl_usd, side=sig.side, entry=entry,
                           exit_bar_index=k + 1, exit_reason=exit_reason)
        i = k + 1
    return trades


def run(cont: pd.DataFrame, sessions: list[str], strategy: Strategy,
        cfg: Config = Config(), *, m1: Optional[pd.DataFrame] = None,
        stats: Optional[dict] = None,
        cancel_if: Optional[Callable] = None) -> list[Trade]:
    out: list[Trade] = []
    for s in sessions:
        out.extend(run_session(cont, s, strategy, cfg, m1=m1, stats=stats,
                               cancel_if=cancel_if))
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
