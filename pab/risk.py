"""Independent, code-enforced risk layer — the veto module.

Two jobs:
  1. Pre-trade gate (`validate_entry`): reject any proposed trade that fails geometry,
     the per-trade $ cap, or the R:R floor. The LLM is NOT trusted to self-enforce these
     (observed: it proposed sub-1R targets), so the code has the final say. With a `ctx`
     (v2, built from the numeric sidecar) it also applies Brooks structural vetoes —
     zone, pullback second-entry, counter-trend, climax no-chase, post-stop cooldown.
  2. Session circuit breakers (`session_halted` + `on_trade_closed`): halt the day on
     max-trades / consecutive-loss / daily-loss, and remember the last stop-out so the
     cooldown veto can suppress immediate churn.

This is the backtest-time core. Live execution extends it with a broker-truth
reconciliation loop, idempotent client order ids, and a runtime kill switch.

ctx schema (shared contract; None ⇒ exactly v1 behavior, all v2 vetoes skipped):
  ctx = {
    "regime": "open"|"trend"|"range",   # features.classify_regime output
    "always_in": "ail"|"ais"|"neutral", # sidecar always_in_hint
    "zone": "lower_third"|"middle"|"upper_third",
    "price_position": float,            # 0..1 within the working range
    "working_range": {"hi","lo","src_hi","src_lo"},
    "avg_range_10": float,              # ATR proxy (mean bar range, <=10 bars)
    "hl_count": {"dir": "bull"|"bear"|"none", "count": int, ...},
    "consec_trend_bars": {"dir": "bull"|"bear"|"none", "n": int},
    "signal_bar_range": float,          # current signal bar high-low
    "bar_index": int,                   # sidecar bar_index_from_open (1-based)
    "entry_type": "stop"|"market"|"limit",  # NEW in v2 (introduced here for the
                                            # climax no-chase check; the caller sets it
                                            # from the proposed signal's entry_type)
  }
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskDecision:
    ok: bool          # True = trade allowed
    reason: str = ""  # rejection tag when ok is False


class RiskManager:
    """One instance per session. Duck-typed on `cfg` (needs point_value,
    per_trade_risk_usd, min_rr, max_trades, max_consec_loss, daily_loss_cap; the v2
    vetoes additionally read the getattr-defaulted knobs zone_veto / pullback_gate /
    no_chase / cooldown / cooldown_bars / cooldown_ticks / tick)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_trades = 0
        self.consec_loss = 0
        self.day_pnl = 0.0
        # last stop-out for the cooldown veto: {"side","entry","exit_bar_index"} or None
        self._last_stop: dict | None = None

    def session_halted(self) -> tuple[bool, str]:
        c = self.cfg
        if self.n_trades >= c.max_trades:
            return True, "max_trades"
        if self.consec_loss >= c.max_consec_loss:
            return True, "consec_loss"
        if self.day_pnl <= -c.daily_loss_cap:
            return True, "daily_loss_cap"
        return False, ""

    def validate_entry(self, side: str, entry: float, stop, target,
                       obstacles: list[float] | None = None,
                       ctx: dict | None = None) -> RiskDecision:
        c = self.cfg
        # ---- v1 geometry/$/R:R gate (unchanged). ctx=None returns here => v1-identical ----
        if side not in ("long", "short"):
            return RiskDecision(False, "bad_side")
        if stop is None or target is None:
            return RiskDecision(False, "missing_stop_or_target")
        stop, target = float(stop), float(target)
        # geometry: long -> stop < entry < target ; short -> target < entry < stop
        if side == "long" and not (stop < entry < target):
            return RiskDecision(False, "geometry")
        if side == "short" and not (target < entry < stop):
            return RiskDecision(False, "geometry")
        risk = abs(entry - stop)
        if risk <= 0:
            return RiskDecision(False, "zero_risk")
        # micro-stop floor — below ~2pts, fixed costs (commission + slippage) dominate
        # the trade's R math (observed: 0.5pt-risk trades all land ~-2R on costs alone)
        if risk < getattr(c, "min_risk_pts", 0.0):
            return RiskDecision(False, "risk_too_small")
        # per-trade $ cap — reject if stop too wide (never widen)
        if risk * c.point_value > c.per_trade_risk_usd:
            return RiskDecision(False, "risk_gt_cap")
        # R:R floor on the (near) target — reject sub-min_rr proposals
        if abs(target - entry) / risk < c.min_rr:
            return RiskDecision(False, "rr_below_floor")
        # first-obstacle R:R — a magnet strictly between entry and target caps the
        # reachable reward at that level; veto if reward to the nearest one is sub-min_rr
        magnet_veto = getattr(c, "magnet_veto", True)
        if magnet_veto and obstacles:
            if side == "long":
                between = [px for px in obstacles if entry < px < target]
            else:
                between = [px for px in obstacles if target < px < entry]
            if between:
                eff_reward = ((min(between) - entry) if side == "long"
                              else (entry - max(between)))
                if eff_reward / risk < c.min_rr:
                    return RiskDecision(False, "rr_to_magnet")
        if ctx is None:
            return RiskDecision(True, "")

        # ---- v2 structural vetoes (each behind a getattr-defaulted-ON knob, applied
        # AFTER the v1 checks so a v1 tag always wins a tie) ----
        regime = ctx.get("regime")
        always_in = ctx.get("always_in")

        # zone veto (range regime only): the middle third has no edge either way and
        # vetoes any entry; the outer thirds veto only the chase side (fades toward an
        # extreme — short in the upper third, long in the lower third — pass).
        if getattr(c, "zone_veto", True) and regime == "range":
            zone = ctx.get("zone")
            if zone == "middle":
                return RiskDecision(False, "zone_middle")
            if zone == "upper_third" and side == "long":
                return RiskDecision(False, "zone_long_upper")
            if zone == "lower_third" and side == "short":
                return RiskDecision(False, "zone_short_lower")

        # trend regime: with-trend entries need a second entry (>=2 pullback legs) unless
        # a >=3-bar spike lets an H1/L1 through; against-trend entries are blocked outright.
        if getattr(c, "pullback_gate", True) and regime == "trend":
            with_trend = ((side == "long" and always_in == "ail") or
                          (side == "short" and always_in == "ais"))
            counter = ((side == "long" and always_in == "ais") or
                       (side == "short" and always_in == "ail"))
            if with_trend:
                want = "bull" if side == "long" else "bear"
                hl = ctx.get("hl_count") or {}
                consec = ctx.get("consec_trend_bars") or {}
                hl_ok = hl.get("dir") == want and int(hl.get("count", 0)) >= 2
                spike = consec.get("dir") == want and int(consec.get("n", 0)) >= 3
                if not (hl_ok or spike):
                    return RiskDecision(False, "trend_needs_second_entry")
            elif counter:
                return RiskDecision(False, "counter_trend")

        # climax no-chase (all regimes): never STOP-enter beyond a climax bar (signal bar
        # >= 2x the recent average range). Limit/market fades of the climax are unaffected.
        if getattr(c, "no_chase", True) and ctx.get("entry_type") == "stop":
            avg = ctx.get("avg_range_10")
            sbr = ctx.get("signal_bar_range")
            if avg and sbr is not None and float(sbr) >= 2.0 * float(avg):
                return RiskDecision(False, "climax_no_chase")

        # cooldown: after a stop-out, suppress immediate re-entry churn. bars_since counts
        # bars elapsed since the stop-out bar (both indices are the 1-based sidecar
        # bar_index_from_open); the window covers the stop-out bar and the bars that
        # follow it. Opposite-side is blocked for cooldown_bars bars only when the new
        # entry sits within cooldown_ticks of the stopped entry; same-side is blocked for
        # 2 bars regardless of price.
        if getattr(c, "cooldown", True) and self._last_stop is not None:
            last = self._last_stop
            bar_index = ctx.get("bar_index")
            exit_bar = last.get("exit_bar_index")
            if bar_index is not None and exit_bar is not None:
                bars_since = int(bar_index) - int(exit_bar)
                if bars_since >= 0:
                    if side == last.get("side"):
                        if bars_since <= 2:
                            return RiskDecision(False, "cooldown_same")
                    else:
                        cd_bars = getattr(c, "cooldown_bars", 3)
                        cd_ticks = getattr(c, "cooldown_ticks", 3)
                        tick = getattr(c, "tick", 0.25)
                        le = last.get("entry")
                        if (bars_since <= cd_bars and le is not None
                                and abs(entry - float(le)) <= cd_ticks * tick):
                            return RiskDecision(False, "cooldown_opposite")

        return RiskDecision(True, "")

    def on_trade_closed(self, pnl_usd: float, *, side: str | None = None,
                        entry: float | None = None,
                        exit_bar_index: int | None = None,
                        exit_reason: str | None = None) -> None:
        # v1 circuit-breaker state (positional call keeps working for v1 callers)
        self.n_trades += 1
        self.day_pnl += pnl_usd
        self.consec_loss = self.consec_loss + 1 if pnl_usd < 0 else 0
        # v2: remember the last stop-out so the cooldown veto has something to key on
        if exit_reason == "stop":
            self._last_stop = {"side": side, "entry": entry,
                               "exit_bar_index": exit_bar_index}
