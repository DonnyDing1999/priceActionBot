"""Independent, code-enforced risk layer — the veto module.

Two jobs:
  1. Pre-trade gate (`validate_entry`): reject any proposed trade that fails geometry,
     the per-trade $ cap, or the R:R floor. The LLM is NOT trusted to self-enforce these
     (observed: it proposed sub-1R targets), so the code has the final say.
  2. Session circuit breakers (`session_halted` + `on_trade_closed`): halt the day on
     max-trades / consecutive-loss / daily-loss.

This is the backtest-time core. Live execution extends it with a broker-truth
reconciliation loop, idempotent client order ids, and a runtime kill switch.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskDecision:
    ok: bool          # True = trade allowed
    reason: str = ""  # rejection tag when ok is False


class RiskManager:
    """One instance per session. Duck-typed on `cfg` (needs point_value,
    per_trade_risk_usd, min_rr, max_trades, max_consec_loss, daily_loss_cap)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_trades = 0
        self.consec_loss = 0
        self.day_pnl = 0.0

    def session_halted(self) -> tuple[bool, str]:
        c = self.cfg
        if self.n_trades >= c.max_trades:
            return True, "max_trades"
        if self.consec_loss >= c.max_consec_loss:
            return True, "consec_loss"
        if self.day_pnl <= -c.daily_loss_cap:
            return True, "daily_loss_cap"
        return False, ""

    def validate_entry(self, side: str, entry: float, stop, target) -> RiskDecision:
        c = self.cfg
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
        return RiskDecision(True, "")

    def on_trade_closed(self, pnl_usd: float) -> None:
        self.n_trades += 1
        self.day_pnl += pnl_usd
        self.consec_loss = self.consec_loss + 1 if pnl_usd < 0 else 0
