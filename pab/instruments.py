"""Instrument specs — single source for per-symbol trading parameters.

MES is the backtest/eval mainline (Databento data). SPY is the live paper-trading
line (Alpaca free data + paper execution): same index, ~1/10 price scale, so the
Brooks logic transfers; only the risk arithmetic and prompt role text change.

NOTE: MES's role_text must stay BYTE-IDENTICAL to the historical system prompt —
it is part of the decision-cache key for the in-flight eval runs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    symbol: str
    tick: float
    point_value_usd: float      # $ per 1.0 point of price PER POSITION (qty baked in)
    qty: int                    # contracts / shares per trade
    per_trade_risk_usd: float
    min_risk_pts: float         # micro-stop floor (costs dominate below this)
    role_text: str              # the _ROLE paragraph for the system prompt

    @property
    def per_trade_risk_pts(self) -> float:
        return self.per_trade_risk_usd / self.point_value_usd


MES = InstrumentSpec(
    name="mes",
    symbol="MES",
    tick=0.25,
    point_value_usd=5.0,
    qty=1,
    per_trade_risk_usd=75.0,
    min_risk_pts=2.0,
    role_text=(
        "You are an Al Brooks price-action trader trading 1 micro contract of MES "
        "(Micro E-mini S&P 500) on the 5-minute chart. You ENTER only during the first "
        "two hours of the US regular session (09:30-11:30 ET); an open position is then "
        "managed to its stop or target and force-closed only at the 16:00 session close "
        "(never held overnight)."
    ),
)

SPY = InstrumentSpec(
    name="spy",
    symbol="SPY",
    tick=0.01,
    point_value_usd=50.0,       # 50 shares x $1/pt
    qty=50,
    per_trade_risk_usd=75.0,
    min_risk_pts=0.2,           # ~= 2 MES pts at 1/10 price scale
    role_text=(
        "You are an Al Brooks price-action trader trading 50 shares of SPY (S&P 500 "
        "ETF) on the 5-minute chart. You ENTER only during the first two hours of the "
        "US regular session (09:30-11:30 ET); an open position is then managed to its "
        "stop or target and force-closed only at the 16:00 session close (never held "
        "overnight). Per-trade risk is capped at $75, which at 50 shares means the stop "
        "may be at most 1.5 SPY points from entry (minimum 0.2 points — tighter stops "
        "drown in costs). Your knowledge base speaks in MES points; SPY prices are "
        "~1/10 of the index, so scale point-based intuitions accordingly."
    ),
)

SPECS = {"mes": MES, "spy": SPY}


def get_spec(name: str) -> InstrumentSpec:
    return SPECS[name.lower()]
