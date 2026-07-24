"""Single source of run-shape truth: window bounds, engine Config, journal layout,
run ids, and the day-level trade/no-trade gate.

The backtest runner and the live loop both derive their session window, engine
Config, journal directory, and day gate from here, so a window/instrument change
happens in exactly one place instead of being duplicated as if/else branches.
"""
from __future__ import annotations

from pathlib import Path

from pab.backtest import Config
from pab.dayfilter import day_features, grade, grade_llm

ROOT = Path(__file__).resolve().parents[1]

WINDOWS = {
    "am": {"start": "09:30", "end": "11:25", "flat_by": "15:55"},
    "pm": {"start": "13:30", "end": "15:25", "flat_by": "15:55"},
}


def engine_config(spec, window: str = "am", **overrides) -> Config:
    """Build a backtest Config from an InstrumentSpec + WINDOWS[window]. Config
    defaults stay MES-valued for back-compat; `overrides` win over everything."""
    w = WINDOWS[window]
    cfg = Config(tick=spec.tick, point_value=spec.point_value_usd,
                 per_trade_risk_usd=spec.per_trade_risk_usd,
                 min_risk_pts=spec.min_risk_pts,
                 window_start=w["start"], window_end=w["end"], flat_by=w["flat_by"])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def journal_dir(instrument: str = "mes", window: str = "am", *,
                live: bool = False) -> Path:
    return ROOT / "data" / "journal" / instrument / ("live" if live else window)


def run_id(provider: str, instrument: str, window: str) -> str:
    return f"bt_{provider}_{instrument}_{window}"


def day_gate(cont, session: str, *, mode: str, window: str, cfg) -> dict | None:
    """None = A-day (trade normally); a dict = C-day (skip). `mode` is the dayfilter
    variant: "off" never gates; "llm" defers to the model's own A/C judgment; anything
    else is the mechanical prior-day-texture grader."""
    if mode == "off":
        return None
    if mode == "llm":
        g = grade_llm(cont, session, window=window, cfg=cfg)
        return g if g.get("grade") == "C" else None
    return {"grade": "C"} if grade(day_features(cont, session), mode) == "C" else None
