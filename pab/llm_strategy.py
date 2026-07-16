"""Adapter: run the decision agent as a backtest Strategy.

For each flat bar the engine hands us we call the agent on the no-lookahead sidecar
and return a Signal or None. In the default NUMERIC path nothing is rendered — the
agent reads `session_bars` from the sidecar. With cfg.vision set we also render the
chart up to THAT bar only (no-lookahead) and pass the image.

A zero-cost code GATE (PA_GATE=0 to disable) skips the LLM on bars that are obvious
no-trades under the Brooks rules (deliberately minimal — when in doubt, ask the LLM).
Every call is recorded in `self.decisions` (decision | gated | error) so the runner
can journal the full day. Geometry / per-trade cap / R:R are enforced by the code
risk layer downstream, not here. Use ONE instance per session (per-session state).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from pab.agent import AgentConfig, decide
from pab.backtest import Signal


LAST_SIGNAL_BAR = 19  # a signal here fills at bar 20 (11:05) — later entries only get
                      # force-flattened at the window end with no room to work


def obvious_no_trade(sidecar: dict) -> Optional[str]:
    """Mechanical pre-filter: bars that can be skipped without asking the LLM. Kept
    minimal and conservative — when in doubt, ask the LLM. Returns a tag or None."""
    if sidecar["bar_index_from_open"] > LAST_SIGNAL_BAR:
        return "too_late_in_window"
    b = sidecar["bar"]
    hi, lo = sidecar["session_high"], sidecar["session_low"]
    rng = hi - lo
    pos = (b["c"] - lo) / rng if rng > 0 else 0.5
    if (b["type"] == "doji" and abs(sidecar["close_vs_ema_pts"]) < 2.0
            and 0.3 <= pos <= 0.7 and sidecar["bar_index_from_open"] >= 3):
        return "doji_mid_range_near_ema"
    return None


class LLMStrategy:
    def __init__(self, cont: pd.DataFrame, *, render_dir: str | Path | None = None,
                 cfg: AgentConfig | None = None, gate: Optional[bool] = None):
        self.cont = cont
        self.cfg = cfg or AgentConfig()
        self.gate = (os.getenv("PA_GATE", "1").lower() in ("1", "true", "yes")
                     if gate is None else gate)
        self.render_dir = Path(render_dir) if render_dir else Path(tempfile.gettempdir())
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.decisions: list[dict] = []  # every bar: decision | gated | error (for journal)
        self.errors = 0                  # decision calls that failed -> no_trade
        self.error_types: dict[str, int] = {}  # exception class -> count (diagnosis)
        self.gated = 0                   # bars skipped by the code gate (no LLM call)
        self.cache_hits = 0

    def _record(self, sidecar: dict, decision: dict) -> None:
        t = sidecar.get("bar_time_et", "")
        parts = t.split(" ")
        self.decisions.append({
            "bar": sidecar.get("bar_index_from_open"),
            "time": parts[1] if len(parts) > 1 else t,  # HH:MM
            "decision": decision, "sidecar": sidecar})

    def __call__(self, sidecar: dict, bars: pd.DataFrame) -> Optional[Signal]:
        if self.gate:
            tag = obvious_no_trade(sidecar)
            if tag:
                self.gated += 1
                self._record(sidecar, {"action": "no_trade", "setup": "",
                                       "reason": f"gated: {tag}"})
                return None

        image_path = None
        if self.cfg.vision:  # only render when the vision path is explicitly enabled
            from pab.render import render_session
            current_ts = bars.index[-1]
            session_date = str(current_ts.date())
            image_path = self.render_dir / f"{session_date}_{current_ts.strftime('%H%M')}.png"
            render_session(self.cont, session_date, up_to=current_ts, out_path=image_path)
        try:
            d = decide(sidecar, image_path, cfg=self.cfg)
        except Exception as e:  # noqa: BLE001 — any error on THIS bar -> no_trade, day survives
            self.errors += 1
            et = type(e).__name__
            self.error_types[et] = self.error_types.get(et, 0) + 1
            self._record(sidecar, {"action": "error",
                                   "reason": f"{et}: {str(e)[:120]}"})
            return None
        if d.get("_usage", {}).get("cached"):
            self.cache_hits += 1
        self._record(sidecar, d)
        return self._to_signal(d, sidecar)

    @staticmethod
    def _to_signal(d: dict, sidecar: dict) -> Optional[Signal]:
        # Raw LLM proposal -> Signal. Geometry / per-trade cap / R:R are enforced
        # authoritatively by the code risk layer (pab.risk.RiskManager), not here.
        if not d or d.get("action") not in ("long", "short"):
            return None
        stop, target = d.get("stop"), d.get("target")
        if stop is None or target is None:
            return None
        reason = f"{d.get('setup', '')}: {str(d.get('reason', ''))[:120]}"
        entry_type = d.get("entry_type") if d.get("entry_type") in ("stop", "limit",
                                                                    "market") else "market"
        return Signal(d["action"], float(stop), float(target), reason,
                      entry_type=entry_type)
