"""Adapter: run the decision agent as a backtest Strategy.

For each flat bar the engine hands us we call the agent on the no-lookahead sidecar
and return a Signal or None. Perception is NUMERIC — nothing is rendered; the agent
reads the developing session out of the sidecar's `session_table`.

A zero-cost code GATE (PA_GATE=0 to disable) skips the LLM on bars that are obvious
no-trades under the Brooks rules (deliberately minimal — when in doubt, ask the LLM).
Every call is recorded in `self.decisions` (decision | gated | error) so the runner
can journal the full day. Geometry / per-trade cap / R:R are enforced by the code
risk layer downstream, not here. Use ONE instance per session (per-session state).
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from pab.agent import AgentConfig, decide
from pab.backtest import Signal
from pab.experience import load_all_cases


class RunAborted(RuntimeError):
    """Raised when decide() fails on 4 consecutive bars — a quota/auth outage, not a
    per-bar blip. The runner catches this to abort the whole run loudly instead of
    silently journaling every remaining bar as an error."""


LAST_SIGNAL_BAR = 19  # a signal here fills at bar 20 (11:05). Entries stay in the opening
                      # structure (our edge); open positions may run PAST the window to
                      # their stop/target (engine force-flats only at end of day)


def terminal_for(decision: dict) -> dict:
    """WP5 terminal taxonomy: tag each recorded decision with the stage it TERMINATED at
    and a short node/label, so the journal (and veto_replay) can bucket outcomes without
    re-parsing free-text reasons. Stages: day_gate | code_gate | llm | validator | risk |
    engine. This function covers the stages knowable at decision-record time (day_gate is
    emitted by the runner; risk/engine outcomes are stamped by the runner from engine
    stats). label is capped at 60 chars per the contract."""
    action = decision.get("action")
    reason = str(decision.get("reason", "") or "")
    if action == "error":
        return {"stage": "llm", "node": "error", "label": ("error: " + reason)[:60]}
    tr = decision.get("terminal_reason")            # WP4 validator/retry downgrade
    if tr:
        return {"stage": "validator", "node": str(tr),
                "label": ("validator: " + str(tr))[:60]}
    if reason.startswith("day_gate:"):              # runner C-day skip routed through here
        return {"stage": "day_gate", "node": "C", "label": reason[:60]}
    if reason.startswith("gated:"):                 # obvious_no_trade code gate
        tag = reason.split("gated:", 1)[1].strip() or "gated"
        return {"stage": "code_gate", "node": tag, "label": ("code gate: " + tag)[:60]}
    if action in ("long", "short"):                 # LLM proposed a trade (risk/engine next)
        label = f"{action} {decision.get('setup', '') or ''} " \
                f"{decision.get('entry_type', '') or ''}".strip()
        return {"stage": "llm", "node": "proposed", "label": label[:60]}
    return {"stage": "llm", "node": "no_setup", "label": (reason or "no_trade")[:60]}


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
    def __init__(self, cont: pd.DataFrame, *,
                 cfg: AgentConfig | None = None, gate: Optional[bool] = None,
                 frozen_cases: list | None = None):
        self.cont = cont
        self.cfg = cfg or AgentConfig()
        self.gate = (os.getenv("PA_GATE", "1").lower() in ("1", "true", "yes")
                     if gate is None else gate)
        # freeze the experience library for this run (reproducible); snapshot once here
        # if the caller didn't hand one down
        self.frozen_cases = load_all_cases() if frozen_cases is None else frozen_cases
        self.decisions: list[dict] = []  # every bar: decision | gated | error (for journal)
        self.errors = 0                  # decision calls that failed -> no_trade
        self.error_types: dict[str, int] = {}  # exception class -> count (diagnosis)
        self.gated = 0                   # bars skipped by the code gate (no LLM call)
        self.cache_hits = 0
        self.consec_failures = 0         # reset on any success or gated bar
        self.last_error = ""             # for the RunAborted message

    def _record(self, sidecar: dict, decision: dict) -> None:
        t = sidecar.get("bar_time_et", "")
        parts = t.split(" ")
        # WP5: stamp the terminal taxonomy additively (copy so decide()'s dict is untouched).
        # The runner may later OVERRIDE this with the downstream risk/engine outcome.
        decision = {**decision, "terminal": terminal_for(decision)}
        self.decisions.append({
            "bar": sidecar.get("bar_index_from_open"),
            "time": parts[1] if len(parts) > 1 else t,  # HH:MM
            "decision": decision, "sidecar": sidecar})

    def __call__(self, sidecar: dict, bars: pd.DataFrame) -> Optional[Signal]:
        if self.gate:
            tag = obvious_no_trade(sidecar)
            if tag:
                self.gated += 1
                self.consec_failures = 0     # a reachable bar -> not a run-wide outage
                self._record(sidecar, {"action": "no_trade", "setup": "",
                                       "reason": f"gated: {tag}"})
                return None

        try:
            d = decide(sidecar, cfg=self.cfg, cases=self.frozen_cases)
        except Exception as e:  # noqa: BLE001 — any error on THIS bar -> no_trade, day survives
            self.errors += 1
            et = type(e).__name__
            self.error_types[et] = self.error_types.get(et, 0) + 1
            self.consec_failures += 1
            self.last_error = f"{et}: {str(e)[:160]}"
            self._record(sidecar, {"action": "error",
                                   "reason": f"{et}: {str(e)[:120]}"})
            if self.consec_failures >= 4:    # quota/auth outage -> abort the whole run
                raise RunAborted(self.last_error)
            return None
        self.consec_failures = 0             # a success clears the streak
        if d.get("_usage", {}).get("cached"):
            self.cache_hits += 1
        self._record(sidecar, d)
        return self._to_signal(d, sidecar)

    @staticmethod
    def _to_signal(d: dict, sidecar: dict) -> Optional[Signal]:
        # Raw LLM proposal -> Signal. Geometry / per-trade cap / R:R are enforced
        # authoritatively by the code risk layer (pab.risk.RiskManager), not here.
        # A validator-downgraded dict has action=="no_trade" (+ terminal_reason), so it
        # returns None here; _record() already journals the full decision dict, so the
        # terminal_reason flows into the journal unchanged.
        if not d or d.get("action") not in ("long", "short"):
            return None
        stop, target = d.get("stop"), d.get("target")
        if stop is None or target is None:
            return None
        reason = f"{d.get('setup', '')}: {str(d.get('reason', ''))[:120]}"
        entry_type = d.get("entry_type") if d.get("entry_type") in ("stop", "limit",
                                                                    "market") else "market"
        epx = d.get("entry_px")
        epx = float(epx) if isinstance(epx, (int, float)) and not isinstance(epx, bool) else None
        try:  # WP3 adds Signal.entry_px; stay compatible before that change lands
            return Signal(d["action"], float(stop), float(target), reason,
                          entry_type=entry_type, entry_px=epx)
        except TypeError:
            return Signal(d["action"], float(stop), float(target), reason,
                          entry_type=entry_type)
