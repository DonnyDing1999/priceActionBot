"""Full-fidelity journaling: every decision + resulting trade to disk, per session.

This is the fuel for the daily-review agent and for debugging. One JSON per session under
data/journal/<session>.json holding the ordered decisions (bar context + what the agent
decided + why) and the trades the engine actually executed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
JOURNAL_DIR = ROOT / "data" / "journal"


def _brief(sidecar: dict) -> dict:
    """Compact bar context worth keeping next to each decision."""
    b = sidecar.get("bar", {})
    return {
        "bar_type": b.get("type"),
        "close": b.get("c"),
        "close_pos": b.get("close_pos_in_range"),
        "ema20": sidecar.get("ema20"),
        "close_vs_ema_pts": sidecar.get("close_vs_ema_pts"),
        "always_in": sidecar.get("always_in_hint"),
        "session_high": sidecar.get("session_high"),
        "session_low": sidecar.get("session_low"),
        "open_gap_pts": sidecar.get("open_gap_pts"),
    }


class Journal:
    """Collects decisions + trades in memory, then writes one file per session."""

    def __init__(self, run_id: str = "run", provider: str = "", model: str = ""):
        self.run_id = run_id
        self.meta = {"run_id": run_id, "provider": provider, "model": model}
        self.sessions: dict[str, dict] = {}

    def _bucket(self, session: str) -> dict:
        return self.sessions.setdefault(session, {"decisions": [], "trades": []})

    def record_decision(self, session: str, bar: int, bar_time: str,
                        decision: dict, image: Optional[str] = None,
                        sidecar: Optional[dict] = None) -> None:
        keep = {k: decision.get(k) for k in
                ("action", "setup", "stop", "target", "confidence", "reason")}
        self._bucket(session)["decisions"].append(
            {"bar": bar, "time": bar_time, "decision": keep,
             "context": _brief(sidecar) if sidecar else None, "image": image})

    def record_trades(self, session: str, trades: list) -> None:
        self._bucket(session)["trades"].extend(
            asdict(t) if is_dataclass(t) else dict(t) for t in trades)

    def save(self) -> list[str]:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for s, rec in self.sessions.items():
            day_pnl = round(sum(t["pnl_usd"] for t in rec["trades"]), 2)
            out = {"meta": {**self.meta, "session": s,
                            "n_decisions": len(rec["decisions"]),
                            "n_trades": len(rec["trades"]), "day_pnl_usd": day_pnl},
                   **rec}
            p = JOURNAL_DIR / f"{s}.json"
            p.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
            paths.append(str(p))
        return paths
