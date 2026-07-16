"""Experience library: regime-partitioned cases the review agent WRITES and the decision
agent can READ back. Stored as JSONL (one case per line) at experience/cases.jsonl.

This is the automated version of PA_Agent's read-only experience/ folders: the daily-review
agent promotes durable lessons into cases here; the decision agent retrieves the ones matching
the current regime to steer future decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "experience" / "cases.jsonl"


def add_cases(cases: list[dict], session: str) -> int:
    CASES.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with CASES.open("a", encoding="utf-8") as f:
        for c in cases:
            rec = {"session": session,
                   "regime": c.get("regime", "unknown"),
                   "outcome": c.get("outcome", "unknown"),
                   "setup": c.get("setup", ""),
                   "note": c.get("note", "")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_cases(regime: str | None = None, k: int = 5) -> list[dict]:
    if not CASES.exists():
        return []
    rows = [json.loads(x) for x in CASES.read_text("utf-8").splitlines() if x.strip()]
    if regime:
        matched = [r for r in rows if r.get("regime") == regime]
        rows = matched or rows          # fall back to recent-any if none for this regime
    return rows[-k:]


def as_prompt_block(cases: list[dict]) -> str:
    if not cases:
        return ""
    lines = ["PAST EXPERIENCE (your own reviewed lessons on similar setups):"]
    for c in cases:
        lines.append(f"- [{c.get('outcome')}/{c.get('regime')}] {c.get('setup')}: {c.get('note')}")
    return "\n".join(lines)
