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


def load_all_cases() -> list[dict]:
    """Full JSONL parse in file order ([] if missing). Snapshot this ONCE per run and
    feed it to select_cases so a run's experience retrieval is frozen (reproducible)."""
    if not CASES.exists():
        return []
    return [json.loads(x) for x in CASES.read_text("utf-8").splitlines() if x.strip()]


def select_cases(cases: list[dict], regime: str | tuple[str, ...] | None = None,
                 k: int = 5, before: str | None = None) -> list[dict]:
    """Most-recent k cases, preferring regime matches. `regime` may be one regime or a
    tuple (a coarse family); falls back to recent-any. `before` (YYYY-MM-DD) keeps the
    walk-forward honest: only lessons from sessions STRICTLY BEFORE that date are
    eligible — a lesson may describe its own day, never a later one."""
    rows = cases
    if before:
        rows = [r for r in rows if r.get("session", "") < before]
    if regime:
        want = (regime,) if isinstance(regime, str) else tuple(regime)
        matched = [r for r in rows if r.get("regime") in want]
        rows = matched or rows          # fall back to recent-any if none for this family
    return rows[-k:]


def read_cases(regime: str | tuple[str, ...] | None = None, k: int = 5,
               before: str | None = None) -> list[dict]:
    """Convenience: select_cases over the current on-disk library (single impl)."""
    return select_cases(load_all_cases(), regime=regime, k=k, before=before)


def as_prompt_block(cases: list[dict]) -> str:
    if not cases:
        return ""
    lines = ["PAST EXPERIENCE (your own reviewed lessons on similar setups):"]
    for c in cases:
        lines.append(f"- [{c.get('outcome')}/{c.get('regime')}] {c.get('setup')}: {c.get('note')}")
    return "\n".join(lines)
