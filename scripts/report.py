"""Full performance report over journaled sessions: capital metrics + benchmark.

Usage:
    PA_CAPITAL=5000 python scripts/report.py            # backtest journals
    PA_JOURNAL_DIR=data/journal_live python scripts/report.py   # live line
Env:
    PA_CAPITAL     account capital for capital-relative metrics (default 5000)
    PA_JOURNAL_DIR journal directory (default data/journal)
    PA_SPLIT_DATE  optional: report <=date and >date segments separately
                   (default 2026-02-18 = experience-contamination boundary)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.metrics import capital_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JDIR = ROOT / os.getenv("PA_JOURNAL_DIR", "data/journal")
CAPITAL = float(os.getenv("PA_CAPITAL", "5000"))
SPLIT = os.getenv("PA_SPLIT_DATE", "2026-02-18")


def load_trades():
    trades, sessions = [], set()
    for p in sorted(JDIR.glob("*.json")):
        if ".review." in p.name:
            continue
        j = json.loads(p.read_text("utf-8"))
        sessions.add(j["meta"]["session"])
        trades.extend(j["trades"])
    return trades, len(sessions)


def show(tag: str, trades, n_sessions: int) -> None:
    m = capital_metrics(trades, capital=CAPITAL, n_sessions=n_sessions)
    print(f"\n===== {tag} =====")
    if m.get("trades") == 0:
        print("no trades")
        return
    for section in ("utilization", "returns", "trading"):
        print(f"[{section}]")
        for k, v in m[section].items():
            print(f"  {k:<36} {v}")


def main() -> None:
    trades, n_sessions = load_trades()
    print(f"journals: {JDIR} | sessions: {n_sessions} | trades: {len(trades)} "
          f"| capital ${CAPITAL:,.0f}")
    show("ALL", trades, n_sessions)
    if SPLIT:
        a = [t for t in trades if t["session"] <= SPLIT]
        b = [t for t in trades if t["session"] > SPLIT]
        sa = len({t["session"] for t in a})
        sb = n_sessions - sa
        if a and b:
            show(f"<= {SPLIT} (in-experience segment)", a, sa)
            show(f">  {SPLIT} (clean segment)", b, sb)


if __name__ == "__main__":
    main()
