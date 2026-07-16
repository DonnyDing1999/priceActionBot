"""Replay saved decisions through the engine WITH journaling (free, no LLM).

Reuses data/artifacts/claude_decisions.json to produce per-session journals under
data/journal/ — real fuel to build & test the daily-review agent against.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.backtest import Config, Signal, run_session  # noqa: E402
from pab.bars import load_bars  # noqa: E402
from pab.journal import Journal  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bars_5m_60d.parquet"
DEC = ROOT / "data" / "artifacts" / "claude_decisions.json"


def make_strategy(decisions: dict, journal: Journal):
    def strat(sidecar, bars):
        s, b = sidecar["session_date"], sidecar["bar_index_from_open"]
        dec = next((d for d in decisions.get(s, []) if d["bar"] == b), None)
        if dec:
            journal.record_decision(s, b, sidecar["bar_time_et"],
                {"action": dec["action"], "setup": dec["setup"], "stop": dec["stop"],
                 "target": dec["target"], "reason": dec["reason"]}, sidecar=sidecar)
            return Signal(dec["action"], float(dec["stop"]), float(dec["target"]),
                          f"{dec['setup']}: {dec['reason']}")
        journal.record_decision(s, b, sidecar["bar_time_et"], {"action": "no_trade"},
                                sidecar=sidecar)
        return None
    return strat


def main() -> None:
    decisions = json.loads(DEC.read_text("utf-8"))
    cont = load_bars(RAW)
    j = Journal(run_id="claude-subagent-replay", provider="anthropic",
                model="claude-sonnet (subagents)")
    strat = make_strategy(decisions, j)
    for s in sorted(decisions):
        j.record_trades(s, run_session(cont, s, strat, Config()))
    paths = j.save()

    print(f"wrote {len(paths)} session journals -> data/journal/")
    sample = json.loads(Path(paths[-1]).read_text("utf-8"))
    print(f"\nsample {Path(paths[-1]).name}: {sample['meta']}")
    print(f"  decisions logged: {len(sample['decisions'])} "
          f"(incl no_trade) | trades: {len(sample['trades'])}")
    if sample["trades"]:
        t = sample["trades"][0]
        print(f"  first trade: {t['side']} {t['entry_ts']}->{t['exit_ts']} "
              f"{t['pnl_usd']:+.2f}$ ({t['r']:+.2f}R) {t['exit_reason']}")


if __name__ == "__main__":
    main()
