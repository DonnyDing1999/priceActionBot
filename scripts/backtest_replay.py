"""Replay pre-computed Claude decisions through the same fill + risk engine.

The per-bar decisions were produced by Claude subagents acting as the Al Brooks agent
(one per session), saved to data/artifacts/claude_decisions.json. Here we feed them into
the SAME backtest engine (next-bar-open fills, commission + slippage) and the SAME code
risk layer (geometry / per-trade cap / R:R floor / circuit breakers), so results are
apples-to-apples with the mechanical and LLM-API runs. Proposals the risk layer rejects
(RR<1, stop too wide, bad geometry) simply don't become trades.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.backtest import Config, Signal, run_session, summarize  # noqa: E402
from pab.bars import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bars_5m_60d.parquet"
DEC = ROOT / "data" / "artifacts" / "claude_decisions.json"


def make_strategy(decisions: dict):
    def strat(sidecar, bars):
        s, b = sidecar["session_date"], sidecar["bar_index_from_open"]
        for d in decisions.get(s, []):
            if d["bar"] == b:
                return Signal(d["action"], float(d["stop"]), float(d["target"]),
                              f"bar{b} {d['setup']}: {d['reason']}")
        return None
    return strat


def main() -> None:
    decisions = json.loads(DEC.read_text("utf-8"))
    cont = load_bars(RAW)
    strat = make_strategy(decisions)

    all_trades = []
    for s in sorted(decisions):
        ts = run_session(cont, s, strat, Config())
        all_trades.extend(ts)
        proposed = len(decisions[s])
        print(f"[{s}] proposed {proposed} -> filled {len(ts)}, "
              f"day pnl {sum(t.pnl_usd for t in ts):+.2f}$")
        for t in ts:
            print(f"    {t.side:<5} {t.entry_ts}->{t.exit_ts} {t.pnl_usd:+.2f}$ "
                  f"({t.r:+.2f}R) {t.exit_reason} | {t.reason[:80]}")

    print("\nSUMMARY (Claude decisions):", json.dumps(summarize(all_trades), ensure_ascii=False))


if __name__ == "__main__":
    main()
