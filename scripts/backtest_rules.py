"""Mechanical Brooks-rules backtest — NO LLM, runs fully in-sandbox (free, instant).

A code approximation of the with-trend opening setups from the numeric sidecar:
only trade in the always-in direction, on a strong trend bar closing near its extreme,
on the correct side of the 20-EMA; stop beyond the bar, fixed 2R target. The code risk
layer (geometry / per-trade cap / R:R / circuit breakers) gates every proposal.

This is a rules baseline, NOT the LLM-vision approach (that reads the chart image and
needs an external LLM API). Useful as a free, deterministic comparison point.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.backtest import Config, Signal, run_session, summarize  # noqa: E402
from pab.bars import complete_sessions, load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# default to the clean Databento MES 5m data; override with PA_BARS=<file>
RAW = ROOT / "data" / "raw" / os.getenv("PA_BARS", "mes_5m.parquet")
RAW_1M = ROOT / "data" / "raw" / "mes_1m.parquet"


def brooks_rules(sidecar: dict, bars) -> "Signal | None":
    idx = sidecar["bar_index_from_open"]
    if not (2 <= idx <= 20):          # trade the meat of the opening window
        return None
    b = sidecar["bar"]
    ai = sidecar["always_in_hint"]
    ema_gap = sidecar["close_vs_ema_pts"]
    cp = b["close_pos_in_range"]
    if ai == "ail" and b["type"] == "trend_bull" and cp >= 0.6 and ema_gap > 0:
        stop = b["l"] - 0.25
        risk = b["c"] - stop
        return Signal("long", stop, b["c"] + 2 * risk, "rule: AIL bull trend bar > EMA")
    if ai == "ais" and b["type"] == "trend_bear" and cp <= 0.4 and ema_gap < 0:
        stop = b["h"] + 0.25
        risk = stop - b["c"]
        return Signal("short", stop, b["c"] - 2 * risk, "rule: AIS bear trend bar < EMA")
    return None


def main() -> None:
    cont = load_bars(RAW)
    m1 = load_bars(RAW_1M) if RAW_1M.exists() else None
    sessions = complete_sessions(cont)

    all_trades = []
    stats: dict = {}
    for s in sessions:
        all_trades.extend(run_session(cont, s, brooks_rules, Config(), m1=m1, stats=stats))

    print(f"mechanical Brooks-rules backtest — {len(sessions)} sessions, NO LLM (in-sandbox), "
          f"m1={'on' if m1 is not None else 'off'} | stats={stats}")
    print("SUMMARY:", json.dumps(summarize(all_trades), ensure_ascii=False))
    wins = [t.pnl_usd for t in all_trades if t.pnl_usd > 0]
    print(f"avg win ${sum(wins)/len(wins):.1f} | " if wins else "no wins | ",
          f"avg loss ${sum(t.pnl_usd for t in all_trades if t.pnl_usd<=0)/max(1,len(all_trades)-len(wins)):.1f}")


if __name__ == "__main__":
    main()
