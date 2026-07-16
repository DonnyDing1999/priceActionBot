"""Replay CACHED decisions through the real engine — zero API calls.

For each bar the engine visits, look up the on-disk decision cache using the exact
key the live agent would compute; a cache miss plays as no_trade (same as an
errored call in the live run). Because the engine, gate, risk veto, and fills are
identical, this reproduces the live run's trajectory over whatever coverage the
cache currently holds — free. Useful mid-run (partial read) and after engine/risk
changes (instant re-evaluation without re-paying the LLM).

Coverage is printed first — treat low-coverage results as directional, not final.
Env: PA_PROVIDER / PA_MODEL / PA_TEMPERATURE etc. must MATCH the run that filled
the cache, or every key misses.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.agent import CACHE_DIR, AgentConfig, _cache_key, _system, _user_text  # noqa: E402
from pab.backtest import Config, run_session, summarize  # noqa: E402
from pab.bars import complete_sessions, load_bars  # noqa: E402
from pab.llm_strategy import LLMStrategy, obvious_no_trade  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / os.getenv("PA_BARS", "mes_5m.parquet")
RAW_1M = ROOT / "data" / "raw" / "mes_1m.parquet"


class CacheOnlyStrategy:
    """LLMStrategy's decision path, but the only 'provider' is the disk cache."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.system = _system(cfg.vision)
        self.hits = 0
        self.misses = 0
        self.gated = 0

    def __call__(self, sidecar, bars):
        if obvious_no_trade(sidecar):          # same gate as the live run
            self.gated += 1
            return None
        key = _cache_key(self.cfg, self.system, _user_text(sidecar), None)
        f = CACHE_DIR / f"{key}.json"
        if not f.exists():
            self.misses += 1                   # errored/unreached in live run -> no_trade
            return None
        self.hits += 1
        d = json.loads(f.read_text("utf-8"))
        return LLMStrategy._to_signal(d, sidecar)


def main() -> None:
    cfg = AgentConfig()
    cont = load_bars(RAW)
    m1 = load_bars(RAW_1M) if RAW_1M.exists() else None
    sessions = complete_sessions(cont)[-int(os.getenv("N_SESSIONS", "128")):]

    strat = CacheOnlyStrategy(cfg)
    stats: dict = {}
    trades = []
    per_session_hits: dict[str, int] = {}
    for s in sessions:
        before = strat.hits
        trades.extend(run_session(cont, s, strat, Config(), m1=m1, stats=stats))
        per_session_hits[s] = strat.hits - before

    asked = strat.hits + strat.misses
    covered = sum(1 for v in per_session_hits.values() if v > 0)
    print(f"cache replay — provider={cfg.provider} model={cfg.resolved_model()} "
          f"temp={cfg.temperature} m1={'on' if m1 is not None else 'off'}")
    print(f"coverage: {strat.hits}/{asked} visited bars had cached decisions "
          f"({100 * strat.hits / max(1, asked):.0f}%), gated={strat.gated}, "
          f"sessions with any coverage: {covered}/{len(sessions)}")
    print(f"engine stats: {stats}")
    print("SUMMARY:", json.dumps(summarize(trades), ensure_ascii=False))
    print()
    for t in trades:
        print(f"  {t.session} {t.side:<5} {t.entry_ts}->{t.exit_ts} "
              f"{t.pnl_usd:+8.2f}$ ({t.r:+.2f}R) {t.exit_reason} | {t.reason[:70]}")


if __name__ == "__main__":
    main()
