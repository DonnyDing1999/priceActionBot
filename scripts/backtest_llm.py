"""Run the LLM decision agent over the LAST N complete opening sessions — in parallel.

Provider/model from .env (PA_PROVIDER / PA_MODEL). Env knobs:
    N_SESSIONS = how many most-recent sessions (default 3)
    PA_WORKERS = concurrent sessions (default 3; launches are still globally spaced by
                 ZHIPU_MIN_INTERVAL, but long response waits overlap)
    PA_BARS    = 5m parquet under data/raw/ (default mes_5m.parquet)
    PA_CACHE/PA_GATE/PA_TEMPERATURE — see pab.agent / pab.llm_strategy

Per session: fresh LLMStrategy (per-session state) + engine stats; every decision
(incl. gated bars and errored calls) plus the executed trades are journaled to
data/journal/<session>.json — the same files the daily-review agent consumes.
1-minute bars (data/raw/mes_1m.parquet) are used for intrabar fill/exit resolution
when present.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.agent import AgentConfig, key_status  # noqa: E402 (loads .env)
from pab.backtest import Config, run_session, summarize  # noqa: E402
from pab.bars import complete_sessions, load_bars  # noqa: E402
from pab.dayfilter import day_features, grade  # noqa: E402
from pab.journal import Journal  # noqa: E402
from pab.llm_strategy import LLMStrategy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / os.getenv("PA_BARS", "mes_5m.parquet")
RAW_1M = ROOT / "data" / "raw" / "mes_1m.parquet"


def main() -> None:
    st = key_status()
    if not st["ok"]:
        print(f"No API key for provider '{st['provider']}'. Put {st['hint']} in .env "
              "(or set PA_PROVIDER=gemini for the free option).")
        return

    n = int(os.getenv("N_SESSIONS", "3"))
    workers = int(os.getenv("PA_WORKERS", "3"))
    dayfilter = os.getenv("PA_DAYFILTER", "overlap")   # off | overlap | efficiency | combo
    cfg = AgentConfig()

    cont = load_bars(RAW)
    m1 = load_bars(RAW_1M) if RAW_1M.exists() else None
    sessions = complete_sessions(cont)[-n:]

    print(f"LLM backtest — provider={cfg.provider} model={cfg.resolved_model()} "
          f"temp={cfg.temperature} cache={'on' if cfg.cache else 'off'} "
          f"m1={'on' if m1 is not None else 'off'} workers={workers} "
          f"dayfilter={dayfilter}")
    print(f"{len(sessions)} sessions {sessions}\n", flush=True)

    def run_one(s: str):
        # day-level chop gate: a C-graded day is skipped entirely (no LLM calls) —
        # Brooks' 'most days are not worth trading', validated by replay experiment
        if dayfilter != "off" and grade(day_features(cont, s), dayfilter) == "C":
            return s, [], None, {"day_grade": "C"}
        strat = LLMStrategy(cont, cfg=cfg)   # one instance per session (own state)
        stats: dict = {}
        trades = run_session(cont, s, strat, Config(), m1=m1, stats=stats)
        return s, trades, strat, stats

    journal = Journal(run_id=f"bt_{cfg.provider}", provider=cfg.provider,
                      model=cfg.resolved_model())
    all_trades, err_total, gate_total, hit_total = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, s): s for s in sessions}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                s, trades, strat, stats = fut.result()
            except Exception as e:  # noqa: BLE001 — a whole-session failure
                print(f"[{s}] ERROR: {type(e).__name__}: {str(e)[:160]} — skipping",
                      flush=True)
                continue
            if strat is None:            # C-graded day: journal the skip, zero trades
                journal.record_decision(s, 0, "09:30",
                                        {"action": "no_trade",
                                         "reason": f"day_gate: C ({dayfilter})"})
                journal.save()
                print(f"[{s}] C-DAY skipped (day gate)", flush=True)
                continue
            all_trades.extend(trades)
            err_total += strat.errors
            gate_total += strat.gated
            hit_total += strat.cache_hits
            for d in strat.decisions:
                journal.record_decision(s, d["bar"], d["time"], d["decision"],
                                        sidecar=d["sidecar"])
            journal.record_trades(s, trades)
            journal.save()   # flush incrementally so a mid-run check can read journals
            pnl = sum(t.pnl_usd for t in trades)
            veto = stats.get("veto", {})
            err = f" {strat.error_types}" if strat.error_types else ""
            print(f"[{s}] {len(trades)} trades, pnl {pnl:+.2f}$ | "
                  f"signals={stats.get('signals', 0)} no_fill={stats.get('no_fill', 0)} "
                  f"veto={veto if veto else '{}'} gated={strat.gated} "
                  f"errors={strat.errors}{err} cache_hits={strat.cache_hits}", flush=True)
            for t in trades:
                print(f"    {t.side:<5} {t.entry_ts}->{t.exit_ts} {t.pnl_usd:+.2f}$ "
                      f"({t.r:+.2f}R) {t.exit_reason} | {t.reason[:90]}", flush=True)

    all_trades.sort(key=lambda t: (t.session, t.entry_ts))
    paths = journal.save()
    print("\nSUMMARY:", json.dumps(summarize(all_trades), ensure_ascii=False), flush=True)
    print(f"totals: errors={err_total} gated={gate_total} cache_hits={hit_total} | "
          f"{len(paths)} journals -> data/journal/", flush=True)


if __name__ == "__main__":
    main()
