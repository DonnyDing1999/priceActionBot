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
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.agent import AgentConfig, key_status  # noqa: E402 (loads .env)
from pab.backtest import run_session, summarize  # noqa: E402
from pab.bars import complete_sessions, load_bars  # noqa: E402
from pab.experience import load_all_cases  # noqa: E402
from pab.instruments import get_spec  # noqa: E402
from pab.journal import Journal  # noqa: E402
from pab.llm_strategy import LLMStrategy, RunAborted  # noqa: E402
from pab.orchestration import (WINDOWS, day_gate, engine_config,  # noqa: E402
                               journal_dir, run_id)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    st = key_status()
    if not st["ok"]:
        print(f"No API key for provider '{st['provider']}'. Put {st['hint']} in .env "
              "(or set PA_PROVIDER=gemini for the free option).")
        return

    n = int(os.getenv("N_SESSIONS", "3"))
    workers = int(os.getenv("PA_WORKERS", "3"))
    dayfilter = os.getenv("PA_DAYFILTER", "overlap")  # off | overlap | efficiency | combo | llm
    cfg = AgentConfig()

    # session window: am = the opening two hours (default); pm = 13:30-15:30 afternoon
    window = cfg.window
    spec = get_spec(cfg.instrument)
    ecfg = engine_config(spec, window)
    w_first, w_last = WINDOWS[window]["start"], WINDOWS[window]["end"]

    # parquet paths follow the instrument (mes_5m/mnq_5m/...); PA_BARS overrides the 5m file
    raw = ROOT / "data" / "raw" / os.getenv("PA_BARS", f"{cfg.instrument}_5m.parquet")
    raw_1m = ROOT / "data" / "raw" / f"{cfg.instrument}_1m.parquet"
    cont = load_bars(raw)
    m1 = load_bars(raw_1m) if raw_1m.exists() else None
    all_sessions = complete_sessions(cont, first=w_first, last=w_last)
    sl = os.getenv("PA_SLICE")           # e.g. "0:20" = first 20 sessions; overrides N_SESSIONS
    if sl:
        a, _, b = sl.partition(":")
        sessions = all_sessions[int(a or 0):(int(b) if b else None)]
    else:
        sessions = all_sessions[-n:]

    print(f"LLM backtest — provider={cfg.provider} model={cfg.resolved_model()} "
          f"temp={cfg.temperature} cache={'on' if cfg.cache else 'off'} "
          f"m1={'on' if m1 is not None else 'off'} workers={workers} "
          f"dayfilter={dayfilter} window={window}")
    print(f"{len(sessions)} sessions {sessions}\n", flush=True)

    cases = load_all_cases()   # snapshot the experience library ONCE for the whole run

    def run_one(s: str):
        # day-level chop gate: a C-graded day is skipped entirely (no per-bar LLM calls).
        # 'llm' = the model judges A/C itself from strictly-prospective inputs
        # (yesterday + overnight; + this morning for the pm window); else mechanical.
        gate = day_gate(cont, s, mode=dayfilter, window=window, cfg=cfg)
        if gate is not None:
            gi = gate if gate.get("reason") else {}
            return s, [], None, {"day_grade": "C", "grade_info": gi}
        strat = LLMStrategy(cont, cfg=cfg, frozen_cases=cases)  # per-session state
        stats: dict = {}
        trades = run_session(cont, s, strat, ecfg, m1=m1, stats=stats)
        return s, trades, strat, stats

    jdir = journal_dir(cfg.instrument, window)
    jdir.mkdir(parents=True, exist_ok=True)
    snap = jdir / "experience_snapshot.jsonl"      # freeze this run's experience library
    snap.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases),
                    "utf-8")
    exp_sha = hashlib.sha256(snap.read_bytes()).hexdigest()
    journal = Journal(run_id=run_id(cfg.provider, cfg.instrument, window),
                      provider=cfg.provider, model=cfg.resolved_model(),
                      extra_meta={"experience_sha": exp_sha})
    all_trades, err_total, gate_total, hit_total = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, s): s for s in sessions}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                s, trades, strat, stats = fut.result()
            except RunAborted as e:          # consecutive call failures -> stop the run
                print("\n!! RUN ABORTED (consecutive call failures — quota/auth?)",
                      flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                journal.meta["aborted"] = str(e)
                journal.save(dir=jdir)
                sys.exit(2)
            except Exception as e:  # noqa: BLE001 — a whole-session failure
                print(f"[{s}] ERROR: {type(e).__name__}: {str(e)[:160]} — skipping",
                      flush=True)
                continue
            if strat is None:            # C-graded day: journal the skip, zero trades
                gi = stats.get("grade_info") or {}
                journal.record_decision(s, 0, w_first,
                                        {"action": "no_trade",
                                         "reason": f"day_gate: C ({dayfilter}) "
                                                   f"{gi.get('reason', '')}"[:220]})
                journal.save(dir=jdir)
                why = f" | {gi.get('reason', '')[:90]}" if gi else ""
                print(f"[{s}] C-DAY skipped (day gate){why}", flush=True)
                continue
            all_trades.extend(trades)
            err_total += strat.errors
            gate_total += strat.gated
            hit_total += strat.cache_hits
            for d in strat.decisions:
                journal.record_decision(s, d["bar"], d["time"], d["decision"],
                                        sidecar=d["sidecar"])
            journal.record_trades(s, trades)
            journal.save(dir=jdir)  # flush incrementally so a mid-run check can read journals
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
    paths = journal.save(dir=jdir)
    print("\nSUMMARY:", json.dumps(summarize(all_trades), ensure_ascii=False), flush=True)
    print(f"totals: errors={err_total} gated={gate_total} cache_hits={hit_total} | "
          f"{len(paths)} journals -> {jdir.relative_to(ROOT)}/", flush=True)


if __name__ == "__main__":
    main()
