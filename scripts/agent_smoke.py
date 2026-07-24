"""Numeric smoke test: build the sidecar for one bar of the latest complete session
and ask the LLM once (numeric path only, no rendering). Prints one decision.
Provider/model/instrument/window come from .env (PA_PROVIDER / PA_MODEL / PA_INSTRUMENT / PA_WINDOW).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from pab.agent import AgentConfig, decide, key_status  # noqa: E402  (also loads .env)
from pab.bars import ET, complete_sessions, load_bars  # noqa: E402
from pab.features import build_sidecar  # noqa: E402
from pab.instruments import get_spec  # noqa: E402
from pab.orchestration import WINDOWS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    st = key_status()
    if not st["ok"]:
        print(f"No API key for provider '{st['provider']}' (model {st['model']}).")
        print(f"  Put  {st['hint']}  in a repo-root .env, then re-run.")
        return

    cfg = AgentConfig()
    spec = get_spec(cfg.instrument)
    w = WINDOWS[cfg.window]
    cont = load_bars(ROOT / "data" / "raw" / f"{cfg.instrument}_5m.parquet")
    sess = complete_sessions(cont, first=w["start"], last=w["end"])[-1]

    open_ts = pd.Timestamp(f"{sess} {w['start']}", tz=ET)
    window = cont[cont.index >= open_ts]
    cur_ts = window.index[min(7, len(window) - 1)]      # ~bar 8 from the open

    sidecar = build_sidecar(cont, cur_ts, symbol=spec.symbol, spec=spec,
                            window_start=w["start"])
    print(f"provider={st['provider']} model={st['model']} "
          f"instrument={cfg.instrument} window={cfg.window} — session {sess}, "
          f"bar {sidecar['bar_index_from_open']} @ {cur_ts.strftime('%H:%M')}\n")
    d = decide(sidecar, cfg=cfg)
    usage = d.pop("_usage", {})
    print(f"action={d.get('action')}  setup={d.get('setup')}")
    print(f"reason: {d.get('reason')}")
    print("usage:", usage)


if __name__ == "__main__":
    main()
