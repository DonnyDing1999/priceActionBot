"""Smoke test: render the latest complete session's last window bar and ask the LLM once.

Provider/model from .env (PA_PROVIDER / PA_MODEL). Prints one decision.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from pab.agent import decide, key_status  # noqa: E402  (also loads .env on import)
from pab.bars import ET, load_bars  # noqa: E402
from pab.features import build_sidecar  # noqa: E402
from pab.render import render_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bars_5m_60d.parquet"


def main() -> None:
    st = key_status()
    if not st["ok"]:
        print(f"No API key for provider '{st['provider']}' (model {st['model']}).")
        print(f"  Put  {st['hint']}  in a repo-root .env, then re-run.")
        print("  Switch provider in .env:  PA_PROVIDER=gemini   (free) or  anthropic")
        print("  Override model in .env:   PA_MODEL=gemini-2.5-pro  (etc.)")
        return

    cont = load_bars(RAW)
    hhmm = cont.index.strftime("%H:%M")
    have_open = {d for d, t in zip(cont.index.date, hhmm) if t == "09:30"}
    have_1125 = {d for d, t in zip(cont.index.date, hhmm) if t == "11:25"}
    sess = sorted(have_open & have_1125)[-1]

    open_ts = pd.Timestamp(f"{sess} 09:30", tz=ET)
    end_ts = pd.Timestamp(f"{sess} 11:25", tz=ET)
    cur_ts = cont[(cont.index >= open_ts) & (cont.index <= end_ts)].index[-1]

    sidecar = build_sidecar(cont, cur_ts)
    out = ROOT / "data" / "renders" / f"smoke_{sess}_{cur_ts.strftime('%H%M')}.png"
    render_session(cont, str(sess), up_to=cur_ts, out_path=out)

    print(f"provider={st['provider']}  model={st['model']}")
    print(f"session {sess}, bar @ {cur_ts.strftime('%H:%M')} (bar {sidecar['bar_index_from_open']})")
    print("calling the model for one decision...\n")
    d = decide(sidecar, out)
    usage = d.pop("_usage", {})
    print(json.dumps(d, indent=2, ensure_ascii=False))
    print("\nusage:", usage)


if __name__ == "__main__":
    main()
