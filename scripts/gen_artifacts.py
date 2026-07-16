"""Generate per-bar, no-lookahead decision artifacts for the last N sessions.

For each decision bar K (1..23) of a session it renders the chart truncated to bars <= K
and builds the K-only sidecar, then writes data/artifacts/<session>/manifest.json listing
{bar, time, image, sidecar}. A Claude decision subagent consumes one session's manifest and
decides bar-by-bar. Because each image/sidecar is truncated to <= K, processing the list in
order is no-lookahead.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from pab.bars import ET, load_bars  # noqa: E402
from pab.features import build_sidecar  # noqa: E402
from pab.render import render_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bars_5m_60d.parquet"
OUT = ROOT / "data" / "artifacts"


def main() -> None:
    n = int(os.getenv("N_SESSIONS", "5"))
    cont = load_bars(RAW)
    hhmm = cont.index.strftime("%H:%M")
    have_open = {d for d, t in zip(cont.index.date, hhmm) if t == "09:30"}
    have_1125 = {d for d, t in zip(cont.index.date, hhmm) if t == "11:25"}
    sessions = [str(d) for d in sorted(have_open & have_1125)][-n:]

    for s in sessions:
        sdir = OUT / s
        sdir.mkdir(parents=True, exist_ok=True)
        open_ts = pd.Timestamp(f"{s} 09:30", tz=ET)
        end_ts = pd.Timestamp(f"{s} 11:25", tz=ET)
        win = cont[(cont.index >= open_ts) & (cont.index <= end_ts)]
        manifest = []
        for k, ts in enumerate(win.index[:-1], start=1):  # bars 1..23 (need a next bar to fill)
            img = sdir / f"bar_{k:02d}.png"
            render_session(cont, s, up_to=ts, out_path=img)
            manifest.append({"bar": k, "time": ts.strftime("%H:%M"),
                             "image": str(img), "sidecar": build_sidecar(cont, ts)})
        (sdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), "utf-8")
        print(f"{s}: {len(manifest)} decision bars -> {sdir/'manifest.json'}")

    print("sessions:", ",".join(sessions))


if __name__ == "__main__":
    main()
