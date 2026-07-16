"""Demo: build the numeric sidecar for the last window bar of the latest session."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from pab.bars import ET, load_bars  # noqa: E402
from pab.features import build_sidecar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bars_5m_60d.parquet"


def main() -> None:
    cont = load_bars(RAW)
    hhmm = cont.index.strftime("%H:%M")
    have_open = {d for d, t in zip(cont.index.date, hhmm) if t == "09:30"}
    have_1125 = {d for d, t in zip(cont.index.date, hhmm) if t == "11:25"}
    sess = sorted(have_open & have_1125)[-1]

    open_ts = pd.Timestamp(f"{sess} 09:30", tz=ET)
    end_ts = pd.Timestamp(f"{sess} 11:25", tz=ET)
    win = cont[(cont.index >= open_ts) & (cont.index <= end_ts)]
    cur_ts = win.index[-1]

    sc = build_sidecar(cont, cur_ts)
    print(f"# perception sidecar — session {sess}, bar @ {cur_ts.strftime('%H:%M')} "
          f"(bar {sc['bar_index_from_open']})")
    print(json.dumps(sc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
