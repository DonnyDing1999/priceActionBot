"""Render the most recent complete opening-window session to a PNG (demo / eyeball)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from pab.render import render_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bars_5m_60d.parquet"
OUTDIR = ROOT / "data" / "renders"


def main() -> None:
    cont = pd.read_parquet(RAW).sort_index()
    cont.columns = [c.lower() for c in cont.columns]

    hhmm = cont.index.strftime("%H:%M")
    have_open = {d for d, t in zip(cont.index.date, hhmm) if t == "09:30"}
    have_1125 = {d for d, t in zip(cont.index.date, hhmm) if t == "11:25"}
    complete = sorted(have_open & have_1125)
    chosen = complete[-1] if complete else sorted(have_open)[-1]

    out = OUTDIR / f"session_{chosen}.png"
    info = render_session(cont, str(chosen), out_path=out)
    print("rendered:", info)


if __name__ == "__main__":
    main()
