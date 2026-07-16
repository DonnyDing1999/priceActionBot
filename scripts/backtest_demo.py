"""Run the toy strategy over all complete opening sessions (plumbing validation)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.backtest import Config, run, summarize, toy_strategy  # noqa: E402
from pab.bars import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bars_5m_60d.parquet"


def main() -> None:
    cont = load_bars(RAW)
    hhmm = cont.index.strftime("%H:%M")
    have_open = {d for d, t in zip(cont.index.date, hhmm) if t == "09:30"}
    have_1125 = {d for d, t in zip(cont.index.date, hhmm) if t == "11:25"}
    sessions = [str(d) for d in sorted(have_open & have_1125)]

    trades = run(cont, sessions, toy_strategy, Config())
    print(f"sessions: {len(sessions)}   trades: {len(trades)}")
    print("summary:", json.dumps(summarize(trades), ensure_ascii=False))
    print("\nfirst 8 trades:")
    for t in trades[:8]:
        print(f"  {t.session} {t.side:<5} {t.entry_ts}->{t.exit_ts} "
              f"entry {t.entry} exit {t.exit} risk {t.risk_pts}pt "
              f"{t.pnl_usd:+.2f}$ ({t.r:+.2f}R) {t.exit_reason}")


if __name__ == "__main__":
    main()
