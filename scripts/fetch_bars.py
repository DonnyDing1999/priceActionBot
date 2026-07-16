"""Fetch ~60 days of 5-minute bars (free, via yfinance) to validate the data pipeline.

NOT for serious backtest: shallow history + crude continuous-contract stitching.
Tries ES=F (E-mini S&P futures) first; falls back to MES=F, then SPY (a
pipeline-only proxy) if Yahoo has no intraday data for the futures symbol.

Output: data/raw/bars_5m_60d.(parquet|csv)      full session
        data/raw/bars_5m_open_window.(parquet|csv)  RTH open window 09:30-11:30 ET
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

SYMBOLS = ["ES=F", "MES=F", "SPY"]   # preference order; SPY = pipeline-only proxy
INTERVAL = "5m"
PERIOD = "60d"                        # yfinance 5m history cap
ET = "America/New_York"
RTH_OPEN = "09:30"
WINDOW_LAST_BAR = "11:25"             # last 5m bar of the 09:30-11:30 window (24 bars)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch(sym: str) -> pd.DataFrame | None:
    df = yf.download(sym, interval=INTERVAL, period=PERIOD,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):          # newer yfinance: (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna(how="any")
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(ET)
    df.index.name = "ts_et"
    return df


def save(df: pd.DataFrame, stem: Path) -> Path:
    try:
        p = stem.with_suffix(".parquet")
        df.to_parquet(p)
        return p
    except Exception:                                   # no pyarrow -> CSV
        p = stem.with_suffix(".csv")
        df.to_csv(p)
        return p


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, used = None, None
    for s in SYMBOLS:
        df = fetch(s)
        if df is not None and len(df):
            used = s
            break
        print(f"[warn] no intraday data for {s}", file=sys.stderr)
    if df is None:
        print("ERROR: no data from any symbol", file=sys.stderr)
        sys.exit(1)

    full = save(df, OUT / "bars_5m_60d")
    win = df.between_time(RTH_OPEN, WINDOW_LAST_BAR)
    winp = save(win, OUT / "bars_5m_open_window")

    print(f"symbol_used = {used}   interval = {INTERVAL}   period = {PERIOD}")
    print(f"full bars   : {len(df):>6}   {df.index.min()}  ->  {df.index.max()}")
    print(f"window bars : {len(win):>6}   (09:30-11:30 ET)   "
          f"sessions ~ {win.index.normalize().nunique()}")
    print(f"saved       : {full.name} , {winp.name}")
    print("--- last 3 window bars ---")
    print(win.tail(3).to_string())


if __name__ == "__main__":
    main()
