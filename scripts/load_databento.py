"""Load the Databento MES 1-min OHLCV export -> clean continuous 1-min + 5-min parquets.

Input : data/databento/*.ohlcv-1m.json.zst  (parent MES.FUT = all contracts, UTC, pretty px)
Steps : parse JSONL -> tz-convert UTC->ET -> pick the front-month per DAY using the PRIOR
        day's volume winner (no intraday lookahead: at today's open you only know which
        contract won yesterday; first day uses its own) -> save the continuous 1m frame
        (data/raw/mes_1m.parquet — used by the backtester to resolve intrabar ordering)
        -> resample 1m to 5m -> save data/raw/mes_5m.parquet.
Output matches the bars parquet the pipeline already uses (ohlcv + tz-aware ET index).
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import zstandard as zstd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = glob.glob(str(ROOT / "data" / "databento" / "*.ohlcv-1m.json.zst"))
OUT_5M = ROOT / "data" / "raw" / "mes_5m.parquet"
OUT_1M = ROOT / "data" / "raw" / "mes_1m.parquet"
ET = "America/New_York"


def read_records(path: str) -> pd.DataFrame:
    with open(path, "rb") as fh:
        text = zstd.ZstdDecompressor().stream_reader(fh).read().decode("utf-8")
    rows = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        rows.append((r["hd"]["ts_event"], float(r["open"]), float(r["high"]),
                     float(r["low"]), float(r["close"]), int(r["volume"]), r["symbol"]))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                       "volume", "symbol"])


def main() -> None:
    if not SRC:
        print("no .ohlcv-1m.json.zst under data/databento/ — extract the Databento zip first")
        return
    df = read_records(SRC[0])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
    df["date"] = df["ts"].dt.date

    # front-month per day = PRIOR day's volume winner (usable at today's open, no
    # intraday lookahead; the first day has no prior -> uses its own winner)
    vol = df.groupby(["date", "symbol"])["volume"].sum().reset_index()
    winner = (vol.sort_values("volume").groupby("date").tail(1)
              .set_index("date")["symbol"].sort_index())
    dates = list(winner.index)
    front = pd.DataFrame({
        "date": dates,
        "symbol": [winner.iloc[max(0, k - 1)] for k in range(len(dates))]})
    df = df.merge(front, on=["date", "symbol"], how="inner")

    df = df.set_index("ts").sort_index()
    m1 = df[["open", "high", "low", "close", "volume"]]
    m1.index.name = "ts_et"
    OUT_1M.parent.mkdir(parents=True, exist_ok=True)
    m1.to_parquet(OUT_1M)

    # resample continuous 1m -> 5m (bins align to ET wall clock; 09:30 is a bin start)
    bars = (m1.resample("5min", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna(subset=["open"]))
    bars.index.name = "ts_et"
    bars.to_parquet(OUT_5M)

    hhmm = bars.index.strftime("%H:%M")
    sess = sorted({d for d, t in zip(bars.index.date, hhmm) if t == "09:30"}
                  & {d for d, t in zip(bars.index.date, hhmm) if t == "11:25"})
    roll_days = front[front["symbol"] != front["symbol"].shift()]
    print(f"saved {OUT_1M.name}: {len(m1):,} 1m bars | {OUT_5M.name}: {len(bars):,} 5m bars")
    print(f"range: {bars.index.min()} -> {bars.index.max()}")
    print(f"complete opening-window sessions (09:30 & 11:25 present): {len(sess)}")
    print(f"contracts / rolls: {dict(zip(roll_days['date'].astype(str), roll_days['symbol']))}")
    print(f"price range: {bars['low'].min():.2f} - {bars['high'].max():.2f}")


if __name__ == "__main__":
    main()
