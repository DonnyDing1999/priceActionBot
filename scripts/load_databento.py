"""Load Databento OHLCV-1m exports -> clean continuous 1-min + 5-min parquets, per root.

Input : data/databento/*.ohlcv-1m.{json,dbn}.zst  (parent .FUT = all contracts, UTC,
        pretty px; JSON lines or Databento binary — both decode to the same frame).
        Multiple exports may be extracted into data/databento/ side by side (e.g. the
        2024-25 MES history next to the 2026 MES file, and an MNQ export) — every
        matching file is parsed and concatenated, then split by instrument root.
Steps : parse each export -> tz-convert UTC->ET -> derive each row's instrument ROOT from its
        contract symbol (MESH6 -> MES, MNQU6 -> MNQ; calendar spreads are skipped) ->
        for EACH root: dedupe -> pick the front-month per DAY using the PRIOR day's volume
        winner (no intraday lookahead: at today's open you only know which contract won
        yesterday; first day uses its own) -> save the continuous 1m frame
        (data/raw/<root>_1m.parquet — used by the backtester to resolve intrabar ordering)
        -> resample 1m to 5m -> save data/raw/<root>_5m.parquet.
Output matches the bars parquet the pipeline already uses (ohlcv + tz-aware ET index);
MES filenames (mes_1m.parquet / mes_5m.parquet) are unchanged.
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import zstandard as zstd  # noqa: E402

from pab.bars import contract_root  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = sorted(glob.glob(str(ROOT / "data" / "databento" / "*.ohlcv-1m.json.zst"))
             + glob.glob(str(ROOT / "data" / "databento" / "*.ohlcv-1m.dbn.zst")))
OUT_DIR = ROOT / "data" / "raw"
ET = "America/New_York"


def read_records(path: str) -> pd.DataFrame:
    if path.endswith(".dbn.zst"):
        import databento as db  # optional dep: only needed for the binary exports
        df = db.DBNStore.from_file(path).to_df()  # tz-aware UTC index, decimal px, int vol
        if "symbol" not in df.columns:
            raise SystemExit(f"{path}: DBNStore.to_df() returned no 'symbol' column — "
                             "cannot derive contract roots (re-export with symbology on)")
        # normalize to the same frame as the JSON branch below (outrights like MESH4 plus
        # spreads MESH4-MESM4; contract_root() drops the spreads downstream)
        out = df.reset_index()[["ts_event", "open", "high", "low", "close",
                                "volume", "symbol"]].copy()
        out.columns = ["ts", "open", "high", "low", "close", "volume", "symbol"]
        out["volume"] = out["volume"].astype(int)
        return out
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


def process_root(root: str, sub: pd.DataFrame) -> None:
    """Front-month continuous 1m + 5m parquets for one instrument root."""
    sub = sub.drop_duplicates(subset=["ts", "symbol"])   # overlapping exports -> one row

    # front-month per day = PRIOR day's volume winner (usable at today's open, no
    # intraday lookahead; the first day has no prior -> uses its own winner)
    vol = sub.groupby(["date", "symbol"])["volume"].sum().reset_index()
    winner = (vol.sort_values("volume").groupby("date").tail(1)
              .set_index("date")["symbol"].sort_index())
    dates = list(winner.index)
    front = pd.DataFrame({
        "date": dates,
        "symbol": [winner.iloc[max(0, k - 1)] for k in range(len(dates))]})
    sub = sub.merge(front, on=["date", "symbol"], how="inner")

    sub = sub.set_index("ts").sort_index()
    m1 = sub[["open", "high", "low", "close", "volume"]]
    m1.index.name = "ts_et"
    out_1m = OUT_DIR / f"{root.lower()}_1m.parquet"
    out_5m = OUT_DIR / f"{root.lower()}_5m.parquet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m1.to_parquet(out_1m)

    # resample continuous 1m -> 5m (bins align to ET wall clock; 09:30 is a bin start)
    bars = (m1.resample("5min", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna(subset=["open"]))
    bars.index.name = "ts_et"
    bars.to_parquet(out_5m)

    hhmm = bars.index.strftime("%H:%M")
    sess = sorted({d for d, t in zip(bars.index.date, hhmm) if t == "09:30"}
                  & {d for d, t in zip(bars.index.date, hhmm) if t == "11:25"})
    roll_days = front[front["symbol"] != front["symbol"].shift()]
    print(f"[{root}] saved {out_1m.name}: {len(m1):,} 1m bars | "
          f"{out_5m.name}: {len(bars):,} 5m bars")
    print(f"[{root}] range: {bars.index.min()} -> {bars.index.max()}")
    print(f"[{root}] complete opening-window sessions (09:30 & 11:25 present): {len(sess)}")
    print(f"[{root}] contracts / rolls: "
          f"{dict(zip(roll_days['date'].astype(str), roll_days['symbol']))}")
    print(f"[{root}] price range: {bars['low'].min():.2f} - {bars['high'].max():.2f}")


def main() -> None:
    if not SRC:
        print("no .ohlcv-1m.{json,dbn}.zst under data/databento/ — extract the Databento "
              "zip(s) there first (multiple exports may sit side by side)")
        return
    df = pd.concat([read_records(p) for p in SRC], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
    df["date"] = df["ts"].dt.date
    df["root"] = df["symbol"].map(contract_root)
    df = df[df["root"].notna()]          # drop calendar spreads / non-outright symbols

    for root in sorted(df["root"].unique()):
        process_root(root, df[df["root"] == root])


if __name__ == "__main__":
    main()
