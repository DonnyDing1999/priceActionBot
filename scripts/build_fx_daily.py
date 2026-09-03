"""Build daily CME FX futures panel from the Databento ohlcv-1d parent export.

Input : data/databento/*.ohlcv-1d.dbn.zst  (GLBX.MDP3, parent symbols 6E/6J/6B/6A/6C/6S/6N/6M,
        all expiries + calendar spreads; spreads are dropped).
Output: data/raw/fx_daily.parquet — one row per (date, root) with:
        front        : front-month contract symbol, chosen as the PRIOR day's volume winner
                       (no same-day lookahead; first day uses its own winner)
        open/high/low/close/volume : that contract's bar on the date
        ret          : close-to-close log return of the SAME contract (prev close of `front`
                       on the prior date) -> roll gaps never enter the return series
        ret_oc       : open-to-close log return of `front` on the date (used on rebalance days)
        next         : next-expiry contract after `front` (by expiry), close_next its close
        days_between : calendar days between the two expiries
        carry        : annualised implied (r_fx - r_usd) from the futures basis:
                       (close_front / close_next) ** (365 / days_between) - 1
                       (all 8 contracts quote USD per unit of foreign currency, so a long
                       front contract earns positive carry when the foreign rate is higher)
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = sorted(glob.glob(str(ROOT / "data" / "databento" / "*.ohlcv-1d.dbn.zst")))
OUT = ROOT / "data" / "raw" / "fx_daily.parquet"

_FX_RE = re.compile(r"^(6[A-Z])([FGHJKMNQUVXZ])(\d{1,2})$")
_MONTH = {c: i + 1 for i, c in enumerate("FGHJKMNQUVXZ")}


def parse_symbol(sym: str, ts: pd.Timestamp):
    """(root, expiry_date) for an outright; None for spreads. Expiry ~ 2 business days
    before the 3rd Wednesday of the contract month (CME FX rule) — used only for
    annualising carry, so the ~day-level approximation is fine."""
    m = _FX_RE.match(sym)
    if not m:
        return None
    root, mc, yy = m.group(1), m.group(2), int(m.group(3))
    # Databento symbols use 1-digit years ('6EU0') for most history; resolve the decade
    # relative to the bar's own year so 2010-era '0' != 2020-era '0'.
    if yy < 10:
        base = (ts.year // 10) * 10
        year = base + yy
        if year < ts.year - 1:          # e.g. bar in 2019, code '0' -> 2020
            year += 10
    else:
        year = 2000 + yy
    month = _MONTH[mc]
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (2 - first.weekday()) % 7          # days from the 1st to the first Wednesday
    third_wed = first + pd.Timedelta(days=offset + 14)
    expiry = third_wed - pd.offsets.BDay(2)
    return root, expiry.normalize()


def main() -> None:
    if not SRC:
        sys.exit("no *.ohlcv-1d.dbn.zst under data/databento/")
    import databento as db
    frames = [db.DBNStore.from_file(p).to_df().reset_index() for p in SRC]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["ts_event"]).dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    parsed = [parse_symbol(s, d) for s, d in zip(df["symbol"], df["date"])]
    df["root"] = [p[0] if p else None for p in parsed]
    df["expiry"] = [p[1] if p else pd.NaT for p in parsed]
    df = df.dropna(subset=["root"]).copy()
    df = df[df["volume"] > 0]
    df = df[df["date"].dt.dayofweek != 6]          # drop Sunday-UTC bars (thin Globex open)
    df = df.drop_duplicates(subset=["date", "symbol"])
    df = df[["date", "root", "symbol", "expiry", "open", "high", "low", "close", "volume"]]
    df = df.sort_values(["root", "date", "expiry"]).reset_index(drop=True)

    out_rows = []
    for root, g in df.groupby("root"):
        by_date = {d: sub for d, sub in g.groupby("date")}
        dates = sorted(by_date)
        prev_front = None
        prev_close_by_sym = {}
        for i, d in enumerate(dates):
            sub = by_date[d].set_index("symbol")
            winner_today = sub["volume"].idxmax()
            front = prev_front if (prev_front is not None and prev_front in sub.index) else winner_today
            row = sub.loc[front]
            # next expiry after front
            later = sub[sub["expiry"] > row["expiry"]].sort_values("expiry")
            if len(later):
                nxt = later.index[0]
                nrow = later.iloc[0]
                days_between = (nrow["expiry"] - row["expiry"]).days
                carry = (row["close"] / nrow["close"]) ** (365.0 / days_between) - 1.0
                close_next = nrow["close"]
            else:
                nxt, days_between, carry, close_next = None, np.nan, np.nan, np.nan
            pc = prev_close_by_sym.get(front)
            ret = np.log(row["close"] / pc) if pc else np.nan
            ret_oc = np.log(row["close"] / row["open"]) if row["open"] > 0 else np.nan
            out_rows.append(dict(date=d, root=root, front=front, expiry=row["expiry"],
                                 open=row["open"], high=row["high"], low=row["low"],
                                 close=row["close"], volume=int(row["volume"]),
                                 ret=ret, ret_oc=ret_oc, next=nxt, close_next=close_next,
                                 days_between=days_between, carry=carry))
            prev_front = winner_today          # tomorrow's front = today's volume winner
            prev_close_by_sym = sub["close"].to_dict()
    out = pd.DataFrame(out_rows).sort_values(["date", "root"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(f"wrote {OUT}: {len(out)} rows, {out['root'].nunique()} roots, "
          f"{out['date'].min().date()} -> {out['date'].max().date()}")
    print(out.groupby("root").agg(n=("date", "size"), first=("date", "min"), last=("date", "max"),
                                  carry_med=("carry", "median"), nan_ret=("ret", lambda s: s.isna().sum())))


if __name__ == "__main__":
    main()
