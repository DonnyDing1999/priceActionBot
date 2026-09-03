"""Kalshi daily-high-temperature markets — model probability vs market price, exam v1.

Pre-registered in oos/PREREGISTRATION_KALSHI_TEMP_v1.md; every parameter below is fixed there.
Inputs : data/raw/kalshi/highs_markets_meta.json      (settled markets, 13 series)
         <scratchpad>/kcandles/<ticker>.json          (hourly candlesticks with yes_bid/yes_ask)
         data/raw/iem/mos_{GFS,NAM}_<station>.csv    (IEM bulk MOS, 2023-01 -> 2026-09)
         data/raw/iem/cli_high.csv                    (NWS CLI daily highs, 2023 -> 2026)
Outputs: oos/kalshi_temp_v1_results.md, data/artifacts/kalshi_temp_v1_trades.csv
"""
import datetime as dt
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDLES = ROOT / "data" / "raw" / "kalshi" / "candles"        # one JSON per market (hourly bid/ask candles)
META = ROOT / "data" / "raw" / "kalshi" / "highs_markets_meta.json"
IEM = ROOT / "data" / "raw" / "iem"
OUT_MD = ROOT / "oos" / "kalshi_temp_v1_results.md"
OUT_CSV = ROOT / "data" / "artifacts" / "kalshi_temp_v1_trades.csv"

ENTRY_UTC_HOUR = 7
EV_MIN = 0.02
MAX_SPREAD = 0.10
CAL_START, CAL_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-06-24")
CAL_MONTHS = {5, 6, 7, 8, 9}
CAL_MIN_N = 200
IS = (pd.Timestamp("2026-06-25"), pd.Timestamp("2026-07-28"))
OOS = (pd.Timestamp("2026-07-29"), pd.Timestamp("2026-08-31"))
MIN_OOS_TRADES = 100


def fee(p: float) -> float:
    """Kalshi taker fee per contract, dollars: ceil(7 * P * (1-P)) cents."""
    return math.ceil(7.0 * p * (1.0 - p) - 1e-9) / 100.0


SERIES_STATION = {"KXHIGHNY": "KNYC", "KXHIGHCHI": "KMDW", "KXHIGHMIA": "KMIA", "KXHIGHAUS": "KAUS",
                  "KXHIGHDEN": "KDEN", "KXHIGHLAX": "KLAX", "KXHIGHPHIL": "KPHL", "KXHIGHTDC": "KDCA",
                  "KXHIGHTSFO": "KSFO", "KXHIGHTATL": "KATL", "KXHIGHTDAL": "KDFW", "KXHIGHTSEA": "KSEA",
                  "KXHIGHTPHX": "KPHX"}       # per the manifest; rules text only carries the CLI code from 2026-08-14


def station_of(rules: str) -> str | None:
    m = re.search(r"\((CLI[A-Z]{3,4})\)", rules or "")
    return "K" + m.group(1)[3:] if m else None


def station_for(m: dict) -> str | None:
    s = SERIES_STATION.get(m["series"])
    parsed = station_of(m.get("rules_primary"))
    if parsed and s and parsed != s:
        raise SystemExit(f"{m['ticker']}: rules station {parsed} != series map {s}")
    return s or parsed


def event_date(event_ticker: str) -> pd.Timestamp:
    tail = event_ticker.split("-")[1]                     # 26SEP01
    return pd.Timestamp(dt.datetime.strptime(tail, "%y%b%d").date())


def load_mos() -> pd.DataFrame:
    """Per (station, date): 00Z-run 12h max temp for that day's daytime from GFS and NAM."""
    frames = []
    for f in sorted(IEM.glob("mos_*_K*.csv")):
        model, stn = f.stem.split("_")[1], f.stem.split("_")[2]
        d = pd.read_csv(f, usecols=["runtime", "ftime", "n_x"], low_memory=False)
        d = d[d["n_x"].notna() & (d["n_x"].astype(str) != "M")]
        d["runtime"] = pd.to_datetime(d["runtime"]); d["ftime"] = pd.to_datetime(d["ftime"])
        d = d[(d["runtime"].dt.hour == 0)]
        d["date"] = d["runtime"].dt.normalize()
        d = d[d["ftime"] == d["date"] + pd.Timedelta(days=1)]        # ftime = D+1 00Z -> daytime max of D
        d = d.assign(station=stn, model=model, n_x=d["n_x"].astype(float))[["station", "date", "model", "n_x"]]
        frames.append(d)
    m = pd.concat(frames).drop_duplicates(["station", "date", "model"])
    piv = m.pivot_table(index=["station", "date"], columns="model", values="n_x")
    piv["f"] = piv[["GFS", "NAM"]].mean(axis=1, skipna=True)
    return piv.reset_index()[["station", "date", "f"]].dropna()


def load_cli() -> pd.DataFrame:
    c = pd.read_csv(IEM / "cli_high.csv")
    c["date"] = pd.to_datetime(c["date"]); c["high"] = pd.to_numeric(c["high"], errors="coerce")
    return c.dropna(subset=["high"])[["station", "date", "high"]]


def bounds(m: dict):
    st, fl, cp = m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike")
    if st == "greater":
        return fl + 1, np.inf
    if st == "less":
        return -np.inf, cp - 1
    return fl, cp                                            # between: inclusive integer bin


def verify_bounds(meta: list) -> str:
    """Data check (not tuning): the bounds rule must reproduce Kalshi's own results exactly."""
    ok = bad = 0
    for m in meta:
        v = m.get("expiration_value")
        if v in (None, ""):
            continue
        lo, hi = bounds(m); pred = "yes" if lo <= float(v) <= hi else "no"
        ok += pred == m["result"]; bad += pred != m["result"]
    return f"区间规则复核：{ok} 一致，{bad} 不一致"


def quotes_at(ticker: str, t_end: int):
    f = CANDLES / f"{ticker}.json"
    if not f.exists():
        return None, "no_candle_file"
    cs = json.load(open(f))
    prior = [c for c in cs if c["end_period_ts"] <= t_end]
    if not prior:
        return None, "no_candle"
    c = max(prior, key=lambda x: x["end_period_ts"])
    bid = float(c["yes_bid"]["close_dollars"]); ask = float(c["yes_ask"]["close_dollars"])
    return (bid, ask, c["end_period_ts"]), None


def main() -> None:
    meta = json.load(open(META))
    mos, cli = load_mos(), load_cli()
    notes = [verify_bounds(meta)]

    # calibration errors per station (summer months, before the Kalshi window)
    cal = cli.merge(mos, on=["station", "date"])
    cal = cal[(cal["date"] >= CAL_START) & (cal["date"] <= CAL_END) & cal["date"].dt.month.isin(CAL_MONTHS)]
    cal["e"] = cal["high"] - cal["f"]
    errs = {s: g["e"].to_numpy() for s, g in cal.groupby("station")}
    notes.append("校准误差条数：" + ", ".join(f"{s} {len(e)} (均值 {e.mean():+.2f}, σ {e.std():.2f})" for s, e in sorted(errs.items())))
    fmap = mos.set_index(["station", "date"])["f"]

    rows, skips = [], {}
    for m in meta:
        D = event_date(m["event_ticker"]); stn = station_for(m)
        e = errs.get(stn)
        if e is None or len(e) < CAL_MIN_N:
            skips["no_calibration"] = skips.get("no_calibration", 0) + 1; continue
        f = fmap.get((stn, D))
        if f is None or pd.isna(f):
            skips["no_forecast"] = skips.get("no_forecast", 0) + 1; continue
        lo, hi = bounds(m)
        pred = np.floor(f + e + 0.5)
        p = float(np.mean((pred >= lo) & (pred <= hi)))
        t_end = int(pd.Timestamp(D.year, D.month, D.day, ENTRY_UTC_HOUR, tz="UTC").timestamp())
        q, why = quotes_at(m["ticker"], t_end)
        if q is None:
            skips[why] = skips.get(why, 0) + 1; continue
        bid, ask, cts = q
        outcome = 1.0 if m["result"] == "yes" else 0.0
        base = dict(ticker=m["ticker"], series=m["series"], station=stn, date=D, kind=m.get("strike_type"),
                    f=f, p=p, bid=bid, ask=ask, mid=(bid + ask) / 2, quote_ts=cts, outcome=outcome,
                    value=float(m["expiration_value"]) if m.get("expiration_value") not in (None, "") else np.nan)
        if bid < 0.01 or ask > 0.99 or ask - bid > MAX_SPREAD:
            skips["bad_quote"] = skips.get("bad_quote", 0) + 1
            rows.append(dict(base, traded=False, side=None, price=np.nan, ev=np.nan, pnl=np.nan)); continue
        ev_yes = p - ask - fee(ask)
        ev_no = bid - p - fee(1 - bid)
        if max(ev_yes, ev_no) < EV_MIN:
            skips["ev_below_min"] = skips.get("ev_below_min", 0) + 1
            rows.append(dict(base, traded=False, side=None, price=np.nan, ev=max(ev_yes, ev_no), pnl=np.nan)); continue
        if ev_yes >= ev_no:
            side, price, win = "yes", ask, outcome
            pnl = win - price - fee(price); ev = ev_yes; p_side = p
        else:
            side, price, win = "no", 1 - bid, 1 - outcome
            pnl = win - price - fee(price); ev = ev_no; p_side = 1 - p
        kelly = 0.25 * max(0.0, (p_side - price) / (1 - price))          # fraction of bankroll staked
        rows.append(dict(base, traded=True, side=side, price=price, ev=ev, pnl=pnl,
                         kelly_ret=kelly * pnl / price))
    T = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True); T.to_csv(OUT_CSV, index=False)

    def win(a, b):
        return T[(T["date"] >= a) & (T["date"] <= b)]

    def summarize(W):
        tr = W[W["traded"] == True]
        n = len(tr); mu = tr["pnl"].mean() if n else np.nan; sd = tr["pnl"].std() if n > 1 else np.nan
        return dict(n=n, total=tr["pnl"].sum(), mean=mu, t=(mu / sd * np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
                    hit=(tr["pnl"] > 0).mean() if n else np.nan, cap=tr["price"].sum(),
                    roc=tr["pnl"].sum() / tr["price"].sum() if n else np.nan,
                    kelly=tr.groupby("date")["kelly_ret"].sum().mean() if n else np.nan,   # mean daily bankroll return
                    brier_model=((W["p"] - W["outcome"]) ** 2).mean(), brier_mkt=((W["mid"] - W["outcome"]) ** 2).mean(),
                    n_quoted=len(W), days=W["date"].nunique())

    S = {"IS": summarize(win(*IS)), "OOS": summarize(win(*OOS))}
    o, i = S["OOS"], S["IS"]
    enough = o["n"] >= MIN_OOS_TRADES
    passed = enough and o["total"] > 0 and o["t"] >= 2.0 and i["total"] > 0

    L = ["# Kalshi 最高气温合约 v1 — 结果（一次性运行，参数见 PREREGISTRATION_KALSHI_TEMP_v1.md）", ""]
    L += [f"- {n}" for n in notes]
    L += [f"- 跳过原因：{skips}", "",
          "| 窗口 | 事件日 | 有报价合约 | 成交 | 净盈亏($) | 每笔均值(¢) | 池化 t | 胜率 | 投入资本($) | 资本回报 | ¼Kelly 日均 | Brier 模型 | Brier 市场 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w, s in S.items():
        L.append(f"| {w} | {s['days']} | {s['n_quoted']} | {s['n']} | {s['total']:+.2f} | {s['mean']*100:+.2f} | "
                 f"{s['t']:+.2f} | {s['hit']:.1%} | {s['cap']:.0f} | {s['roc']:+.1%} | {s['kelly']:+.2%} | "
                 f"{s['brier_model']:.4f} | {s['brier_mkt']:.4f} |")
    verdict = "样本不足" if not enough else ("通过" if passed else "未通过")
    L += ["", f"**主判据：OOS 净盈亏 {o['total']:+.2f}（>0），OOS 池化 t {o['t']:+.2f}（≥2.0），IS 净盈亏 {i['total']:+.2f}（>0），"
          f"OOS 成交 {o['n']}（≥{MIN_OOS_TRADES}）→ {verdict}**", ""]
    tr = T[T["traded"] == True]
    for label, col in [("按城市", "series"), ("按方向", "side"), ("按合约类型", "kind")]:
        L += [f"## {label}（全样本成交）", "", "| 组 | 成交 | 每笔净(¢) | 胜率 | 池化 t |", "|---|---|---|---|---|"]
        for k, g in tr.groupby(col):
            n = len(g); t = g["pnl"].mean() / g["pnl"].std() * np.sqrt(n) if n > 1 and g["pnl"].std() > 0 else np.nan
            L.append(f"| {k} | {n} | {g['pnl'].mean()*100:+.2f} | {(g['pnl']>0).mean():.1%} | {t:+.2f} |")
        L.append("")
    L += ["## 按事件日累计（OOS，净 $）", ""]
    daily = tr[(tr["date"] >= OOS[0])].groupby("date")["pnl"].sum().cumsum()
    L += [f"- {d.date()}: {v:+.2f}" for d, v in daily.iloc[::5].items()]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
