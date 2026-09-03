"""FX v3 — month-end London 4pm fix rebalancing-flow exam.

Pre-registered in oos/PREREGISTRATION_FX_FIX_v3.md; every parameter below is fixed there.
Inputs : data/raw/fx_1h_monthend.parquet  (Databento ohlcv-1h, month-end days only, all contracts)
         data/raw/fx_daily.parquet         (front contract per date = prior-day volume winner)
         data/raw/equity/index_close.parquet (yfinance local-currency index closes)
Outputs: oos/fx_fix_v3_results.md, data/artifacts/fx_fix_v3_trades.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = pd.read_parquet(ROOT / "data" / "raw" / "fx_1h_monthend.parquet")
EQ = pd.read_parquet(ROOT / "data" / "raw" / "equity" / "index_close.parquet")
DAILY = pd.read_parquet(ROOT / "data" / "raw" / "fx_daily.parquet")[["date", "root", "front"]]
OUT_MD = ROOT / "oos" / "fx_fix_v3_results.md"
OUT_CSV = ROOT / "data" / "artifacts" / "fx_fix_v3_trades.csv"

ROOTS = ["6E", "6J", "6B", "6A", "6C", "6S", "6N", "6M"]
COST = {r: 2e-4 for r in ROOTS}
COST.update({"6N": 4e-4, "6M": 4e-4})
MIN_VOL = 50
IS = (pd.Timestamp("2010-07-01"), pd.Timestamp("2019-12-31"))
OOS = (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-08-31"))


def last_sunday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() + 1) % 7)


def fix_hour_utc(d: pd.Timestamp) -> int:
    """London 16:00 fix in UTC: 15 during British Summer Time, else 16."""
    return 15 if last_sunday(d.year, 3) <= d.normalize() < last_sunday(d.year, 10) else 16


def eq_close_on_or_before(col: str, d: pd.Timestamp):
    s = EQ[col].dropna()
    s = s[s.index <= d]
    return s.iloc[-1] if len(s) else np.nan


def eq_close_before(col: str, d: pd.Timestamp):
    s = EQ[col].dropna()
    s = s[s.index < d]
    return s.iloc[-1] if len(s) else np.nan


def main() -> None:
    H["ts"] = pd.to_datetime(H["ts_event"], utc=True)
    front = DAILY.set_index(["date", "root"])["front"]
    all_me = list(pd.date_range("2010-06-01", "2026-09-01", freq="BME"))
    me_dates = sorted(pd.to_datetime(H["me_date"]).unique())

    rows = []
    for D in me_dates:
        D = pd.Timestamp(D)
        prev = max(x for x in all_me if x < D)
        fix_h = fix_hour_utc(D)
        t_trade = pd.Timestamp(D.year, D.month, D.day, fix_h - 1, tz="UTC")
        t_post = pd.Timestamp(D.year, D.month, D.day, fix_h, tz="UTC")
        us_rel = np.log(eq_close_before("US", D) / eq_close_on_or_before("US", prev))
        day = H[H["me_date"] == D]
        for r in ROOTS:
            sym = front.get((D, r))
            if sym is None or pd.isna(sym):
                continue
            bar = day[(day["symbol"] == sym) & (day["ts"] == t_trade)]
            if len(bar) != 1 or bar["volume"].iloc[0] < MIN_VOL:
                continue
            rel = np.log(eq_close_before(r, D) / eq_close_on_or_before(r, prev)) - us_rel
            if not np.isfinite(rel) or rel == 0:
                continue
            pos = -np.sign(rel)
            b = bar.iloc[0]
            move = np.log(b["close"] / b["open"])
            post = day[(day["symbol"] == sym) & (day["ts"] == t_post)]
            post_move = np.log(post["close"].iloc[0] / post["open"].iloc[0]) if len(post) == 1 else np.nan
            rows.append(dict(me_date=D, root=r, symbol=sym, rel=rel, pos=pos, move=move,
                             gross=pos * move, net=pos * move - 2 * COST[r],
                             post_rev=-pos * post_move, abs_move=abs(move), volume=int(b["volume"])))
    T = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT_CSV, index=False)

    def window(a, b):
        return T[(T["me_date"] >= a) & (T["me_date"] <= b)]

    def summarize(W: pd.DataFrame) -> dict:
        m = W.groupby("me_date")[["gross", "net"]].mean()          # equal-weight portfolio per month
        out = {}
        for k in ("gross", "net"):
            mu, sd = m[k].mean(), m[k].std()
            out[k] = dict(sharpe=mu / sd * np.sqrt(12) if sd > 0 else np.nan, cum=m[k].sum(),
                          mean_bp=mu * 1e4, months=len(m))
        n = len(W)
        out["t_pooled_net"] = W["net"].mean() / W["net"].std() * np.sqrt(n) if n > 1 else np.nan
        out["hit_gross"] = (W["gross"] > 0).mean()
        out["trade_bp_gross"] = W["gross"].mean() * 1e4
        out["trade_bp_net"] = W["net"].mean() * 1e4
        q = W["rel"].abs().quantile(0.75)
        out["top_q_bp_gross"] = W[W["rel"].abs() >= q]["gross"].mean() * 1e4
        out["post_rev_bp"] = W["post_rev"].mean() * 1e4
        out["uncond_abs_bp"] = W["abs_move"].mean() * 1e4
        out["n_trades"] = n
        return out

    S = {"IS": summarize(window(*IS)), "OOS": summarize(window(*OOS))}
    o, i = S["OOS"], S["IS"]
    passed = (o["net"]["sharpe"] >= 0.5) and (i["net"]["sharpe"] > 0) and (o["net"]["cum"] > 0)

    L = ["# FX v3 — 月末定盘价流效应 — 结果（一次性运行，参数见 PREREGISTRATION_FX_FIX_v3.md）", "",
         "| 窗口 | 月数 | 笔数 | 毛 Sharpe | 净 Sharpe | 净累计 | 每笔毛 bp | 每笔净 bp | 胜率 | 池化 t(净) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for w, s in S.items():
        L.append(f"| {w} | {s['net']['months']} | {s['n_trades']} | {s['gross']['sharpe']:.2f} | "
                 f"{s['net']['sharpe']:.2f} | {s['net']['cum']:+.2%} | {s['trade_bp_gross']:+.1f} | "
                 f"{s['trade_bp_net']:+.1f} | {s['hit_gross']:.1%} | {s['t_pooled_net']:+.2f} |")
    L += ["", f"**主判据：OOS 净 Sharpe {o['net']['sharpe']:.2f}（≥0.5），IS 净 Sharpe {i['net']['sharpe']:.2f}（>0），"
          f"OOS 净累计 {o['net']['cum']:+.2%}（>0）→ {'通过' if passed else '未通过'}**", "",
          "## 副观测", "", "| 窗口 | |rel| 最高四分位每笔毛 bp | 后一小时反转 bp | 无条件定盘前一小时平均绝对波动 bp |",
          "|---|---|---|---|"]
    for w, s in S.items():
        L.append(f"| {w} | {s['top_q_bp_gross']:+.1f} | {s['post_rev_bp']:+.1f} | {s['uncond_abs_bp']:.1f} |")
    L += ["", "## 按货币（全样本，毛 bp / 笔数）", "", "| 货币 | 每笔毛 bp | 笔数 | 胜率 |", "|---|---|---|---|"]
    for r, g in T.groupby("root"):
        L.append(f"| {r} | {g['gross'].mean()*1e4:+.1f} | {len(g)} | {(g['gross']>0).mean():.1%} |")
    L += ["", "## 按年（组合净收益，等权，总敞口 1）", "", "| 年 | 净累计 | 月数 |", "|---|---|---|"]
    m = T.groupby("me_date")["net"].mean()
    for y, g in m.groupby(m.index.year):
        L.append(f"| {y} | {g.sum():+.2%} | {len(g)} |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
