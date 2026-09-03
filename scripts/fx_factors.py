"""G8 FX futures cross-sectional factor backtest — carry / momentum / value / composite.

Usage: python scripts/fx_factors.py [--version v1|v2]
Each version is pre-registered in oos/PREREGISTRATION_FX_FACTORS_<version>.md; every parameter
below is fixed there. IS and OOS are produced in the same run. Writes
oos/fx_factors_<version>_results.md and data/artifacts/fx_factors_<version>_daily.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "raw" / "fx_daily.parquet"
FRED = ROOT / "data" / "raw" / "fred"

REER = {"6E": "RBXMBIS", "6J": "RBJPBIS", "6B": "RBGBBIS", "6A": "RBAUBIS",
        "6C": "RBCABIS", "6S": "RBCHBIS", "6N": "RBNZBIS", "6M": "RBMXBIS"}
RATES = {"6E": "IR3TIB01EZM156N", "6J": "IR3TIB01JPM156N", "6B": "IR3TIB01GBM156N",
         "6A": "IR3TIB01AUM156N", "6C": "IR3TIB01CAM156N", "6S": "IR3TIB01CHM156N",
         "6N": "IR3TIB01NZM156N", "6M": "IR3TIB01MXM156N"}
RATE_USD = "IR3TIB01USM156N"

VERSIONS = {
    # v1: basis-implied carry (next expiry), no vol floor, 4x leverage cap, threshold 0.4
    "v1": dict(carry="basis", carry_win=5, vol_floor=0.0, max_lev=4.0, sharpe_min=0.4),
    # v2: 3m interbank rate carry (1-month lag), 5% vol floor, 2.5x cap, threshold 0.6
    "v2": dict(carry="rates", rate_lag=1, vol_floor=0.05, max_lev=2.5, sharpe_min=0.6),
}
COST = 0.0003          # one-way, per unit notional turnover
TARGET_VOL = 0.10
VOL_WIN = 60
MOM_WIN = 252
VALUE_MONTHS = 60
REER_LAG = 2           # months
IS = ("2011-07-01", "2019-12-31")
OOS = ("2020-01-01", "2026-08-31")
ANN = 252


def load_panel():
    p = pd.read_parquet(PANEL)
    piv = lambda c: p.pivot(index="date", columns="root", values=c).sort_index()
    d = dict(ret=piv("ret").fillna(0.0), ret_oc=piv("ret_oc").fillna(0.0),
             carry=piv("carry"), front=piv("front"))
    d["gap"] = d["ret"] - d["ret_oc"]                 # log(open / prev close), same contract
    return d


def fred_monthly(series: str) -> pd.Series:
    s = pd.read_csv(FRED / f"{series}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s.index = s.index.to_period("M")
    return s


def value_signal(dates: pd.DatetimeIndex, roots) -> pd.DataFrame:
    out = pd.DataFrame(index=dates, columns=roots, dtype=float)
    months = dates.to_period("M") - REER_LAG
    for r in roots:
        s = fred_monthly(REER[r])
        val = -np.log(s / s.shift(VALUE_MONTHS))
        out[r] = val.reindex(months).to_numpy()
    return out


def rate_carry_signal(dates: pd.DatetimeIndex, roots, lag: int):
    """3m interbank rate differential vs USD, month m-lag observation, forward-filled gaps.
    Returns (signal frame, {root: months forward-filled inside the sample})."""
    months = dates.to_period("M") - lag
    full = pd.period_range(months.min(), months.max(), freq="M")
    us = fred_monthly(RATE_USD).reindex(full)
    us_ff = us.ffill()
    out = pd.DataFrame(index=dates, columns=roots, dtype=float)
    filled = {}
    for r in roots:
        s = fred_monthly(RATES[r]).reindex(full)
        filled[r] = int((s.isna() & s.ffill().notna()).sum() + (us.isna() & us_ff.notna()).sum())
        out[r] = (s.ffill() - us_ff).reindex(months).to_numpy() / 100.0
    return out, filled


def rank_weights(sig: pd.Series, vol: pd.Series) -> pd.Series | None:
    s = sig.dropna()
    s = s[vol.reindex(s.index).notna()]
    if len(s) < 4:
        return None
    rk = s.rank() - (len(s) + 1) / 2.0             # demeaned cross-sectional rank
    w = rk / vol[s.index]
    w = w / w.abs().sum()
    return w.reindex(sig.index).fillna(0.0)


def simulate(d, schedule: dict, lev: pd.Series | None):
    """schedule: {exec_date: unscaled weights (gross 1)}; lev: leverage per exec_date (or None=1).
    Returns (daily net return, daily gross return, daily turnover, daily gross exposure)."""
    ret, ret_oc, gap, front = d["ret"], d["ret_oc"], d["gap"], d["front"]
    roots = ret.columns
    w = pd.Series(0.0, index=roots)
    net, gross, turn, expo = [], [], [], []
    prev_front = None
    for day in ret.index:
        r_day = ret.loc[day]
        cost = 0.0
        if day in schedule:
            w_new = schedule[day] * (1.0 if lev is None else lev.loc[day])
            r = float((w * gap.loc[day]).sum() + (w_new * ret_oc.loc[day]).sum())
            t = float((w_new - w).abs().sum())
            cost += COST * t
            w = w_new
        else:
            r = float((w * r_day).sum())
            t = 0.0
        if prev_front is not None:
            rolled = (front.loc[day] != prev_front) & front.loc[day].notna() & prev_front.notna()
            if rolled.any():
                cost += 2 * COST * float(w[rolled].abs().sum())
        prev_front = front.loc[day]
        gross.append(r)
        net.append(r - cost)
        turn.append(t)
        expo.append(float(w.abs().sum()))
    idx = ret.index
    return (pd.Series(net, idx), pd.Series(gross, idx), pd.Series(turn, idx), pd.Series(expo, idx))


def metrics(r: pd.Series, turn: pd.Series | None = None, expo: pd.Series | None = None):
    if len(r) == 0:
        return {}
    mu, sd = r.mean() * ANN, r.std() * np.sqrt(ANN)
    sr = mu / sd if sd > 0 else np.nan
    eq = np.exp(r.cumsum())
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(r) / ANN
    se = np.sqrt((1 + sr ** 2 / 2) / yrs) if yrs > 0 else np.nan
    out = dict(ann_ret=mu, ann_vol=sd, sharpe=sr, sharpe_se=se, max_dd=dd, years=yrs)
    if turn is not None:
        out["turnover_pa"] = turn.sum() / yrs
    if expo is not None:
        out["avg_lev"] = expo.mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", choices=sorted(VERSIONS))
    ver = ap.parse_args().version
    cfg = VERSIONS[ver]
    out_md = ROOT / "oos" / f"fx_factors_{ver}_results.md"
    out_csv = ROOT / "data" / "artifacts" / f"fx_factors_{ver}_daily.csv"

    d = load_panel()
    ret = d["ret"]
    dates, roots = ret.index, ret.columns

    vol = (ret.rolling(VOL_WIN).std() * np.sqrt(ANN)).clip(lower=cfg["vol_floor"])
    notes = []
    if cfg["carry"] == "basis":
        carry_sig = d["carry"].rolling(cfg["carry_win"]).mean()
    else:
        carry_sig, filled = rate_carry_signal(dates, roots, cfg["rate_lag"])
        notes.append("利率序列前向填充月数（含美元）：" + ", ".join(f"{k} {v}" for k, v in filled.items()))
    sig = {"carry": carry_sig,
           "momentum": ret.rolling(MOM_WIN).sum(),
           "value": value_signal(dates, roots)}

    # weekly: signal on the last trading day of each ISO week, executed next trading day
    iso = dates.isocalendar()
    key = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    last_of_week = pd.Series(dates, index=dates).groupby(key.values).max().sort_values()
    pos = {dt: i for i, dt in enumerate(dates)}
    pairs = [(s, dates[pos[s] + 1]) for s in last_of_week if pos[s] + 1 < len(dates)]

    sched = {k: {} for k in list(sig) + ["composite"]}
    for s_day, x_day in pairs:
        ws = {}
        for k, S in sig.items():
            w = rank_weights(S.loc[s_day], vol.loc[s_day])
            if w is not None:
                ws[k] = w
                sched[k][x_day] = w
        if len(ws) == 3:
            comp = sum(ws.values()) / 3.0
            sched["composite"][x_day] = comp / comp.abs().sum()

    results, daily = {}, {}
    for k, sc in sched.items():
        _, r_u, _, _ = simulate(d, sc, None)               # pass 1: unscaled, for vol targeting
        rv = r_u.rolling(VOL_WIN).std() * np.sqrt(ANN)
        lev = (TARGET_VOL / rv).clip(upper=cfg["max_lev"]).shift(1)   # history up to signal day
        lev_exec = pd.Series({x: lev.loc[s] for s, x in pairs if x in sc}).dropna()
        sc2 = {x: w for x, w in sc.items() if x in lev_exec.index}
        net, gross, turn, expo = simulate(d, sc2, lev_exec)   # pass 2: scaled + costs
        daily[k] = net
        results[k] = {}
        for name, (a, b) in {"IS": IS, "OOS": OOS}.items():
            m = slice(pd.Timestamp(a), pd.Timestamp(b))
            results[k][name] = dict(net=metrics(net.loc[m], turn.loc[m], expo.loc[m]),
                                    gross=metrics(gross.loc[m]))

    lines = [f"# FX 三因子组合 {ver} — 结果（一次性运行，参数见 PREREGISTRATION_FX_FACTORS_{ver}.md）", ""]
    lines += [f"- {n}" for n in notes] + ([""] if notes else [])
    lines += ["| 因子 | 窗口 | 年化净收益 | 年化波动 | 净 Sharpe (±SE) | 毛 Sharpe | 最大回撤 | 年换手 | 平均杠杆 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k in results:
        for win in ("IS", "OOS"):
            n, g = results[k][win]["net"], results[k][win]["gross"]
            lines.append(f"| {k} | {win} | {n['ann_ret']:+.2%} | {n['ann_vol']:.2%} | "
                         f"{n['sharpe']:.2f} (±{n['sharpe_se']:.2f}) | {g['sharpe']:.2f} | "
                         f"{n['max_dd']:.1%} | {n['turnover_pa']:.1f}x | {n['avg_lev']:.2f} |")
    comp = results["composite"]["OOS"]["net"]
    passed = comp["sharpe"] >= cfg["sharpe_min"] and comp["ann_ret"] > 0
    lines += ["", f"**主判据：组合 OOS 净 Sharpe {comp['sharpe']:.2f}，净收益 {comp['ann_ret']:+.2%} → "
              f"{'通过' if passed else '未通过'}（门槛 Sharpe ≥ {cfg['sharpe_min']} 且收益 > 0）**", "",
              "## 组合按年净收益", "", "| 年 | 净收益 | 年内最大回撤 |", "|---|---|---|"]
    c = daily["composite"].loc[IS[0]:]
    for y, ry in c.groupby(c.index.year):
        eq = np.exp(ry.cumsum())
        lines.append(f"| {y} | {ry.sum():+.2%} | {(eq / eq.cummax() - 1).min():.1%} |")
    lines += ["", "## 副观测窗口（净）", ""]
    for label, (a, b) in {"2015-01-14→2015-01-16": ("2015-01-14", "2015-01-16"),
                          "2020-02-15→2020-04-30": ("2020-02-15", "2020-04-30"),
                          "2022-01-01→2022-12-31": ("2022-01-01", "2022-12-31")}.items():
        parts = ", ".join(f"{k} {daily[k].loc[a:b].sum():+.2%}" for k in daily)
        lines.append(f"- {label}: {parts}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(daily).to_csv(out_csv)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
