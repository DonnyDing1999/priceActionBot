"""G8 FX futures cross-sectional factor backtest — carry / momentum / value / composite.

Pre-registered in oos/PREREGISTRATION_FX_FACTORS_v1.md; every parameter below is fixed there.
IS and OOS are produced in the same run. Writes oos/fx_factors_v1_results.md and
data/artifacts/fx_factors_v1_daily.csv (daily net returns per factor).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "raw" / "fx_daily.parquet"
FRED = ROOT / "data" / "raw" / "fred"
OUT_MD = ROOT / "oos" / "fx_factors_v1_results.md"
OUT_CSV = ROOT / "data" / "artifacts" / "fx_factors_v1_daily.csv"

REER = {"6E": "RBXMBIS", "6J": "RBJPBIS", "6B": "RBGBBIS", "6A": "RBAUBIS",
        "6C": "RBCABIS", "6S": "RBCHBIS", "6N": "RBNZBIS", "6M": "RBMXBIS"}
COST = 0.0003          # one-way, per unit notional turnover
TARGET_VOL = 0.10
MAX_LEV = 4.0
VOL_WIN = 60
MOM_WIN = 252
CARRY_WIN = 5
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


def value_signal(dates: pd.DatetimeIndex, roots) -> pd.DataFrame:
    out = pd.DataFrame(index=dates, columns=roots, dtype=float)
    months = dates.to_period("M") - REER_LAG
    for r in roots:
        s = pd.read_csv(FRED / f"{REER[r]}.csv", index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce")
        s.index = s.index.to_period("M")
        val = -np.log(s / s.shift(VALUE_MONTHS))
        out[r] = val.reindex(months).to_numpy()
    return out


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
    Returns (daily net return, daily gross return, daily turnover)."""
    ret, ret_oc, gap, front = d["ret"], d["ret_oc"], d["gap"], d["front"]
    roots = ret.columns
    w = pd.Series(0.0, index=roots)
    net, gross, turn = [], [], []
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
    idx = ret.index
    return (pd.Series(net, idx), pd.Series(gross, idx), pd.Series(turn, idx))


def metrics(r: pd.Series, turn: pd.Series | None = None):
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
    return out


def main() -> None:
    d = load_panel()
    ret, carry = d["ret"], d["carry"]
    dates, roots = ret.index, ret.columns

    vol = ret.rolling(VOL_WIN).std() * np.sqrt(ANN)
    sig = {"carry": carry.rolling(CARRY_WIN).mean(),
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
        _, r_u, _ = simulate(d, sc, None)                  # pass 1: unscaled, for vol targeting
        rv = r_u.rolling(VOL_WIN).std() * np.sqrt(ANN)
        lev = (TARGET_VOL / rv).clip(upper=MAX_LEV)
        lev = lev.shift(1)                                  # only history up to the signal day
        lev_exec = pd.Series({x: lev.loc[s] for s, x in pairs if x in sc}).dropna()
        sc2 = {x: w for x, w in sc.items() if x in lev_exec.index}
        net, gross, turn = simulate(d, sc2, lev_exec)     # pass 2: scaled + costs
        daily[k] = net
        results[k] = {}
        for name, (a, b) in {"IS": IS, "OOS": OOS}.items():
            m = slice(pd.Timestamp(a), pd.Timestamp(b))
            results[k][name] = dict(net=metrics(net.loc[m], turn.loc[m]), gross=metrics(gross.loc[m]))

    lines = ["# FX 三因子组合 v1 — 结果（一次性运行，参数见 PREREGISTRATION_FX_FACTORS_v1.md）", "",
             "| 因子 | 窗口 | 年化净收益 | 年化波动 | 净 Sharpe (±SE) | 毛 Sharpe | 最大回撤 | 年换手 |",
             "|---|---|---|---|---|---|---|---|"]
    for k in results:
        for win in ("IS", "OOS"):
            n, g = results[k][win]["net"], results[k][win]["gross"]
            lines.append(f"| {k} | {win} | {n['ann_ret']:+.2%} | {n['ann_vol']:.2%} | "
                         f"{n['sharpe']:.2f} (±{n['sharpe_se']:.2f}) | {g['sharpe']:.2f} | "
                         f"{n['max_dd']:.1%} | {n['turnover_pa']:.1f}x |")
    comp = results["composite"]["OOS"]["net"]
    passed = comp["sharpe"] >= 0.4 and comp["ann_ret"] > 0
    lines += ["", f"**主判据：组合 OOS 净 Sharpe {comp['sharpe']:.2f}，净收益 {comp['ann_ret']:+.2%} → "
              f"{'通过' if passed else '未通过'}（门槛 Sharpe ≥ 0.4 且收益 > 0）**", "",
              "## 组合按年净收益", "", "| 年 | 净收益 | 年内最大回撤 |", "|---|---|---|"]
    c = daily["composite"].loc[IS[0]:]
    for y, ry in c.groupby(c.index.year):
        eq = np.exp(ry.cumsum())
        lines.append(f"| {y} | {ry.sum():+.2%} | {(eq / eq.cummax() - 1).min():.1%} |")
    lines += ["", "## 副观测窗口（组合，净）", ""]
    for label, (a, b) in {"2020-02-15→2020-04-30": ("2020-02-15", "2020-04-30"),
                          "2022-01-01→2022-12-31": ("2022-01-01", "2022-12-31")}.items():
        seg = daily["composite"].loc[a:b]
        lines.append(f"- {label}: 累计 {seg.sum():+.2%}, 最大回撤 {metrics(seg)['max_dd']:.1%}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(daily).to_csv(OUT_CSV)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
