"""Capital-efficiency and performance metrics over a set of trades.

Answers the questions a funder would ask: how hard is the capital working
(exposure, margin, turnover), what is it earning per unit of risk (ROC, Sharpe,
Calmar), and what does the trading actually look like (holding times, streaks,
risk-budget usage, cost drag). Works on Trade dataclasses or journal dicts, for
backtest and live journals alike.

All capital-relative numbers are parameterized by `capital` (the account you'd
run this on) — pure reporting, nothing here feeds back into decisions.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, is_dataclass


def _minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def capital_metrics(trades, *, capital: float = 5000.0, point_value: float = 5.0,
                    day_margin: float = 50.0, commission_rt: float = 1.5,
                    tick_value: float = 1.25, rth_minutes: int = 390,
                    n_sessions: int | None = None, trading_days_year: int = 252) -> dict:
    """Metric dict over closed trades. `n_sessions` = sessions OBSERVED (incl.
    no-trade days); defaults to the number of distinct sessions with trades."""
    rows = [asdict(t) if is_dataclass(t) else dict(t) for t in trades]
    if not rows:
        return {"trades": 0}

    sessions = sorted({r["session"] for r in rows})
    n_sess = n_sessions or len(sessions)
    pnl = [r["pnl_usd"] for r in rows]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)

    # ---- capital utilization ----
    holds = [max(0, _minutes(r["exit_ts"]) - _minutes(r["entry_ts"])) for r in rows]
    in_market_min = sum(holds)
    share_rth = in_market_min / (n_sess * rth_minutes)
    notionals = [r["entry"] * point_value for r in rows]
    leverage_in_market = statistics.mean(notionals) / capital
    tw_exposure = leverage_in_market * share_rth
    turnover = sum(notionals) / capital            # notional traded per unit capital

    # ---- daily series -> risk-adjusted returns ----
    by_day: dict[str, float] = {}
    for r in rows:
        by_day[r["session"]] = by_day.get(r["session"], 0.0) + r["pnl_usd"]
    daily = [by_day.get(s, 0.0) for s in sessions]
    eq = peak = mdd = 0.0
    for p in daily:
        eq += p
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    total = sum(pnl)
    daily_mean = total / n_sess                    # incl. flat days as 0
    daily_sd = statistics.stdev(daily + [0.0] * (n_sess - len(daily))) if n_sess > 2 else 0.0
    downside = [min(0.0, p) for p in daily] + [0.0] * (n_sess - len(daily))
    dd_sd = statistics.pstdev(downside) if downside else 0.0
    ann_factor = trading_days_year ** 0.5
    ann_return = daily_mean * trading_days_year
    sharpe = (daily_mean / daily_sd * ann_factor) if daily_sd > 0 else None
    sortino = (daily_mean / dd_sd * ann_factor) if dd_sd > 0 else None
    calmar = (ann_return / -mdd) if mdd < 0 else None

    # ---- streaks / risk budget / cost drag ----
    max_w = max_l = cw = cl = 0
    for p in pnl:
        cw, cl = (cw + 1, 0) if p > 0 else (0, cl + 1)
        max_w, max_l = max(max_w, cw), max(max_l, cl)
    risk_usd = [r["risk_pts"] * point_value for r in rows]
    cap_usage = [min(1.0, ru / 75.0) for ru in risk_usd]
    n_stop = sum(1 for r in rows if r["exit_reason"] == "stop")
    est_costs = len(rows) * (commission_rt + tick_value) + n_stop * tick_value

    return {
        "capital": capital,
        "utilization": {
            "day_margin_pct_of_capital": round(100 * day_margin / capital, 2),
            "avg_notional_usd": round(statistics.mean(notionals), 0),
            "leverage_while_in_market": round(leverage_in_market, 2),
            "time_in_market_min_per_day": round(in_market_min / n_sess, 1),
            "share_of_rth_pct": round(100 * share_rth, 1),
            "time_weighted_exposure_x": round(tw_exposure, 3),
            "notional_turnover_x_per_session": round(turnover / n_sess, 2),
        },
        "returns": {
            "total_usd": round(total, 2),
            "roc_period_pct": round(100 * total / capital, 2),
            "roc_annualized_pct_naive": round(100 * ann_return / capital, 1),
            "sharpe_daily_ann": round(sharpe, 2) if sharpe is not None else None,
            "sortino_daily_ann": round(sortino, 2) if sortino is not None else None,
            "max_drawdown_usd": round(mdd, 2),
            "max_drawdown_pct_of_capital": round(100 * mdd / capital, 2),
            "calmar_naive": round(calmar, 2) if calmar is not None else None,
            "positive_day_pct": round(100 * sum(1 for p in daily if p > 0) / n_sess, 1),
            "best_day_usd": round(max(daily), 2),
            "worst_day_usd": round(min(daily), 2),
        },
        "trading": {
            "trades": len(rows),
            "sessions_observed": n_sess,
            "trades_per_session": round(len(rows) / n_sess, 2),
            "participation_pct": round(100 * len(sessions) / n_sess, 1),
            "holding_min_median": statistics.median(holds),
            "holding_min_mean": round(statistics.mean(holds), 1),
            "holding_min_max": max(holds),
            "win_rate_pct": round(100 * len(wins) / len(rows), 1),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
            "expectancy_usd": round(total / len(rows), 2),
            "avg_win_usd": round(statistics.mean(wins), 2) if wins else None,
            "avg_loss_usd": round(statistics.mean(losses), 2) if losses else None,
            "risk_budget_usage_median_pct": round(100 * statistics.median(cap_usage), 0),
            "max_win_streak": max_w,
            "max_loss_streak": max_l,
            "est_friction_usd": round(est_costs, 2),
            "friction_pct_of_gross_wins": round(100 * est_costs / gross_win, 1)
            if gross_win else None,
        },
    }
