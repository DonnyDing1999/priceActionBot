"""Engine + risk-layer correctness tests (synthetic bars, no LLM, no network).

These lock in the fill semantics the whole evaluation rests on:
market entry, Brooks stop entry (trigger / gap-through / cancel), 1m intrabar
stop-vs-target ordering, pre-fill price-action exclusion, and the RiskManager gates.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.backtest import Config, Signal, run_session  # noqa: E402
from pab.bars import ET  # noqa: E402
from pab.risk import RiskManager  # noqa: E402

SESSION = "2026-07-06"


def mk5(bars):
    """5m frame from (o,h,l,c) tuples starting 09:30 ET."""
    idx = pd.date_range(f"{SESSION} 09:30", periods=len(bars), freq="5min", tz=ET)
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 100}
         for o, h, l, c in bars], index=idx)


def mk1(start_hhmm, bars):
    """1m frame from (o,h,l,c) tuples starting at start_hhmm ET."""
    idx = pd.date_range(f"{SESSION} {start_hhmm}", periods=len(bars), freq="1min", tz=ET)
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 20}
         for o, h, l, c in bars], index=idx)


def fire_at(bar_index, signal):
    """Strategy that fires `signal` only at the given bar_index_from_open."""
    def strat(sidecar, bars):
        return signal if sidecar["bar_index_from_open"] == bar_index else None
    return strat


# ---------- RiskManager ----------

def test_risk_geometry_reject():
    rm = RiskManager(Config())
    assert rm.validate_entry("long", 100.0, 101.0, 105.0).reason == "geometry"
    assert rm.validate_entry("short", 100.0, 99.0, 95.0).reason == "geometry"


def test_risk_rr_floor_and_cap():
    rm = RiskManager(Config())
    assert rm.validate_entry("long", 100.0, 98.0, 101.0).reason == "rr_below_floor"
    assert rm.validate_entry("long", 100.0, 84.0, 120.0).reason == "risk_gt_cap"
    assert rm.validate_entry("long", 100.0, 95.0, 106.0).ok


def test_risk_micro_stop_floor():
    rm = RiskManager(Config())
    # 0.5pt risk: fixed costs (~0.8pt) exceed the risk itself -> reject
    assert rm.validate_entry("long", 100.0, 99.5, 101.0).reason == "risk_too_small"
    assert rm.validate_entry("long", 100.0, 98.0, 102.5).ok      # 2.0pt = at the floor


def test_risk_circuit_breakers():
    rm = RiskManager(Config())
    rm.on_trade_closed(-10); rm.on_trade_closed(-10)
    assert rm.session_halted() == (True, "consec_loss")
    rm = RiskManager(Config())
    for _ in range(3):
        rm.on_trade_closed(+10)
    assert rm.session_halted() == (True, "max_trades")
    rm = RiskManager(Config())
    rm.on_trade_closed(-500)
    assert rm.session_halted()[0]


# ---------- entries ----------

def test_market_entry_fills_next_open():
    cont = mk5([(100, 101, 99, 100.5), (101, 102, 100.5, 101), (105, 111, 104, 110)])
    sig = Signal("long", stop=95.5, target=110.5, entry_type="market")
    trades = run_session(cont, SESSION, fire_at(1, sig))
    assert len(trades) == 1
    t = trades[0]
    assert t.entry == 101.25          # next open 101 + 1 tick slip
    assert t.exit_reason == "target" and t.exit == 110.5


def test_stop_entry_triggers_at_ref():
    cont = mk5([(100, 101, 99, 100.5), (100.75, 101.5, 100.25, 101),
                (102, 106.75, 101.5, 106.5)])
    sig = Signal("long", stop=96.0, target=106.5, entry_type="stop")
    stats = {}
    trades = run_session(cont, SESSION, fire_at(1, sig), stats=stats)
    assert len(trades) == 1
    # trigger = signal bar high 101 + tick = 101.25; fill = trigger + slip
    assert trades[0].entry == 101.5
    assert trades[0].risk_pts == 5.25  # measured from the trigger price, not bar close
    assert stats.get("no_fill", 0) == 0


def test_stop_entry_gap_through_fills_at_open():
    cont = mk5([(100, 101, 99, 100.5), (102, 103, 101.5, 102.5),
                (103, 107, 102.5, 106.6)])
    sig = Signal("long", stop=96.0, target=106.5, entry_type="stop")
    trades = run_session(cont, SESSION, fire_at(1, sig))
    assert len(trades) == 1
    assert trades[0].entry == 102.25   # gapped past 101.25 -> next open 102 + slip


def test_stop_entry_cancels_when_never_triggered():
    cont = mk5([(100, 101, 99, 100.5), (100.5, 101.0, 99.9, 100),
                (100, 100.5, 99.5, 100)])
    sig = Signal("long", stop=96.0, target=106.5, entry_type="stop")
    stats = {}
    trades = run_session(cont, SESSION, fire_at(1, sig), stats=stats)
    assert trades == []                # bar 2 high 101.0 < trigger 101.25 -> canceled
    assert stats["no_fill"] == 1


# ---------- 1m intrabar resolution ----------

def test_m1_resolves_target_before_stop():
    # 5m fill bar spans BOTH stop and target; 1m shows target touched first.
    cont = mk5([(100, 100.5, 99.5, 100), (100, 103.6, 97.4, 100.5),
                (100, 101, 99, 100)])
    sig = Signal("long", stop=98.0, target=103.0, entry_type="market")
    no_m1 = run_session(cont, SESSION, fire_at(1, sig))
    assert no_m1[0].exit_reason == "stop"          # 5m-only: conservative stop-first
    m1 = mk1("09:35", [(100, 100.4, 99.8, 100.2), (100.2, 103.6, 100, 103),
                       (103, 103.2, 97.4, 98)])
    with_m1 = run_session(cont, SESSION, fire_at(1, sig), m1=m1)
    assert with_m1[0].exit_reason == "target"      # 1m truth: target minute came first
    assert with_m1[0].exit == 103.0


def test_m1_ignores_prefill_price_action():
    # Bar 2 dips through the stop BEFORE the stop-entry triggers; only post-fill
    # minutes may exit the trade.
    cont = mk5([(100, 101, 99, 100.5), (100, 101.5, 97.0, 101),
                (101.5, 105.75, 101, 105.6)])
    sig = Signal("long", stop=97.25, target=105.5, entry_type="stop")
    no_m1 = run_session(cont, SESSION, fire_at(1, sig))
    assert no_m1[0].exit_reason == "stop"          # 5m fallback can't know the order
    m1 = mk1("09:35", [(100, 100.2, 97.0, 98), (98, 100.8, 98, 100.5),
                       (100.5, 101.5, 100.9, 101.2), (101.2, 101.3, 100.2, 101),
                       (101, 101.2, 100.8, 101)])
    with_m1 = run_session(cont, SESSION, fire_at(1, sig), m1=m1)
    assert len(with_m1) == 1
    assert with_m1[0].exit_reason == "target"      # pre-trigger dip correctly excluded
    assert with_m1[0].entry == 101.5               # trigger 101.25 + slip


# ---------- exits may run past the entry window ----------

def test_exit_runs_past_window_to_target():
    # Entry window ends at 09:40; the target is only reached at 09:50 — the position
    # keeps working past the window instead of being force-flattened.
    cfg = Config(window_end="09:40", flat_by="10:00")
    cont = mk5([(100, 101, 99, 100.5),      # 09:30 signal bar
                (101, 102, 100.5, 101.5),   # 09:35 fill bar, no exit
                (101.5, 102, 100.8, 101),   # 09:40 window ends, still open
                (101, 103, 100.9, 102.8),   # 09:45
                (103, 106.75, 102.5, 106.5),  # 09:50 target hit
                (106, 106.5, 105, 105.5),   # 09:55
                (105.5, 106, 105, 105.8)])  # 10:00
    sig = Signal("long", stop=95.5, target=106.5, entry_type="market")
    trades = run_session(cont, SESSION, fire_at(1, sig), cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "target" and t.exit == 106.5
    assert t.exit_ts == "09:50"             # after the 09:40 entry-window end


def test_eod_force_flat_when_nothing_hits():
    cfg = Config(window_end="09:40", flat_by="09:55")
    cont = mk5([(100, 101, 99, 100.5), (101, 102, 100.5, 101.5),
                (101.5, 102, 100.8, 101), (101, 102.5, 100.9, 102),
                (102, 102.5, 101.5, 102), (102, 102.8, 101.8, 102.5)])  # ..09:55
    sig = Signal("long", stop=95.5, target=120.0, entry_type="market")
    trades = run_session(cont, SESSION, fire_at(1, sig), cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "eod" and t.exit_ts == "09:55"
    assert t.exit == 102.5                  # close of the flat_by bar


# ---------- strategy gate ----------

def _sc(bar_index, bar_type="trend_bull", close=100.0, ema_gap=5.0,
        s_hi=110.0, s_lo=90.0):
    return {"bar_index_from_open": bar_index,
            "bar": {"type": bar_type, "c": close},
            "close_vs_ema_pts": ema_gap, "session_high": s_hi, "session_low": s_lo}


def test_gate_blocks_late_and_doji_bars():
    from pab.llm_strategy import obvious_no_trade
    assert obvious_no_trade(_sc(20)) == "too_late_in_window"
    assert obvious_no_trade(_sc(24)) == "too_late_in_window"
    assert obvious_no_trade(_sc(19)) is None                    # last allowed signal bar
    assert obvious_no_trade(
        _sc(10, bar_type="doji", ema_gap=1.0)) == "doji_mid_range_near_ema"
    assert obvious_no_trade(_sc(10, bar_type="trend_bull")) is None
    assert obvious_no_trade(_sc(10, bar_type="doji", ema_gap=1.0,
                                close=109.0)) is None           # near session high: ask LLM


# ---------- decision latency (live parity) ----------

def test_latency_misses_early_trigger():
    # Stop trigger (101.25) is touched ONLY in minute 0 of the fill bar; with 60s
    # decision latency the order isn't working yet -> no fill. Instant -> fills.
    cont = mk5([(100, 101, 99, 100.5), (100, 101.5, 99.8, 100),
                (100, 100.8, 99.5, 100.2)])
    m1 = mk1("09:35", [(100, 101.5, 99.9, 100.1),      # minute 0: only touch of 101.25
                       (100.1, 100.9, 99.9, 100.2), (100.2, 100.8, 100, 100.4),
                       (100.4, 100.9, 99.8, 100), (100, 100.6, 99.8, 100)])
    sig = Signal("long", stop=96.0, target=106.5, entry_type="stop")
    instant = run_session(cont, SESSION, fire_at(1, sig), m1=m1)
    assert len(instant) == 1 and instant[0].entry == 101.5
    stats = {}
    late = run_session(cont, SESSION, fire_at(1, sig),
                       Config(decision_latency_s=60), m1=m1, stats=stats)
    assert late == [] and stats["no_fill"] == 1


def test_latency_market_fills_at_activation_minute():
    cont = mk5([(100, 101, 99, 100.5), (100, 102, 99.8, 101.5),
                (101.5, 111, 101, 110.6)])
    m1 = mk1("09:35", [(100, 100.5, 99.8, 100.2), (100.9, 101.2, 100.7, 101),
                       (101, 101.5, 100.8, 101.2), (101.2, 101.8, 101, 101.5),
                       (101.5, 102, 101.3, 101.5)])
    sig = Signal("long", stop=95.5, target=110.5, entry_type="market")
    lat = run_session(cont, SESSION, fire_at(1, sig),
                      Config(decision_latency_s=60), m1=m1)
    assert len(lat) == 1
    assert lat[0].entry == 101.15     # minute-1 open 100.9 + slip, not bar open 100


# ---------- diagnose -> route + magnets ----------

def test_classify_regime():
    from pab.features import classify_regime
    assert classify_regime({"bar_index_from_open": 3, "always_in_hint": "ail",
                            "ema_slope_pts_3bar": 5.0, "close_vs_ema_pts": 9.0}) == "open"
    assert classify_regime({"bar_index_from_open": 10, "always_in_hint": "ail",
                            "ema_slope_pts_3bar": 2.0, "close_vs_ema_pts": 3.0}) == "trend"
    assert classify_regime({"bar_index_from_open": 10, "always_in_hint": "neutral",
                            "ema_slope_pts_3bar": 2.0, "close_vs_ema_pts": 9.0}) == "range"
    assert classify_regime({"bar_index_from_open": 10, "always_in_hint": "ais",
                            "ema_slope_pts_3bar": -0.2, "close_vs_ema_pts": -2.0}) == "range"


def test_sidecar_magnets():
    from pab.features import build_sidecar
    d1 = pd.date_range("2026-07-02 09:30", periods=3, freq="5min", tz=ET)  # prior RTH
    on = pd.date_range("2026-07-02 18:00", periods=2, freq="5min", tz=ET)  # overnight
    d2 = pd.date_range(f"{SESSION} 09:30", periods=2, freq="5min", tz=ET)  # today
    rows = [(100, 105, 99, 104), (104, 106, 103, 105), (105, 107, 104, 106),
            (106, 108, 105.5, 107), (107, 109, 106, 108.5),
            (110, 111, 109, 110.5), (110.5, 112, 110, 111)]
    cont = pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 50}
         for o, h, l, c in rows], index=d1.append(on).append(d2))
    sc = build_sidecar(cont, d2[1])
    m = {x["name"]: x for x in sc["magnets"]}
    assert m["yday_high"]["px"] == 107 and m["yday_low"]["px"] == 99
    assert m["yday_close"]["px"] == 106
    assert m["overnight_high"]["px"] == 109 and m["overnight_low"]["px"] == 105.5
    assert m["yday_high"]["pts_from_close"] == -4.0   # 107 vs close 111


def test_routed_system_smaller_than_full():
    from pab.agent import _system
    full = _system(False, None)
    for rg in ("open", "trend", "range"):
        routed = _system(False, rg)
        assert len(routed) < len(full) * 0.8, rg
        assert "DIAGNOSIS HINT" in routed and rg in routed


# ---------- live line: instruments + 5m aggregation ----------

def test_spy_spec_sidecar_config():
    from pab.features import build_sidecar
    from pab.instruments import SPY
    cont = mk5([(760.0, 761.0, 759.0, 760.5), (760.5, 761.5, 760.0, 761.0)])
    sc = build_sidecar(cont, cont.index[-1], symbol="SPY", spec=SPY)
    cfg = sc["config"]
    assert cfg["tick"] == 0.01 and cfg["point_value_usd"] == 50.0
    assert cfg["per_trade_risk_pts"] == 1.5 and cfg["qty"] == 50
    # default (spec=None) keeps the historical MES config byte-identical
    mes = build_sidecar(cont, cont.index[-1])["config"]
    assert mes == {"tick": 0.25, "point_value_usd": 5.0,
                   "per_trade_risk_usd": 75.0, "per_trade_risk_pts": 15.0}


def test_mes_system_prompt_unchanged_by_instrument_refactor():
    # the in-flight eval's decision-cache keys depend on these exact bytes
    from pab.agent import _system
    s = _system(False, "trend", "mes")
    assert s.startswith("You are an Al Brooks price-action trader trading 1 micro "
                        "contract of MES (Micro E-mini S&P 500) on the 5-minute chart. "
                        "You ENTER only during the first two hours")
    spy = _system(False, "trend", "spy")
    assert "50 shares of SPY" in spy and spy != s


def test_live_config_scales_with_qty(monkeypatch):
    from pab.instruments import SPY
    from pab.live import LiveConfig
    c50 = LiveConfig(SPY, 50)
    assert c50.point_value == 50.0 and c50.per_trade_risk_usd == 75.0
    c10 = LiveConfig(SPY, 10)          # smaller size -> wider allowed stop in pts
    assert c10.point_value == 10.0
    assert c10.per_trade_risk_usd / c10.point_value == 7.5
    monkeypatch.setenv("PA_RISK_USD", "40")
    monkeypatch.setenv("PA_DAILY_LOSS_CAP", "200")
    c = LiveConfig(SPY, 50)
    assert c.per_trade_risk_usd == 40.0 and c.daily_loss_cap == 200.0


def test_live_to_5m_closed_only():
    from pab.live import to_5m
    now = pd.Timestamp.now(tz=ET)
    start = (now - pd.Timedelta(minutes=13)).floor("5min")  # spans 3 buckets
    idx = pd.date_range(start, periods=13, freq="1min")
    m1 = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.5, "volume": 10}, index=idx)
    bars = to_5m(m1)
    assert len(bars) >= 1
    assert bars.index[0] == start                      # START-stamped
    assert all(ts + pd.Timedelta(minutes=5) <= now for ts in bars.index)  # closed only
    assert bars.iloc[0]["volume"] == 50                # 5 x 1m aggregated


# ---------- v4: perception features, experience date filter, day gate ----------

def test_sidecar_v4_features():
    from pab.features import build_sidecar
    # up-leg then pullback low then higher high -> swings + table + chop present
    cont = mk5([(100, 101, 99.5, 100.8), (100.8, 102, 100.5, 101.8),
                (101.8, 103, 101.5, 102.8), (102.8, 103.2, 101.0, 101.4),
                (101.4, 102.0, 100.8, 101.8), (101.8, 104, 101.6, 103.8)])
    sc = build_sidecar(cont, cont.index[-1])
    assert "session_table" in sc and "recent_bars" not in sc and "session_bars" not in sc
    lines = sc["session_table"].splitlines()
    assert lines[0].startswith("i|time|") and len(lines) == 7   # header + 6 bars
    assert lines[1].startswith("1|09:30|100|101|99.5|100.8|")
    assert sc["chop"] is not None and 0 <= sc["chop"]["overlap_session"] <= 1
    assert sc["swings"] == ["SH@103.2(bar4)", "SL@100.8(bar5)"]  # confirmed 1-bar pivots


def test_last_bars_pattern_inside_outside():
    from pab.features import build_sidecar
    inside = mk5([(100, 103, 99, 102), (102, 102.5, 100.5, 101)])
    assert build_sidecar(inside, inside.index[-1])["last_bars_pattern"] == "inside"
    outside = mk5([(100, 101, 99.5, 100.5), (100.5, 102, 99, 101.5)])
    assert build_sidecar(outside, outside.index[-1])["last_bars_pattern"] == "outside"


def test_read_cases_date_filter(tmp_path, monkeypatch):
    import pab.experience as ex
    f = tmp_path / "cases.jsonl"
    f.write_text('{"session":"2026-03-01","regime":"spike","note":"a"}\n'
                 '{"session":"2026-03-05","regime":"spike","note":"b"}\n', "utf-8")
    monkeypatch.setattr(ex, "CASES", f)
    assert len(ex.read_cases(k=9)) == 2
    got = ex.read_cases(k=9, before="2026-03-05")
    assert len(got) == 1 and got[0]["note"] == "a"    # same-or-later lessons excluded
    assert ex.read_cases(k=9, before="2026-03-01") == []


def test_dayfilter_grades():
    from pab.dayfilter import grade
    assert grade(None) == "A"                          # no data -> no filter
    chop = {"overlap": 0.70, "efficiency": 0.10, "small_body": 0.5, "range_vs_adr": 1.0}
    trend = {"overlap": 0.40, "efficiency": 0.80, "small_body": 0.2, "range_vs_adr": 1.2}
    assert grade(chop, "overlap") == "C" and grade(trend, "overlap") == "A"
    assert grade(chop, "combo") == "C" and grade(trend, "combo") == "A"


# ---------- capital metrics ----------

def test_capital_metrics():
    from pab.metrics import capital_metrics
    trades = [
        {"session": "2026-07-06", "entry_ts": "09:40", "exit_ts": "10:10",
         "entry": 7000.0, "risk_pts": 10.0, "pnl_usd": +50.0, "exit_reason": "target"},
        {"session": "2026-07-06", "entry_ts": "10:30", "exit_ts": "10:35",
         "entry": 7010.0, "risk_pts": 8.0, "pnl_usd": -40.0, "exit_reason": "stop"},
        {"session": "2026-07-07", "entry_ts": "09:45", "exit_ts": "11:45",
         "entry": 7020.0, "risk_pts": 12.0, "pnl_usd": +100.0, "exit_reason": "target"},
    ]
    m = capital_metrics(trades, capital=5000.0, n_sessions=4)  # 2 flat days observed
    u, r, t = m["utilization"], m["returns"], m["trading"]
    assert u["leverage_while_in_market"] == round(7010 * 5 / 5000, 2)
    assert u["time_in_market_min_per_day"] == round((30 + 5 + 120) / 4, 1)
    assert r["total_usd"] == 110.0 and r["roc_period_pct"] == 2.2
    assert r["max_drawdown_usd"] == 0.0    # daily equity 10 -> 110 never dips below peak
    assert t["trades_per_session"] == 0.75 and t["participation_pct"] == 50.0
    assert t["max_loss_streak"] == 1 and t["max_win_streak"] == 1
    assert t["risk_budget_usage_median_pct"] == round(100 * (10 * 5 / 75), 0)


# ---------- veto observability ----------

def test_veto_reasons_counted():
    cont = mk5([(100, 101, 99, 100.5), (101, 102, 100.5, 101), (102, 103, 101, 102)])
    bad = Signal("long", stop=101.0, target=99.0, entry_type="market")  # broken geometry
    stats = {}
    trades = run_session(cont, SESSION, fire_at(1, bad), stats=stats)
    assert trades == []
    assert stats["veto"] == {"geometry": 1}
    assert stats["signals"] == 1
