"""WP5 — terminal taxonomy, veto_replay ctx reconstruction + report math, calibration.

No LLM, no network. Small fakes and synthetic frames only. The veto_replay tier-2
reconstruction is drift-guarded against features.build_sidecar (build a frame, take its
session_table, feed it back, assert the recomputed v2 keys match).
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pab.backtest import Config, Signal, run_session  # noqa: E402
from pab.bars import ET  # noqa: E402
from pab.features import build_sidecar  # noqa: E402
from pab.journal import Journal  # noqa: E402
from pab.llm_strategy import LLMStrategy, terminal_for  # noqa: E402


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "scripts" / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


veto_replay = _load("veto_replay")
calibration = _load("calibration")

SESSION = "2026-07-06"


def mk5(bars, session=SESSION):
    idx = pd.date_range(f"{session} 09:30", periods=len(bars), freq="5min", tz=ET)
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 100}
         for o, h, l, c in bars], index=idx)


def fire_at(bar_index, signal):
    def strat(sidecar, bars):
        return signal if sidecar["bar_index_from_open"] == bar_index else None
    return strat


# ============================================================ A. terminal taxonomy

def test_terminal_for_all_stages():
    assert terminal_for({"action": "error", "reason": "X"}) == \
        {"stage": "llm", "node": "error", "label": "error: X"}
    assert terminal_for({"action": "no_trade", "terminal_reason": "echo_mismatch",
                         "reason": "validator: echo_mismatch"})["stage"] == "validator"
    assert terminal_for({"action": "no_trade", "reason": "gated: too_late_in_window"}) == \
        {"stage": "code_gate", "node": "too_late_in_window",
         "label": "code gate: too_late_in_window"}
    assert terminal_for({"action": "no_trade", "reason": "day_gate: C (overlap)"})["stage"] \
        == "day_gate"
    prop = terminal_for({"action": "long", "setup": "H2", "entry_type": "stop",
                         "reason": "bar 5"})
    assert prop["stage"] == "llm" and prop["node"] == "proposed"
    assert terminal_for({"action": "no_trade", "reason": "no clean setup"}) == \
        {"stage": "llm", "node": "no_setup", "label": "no clean setup"}


def test_terminal_label_capped_at_60():
    t = terminal_for({"action": "error", "reason": "e" * 200})
    assert len(t["label"]) <= 60


def _sidecar_stub(bar_index=8):
    return {"bar_index_from_open": bar_index, "bar_time_et": f"{SESSION} 10:05 EDT",
            "bar": {"type": "doji", "c": 100.0}, "session_high": 101.0,
            "session_low": 99.0, "close_vs_ema_pts": 0.0}


def test_record_attaches_terminal_gate_llm_validator_error(monkeypatch):
    strat = LLMStrategy(pd.DataFrame(), gate=True)
    # code-gate path: bar past the last-signal bar -> obvious_no_trade fires
    strat(_sidecar_stub(bar_index=23), pd.DataFrame())
    assert strat.decisions[-1]["decision"]["terminal"]["stage"] == "code_gate"

    # llm proposed path
    monkeypatch.setattr("pab.llm_strategy.decide",
                        lambda *a, **k: {"action": "long", "setup": "H2", "stop": 99.0,
                                         "target": 103.0, "entry_type": "stop",
                                         "reason": "bar 5 pullback"})
    strat2 = LLMStrategy(pd.DataFrame(), gate=False)
    strat2(_sidecar_stub(), pd.DataFrame())
    assert strat2.decisions[-1]["decision"]["terminal"] == \
        {"stage": "llm", "node": "proposed", "label": "long H2 stop"}

    # validator-downgrade path (decide returns terminal_reason)
    monkeypatch.setattr("pab.llm_strategy.decide",
                        lambda *a, **k: {"action": "no_trade", "terminal_reason":
                                         "echo_mismatch", "reason": "validator: echo_mismatch"})
    strat3 = LLMStrategy(pd.DataFrame(), gate=False)
    strat3(_sidecar_stub(), pd.DataFrame())
    assert strat3.decisions[-1]["decision"]["terminal"]["stage"] == "validator"

    # error path (decide raises)
    def boom(*a, **k):
        raise RuntimeError("quota")
    monkeypatch.setattr("pab.llm_strategy.decide", boom)
    strat4 = LLMStrategy(pd.DataFrame(), gate=False)
    strat4(_sidecar_stub(), pd.DataFrame())
    assert strat4.decisions[-1]["decision"]["terminal"] == \
        {"stage": "llm", "node": "error", "label": "error: RuntimeError: quota"}


def test_journal_preserves_and_overrides_terminal():
    j = Journal()
    # decision carries its own terminal -> preserved
    j.record_decision("s", 5, "09:55",
                      {"action": "long", "reason": "bar 5",
                       "terminal": {"stage": "llm", "node": "proposed", "label": "x"}})
    assert j.sessions["s"]["decisions"][0]["decision"]["terminal"]["node"] == "proposed"
    # explicit override wins (runner stamping the engine outcome)
    j.record_decision("s", 6, "10:00",
                      {"action": "long", "reason": "bar 6",
                       "terminal": {"stage": "llm", "node": "proposed", "label": "x"}},
                      terminal={"stage": "engine", "node": "filled", "label": "filled"})
    assert j.sessions["s"]["decisions"][1]["decision"]["terminal"] == \
        {"stage": "engine", "node": "filled", "label": "filled"}
    # no terminal anywhere -> key absent (legacy-shape, additive)
    j.record_decision("s", 7, "10:05", {"action": "no_trade", "reason": "x"})
    assert "terminal" not in j.sessions["s"]["decisions"][2]["decision"]


def test_engine_outcomes_risk_and_engine_stages():
    # a bad-geometry signal -> stats["outcomes"] carries a risk-stage veto tag
    cont = mk5([(100, 101, 99, 100.5), (101, 102, 100.5, 101), (105, 111, 104, 110)])
    st = {}
    run_session(cont, SESSION, fire_at(1, Signal("long", stop=101.0, target=105.0,
                                                 entry_type="market")), stats=st)
    assert {"time": "09:30", "stage": "risk", "node": "geometry"} in st["outcomes"]

    # a filling signal -> engine-stage "filled"
    st2 = {}
    run_session(cont, SESSION, fire_at(1, Signal("long", stop=95.5, target=110.5,
                                                 entry_type="market")), stats=st2)
    assert any(o["stage"] == "engine" and o["node"] == "filled" for o in st2["outcomes"])


# ==================================================== B. veto_replay reconstruction

def test_v2_from_rows_matches_build_sidecar(monkeypatch):
    """Drift guard: recomputing v2 keys from a session_table must match the values
    features.build_sidecar produced for the same single-day frame (H1->H2 case)."""
    monkeypatch.setenv("PA_SIDECAR_V2", "1")
    cont = mk5([(100, 101, 99.5, 100.8), (100.8, 102, 100.5, 101.8),
                (101.8, 103, 101.5, 102.8), (102.8, 103.2, 101.4, 101.6),
                (101.6, 104, 101.4, 103.8), (103.8, 104.2, 101.0, 101.5),
                (101.5, 105, 101.3, 104.5)])
    sc = build_sidecar(cont, cont.index[-1])
    assert sc["hl_count"] == {"dir": "bull", "count": 2, "last_label": "H2",
                              "reset_bar": None}       # sanity: the frame is the H1->H2 case
    rows = veto_replay.parse_session_table(sc["session_table"])
    v2 = veto_replay.v2_from_rows(rows)
    assert v2["hl_count"] == sc["hl_count"]
    assert v2["consec_trend_bars"] == sc["consec_trend_bars"]
    assert v2["zone"] == sc["zone"]
    assert v2["avg_range_10"] == sc["avg_range_10"]
    assert v2["working_range"] == sc["working_range"]
    assert v2["signal_bar_range"] == round(sc["bar"]["h"] - sc["bar"]["l"], 2)


def test_parse_session_table_and_zone_from_text():
    text = ("i|time|open|high|low|close|type|cp|ema\n"
            "1|09:30|100|103|100|102|trend_bull|0.67|100.50\n"
            "2|09:35|102|102.5|101|101|trend_bear|0.00|100.80")
    rows = veto_replay.parse_session_table(text)
    assert len(rows) == 2 and rows[0]["h"] == 103.0 and rows[1]["type"] == "trend_bear"
    v2 = veto_replay.v2_from_rows(rows)
    # session hi/lo = 103/100, last close 101 -> pos (101-100)/3 = 0.33 -> middle
    assert v2["zone"] == "middle" and v2["price_position"] == 0.33


def test_reconstruct_ctx_tier3_marks_pullback_nochase_unavailable():
    ctx, unavail = veto_replay.reconstruct_ctx(
        {"action": "long", "reason": "stop entry bar 6"},
        {"close": 100.0, "session_high": 101.0, "session_low": 99.0,
         "always_in": "neutral", "close_vs_ema_pts": 0.0},
        None, bar_index=8, entry_type="stop")
    assert ctx["zone"] is not None                     # zone IS reconstructable from _brief
    assert ctx["hl_count"] is None and ctx["avg_range_10"] is None
    assert veto_replay.UNAVAIL_PULLBACK <= unavail and veto_replay.UNAVAIL_NOCHASE <= unavail


# ---- end-to-end report math over a crafted fixture journal directory --------------

def _full_sidecar(zone, bar_index=10, always_in="neutral"):
    """Minimal full v2 sidecar (tier-1) that forces classify_regime -> range."""
    return {"zone": zone, "always_in_hint": always_in, "price_position": 0.9,
            "working_range": {"hi": 5010.0, "lo": 5000.0, "src_hi": "session",
                              "src_lo": "session"},
            "avg_range_10": 3.0, "hl_count": {"dir": "bull", "count": 0},
            "consec_trend_bars": {"dir": "none", "n": 0},
            "bar": {"h": 5003.0, "l": 5001.0}, "bar_index_from_open": bar_index,
            "ema_slope_pts_3bar": 0.0, "close_vs_ema_pts": 0.0}


def _journal(session, side, stop, target, risk_pts, pnl, exit_reason, zone):
    """One-session journal: a single trade proposal (with an embedded full sidecar) that
    is a realized trade."""
    dec = {"action": side, "setup": "x", "stop": stop, "target": target,
           "reason": f"bar 10 {side} setup enough characters to pass", "entry_type": "stop"}
    return {
        "meta": {"session": session, "n_decisions": 1, "n_trades": 1},
        "decisions": [{"bar": 10, "time": "10:15", "decision": dec,
                       "context": None, "sidecar": _full_sidecar(zone)}],
        "trades": [{"session": session, "side": side, "entry_ts": "10:20",
                    "exit_ts": "10:30", "entry": stop + (risk_pts if side == "long"
                                                         else -risk_pts),
                    "stop": stop, "target": target, "risk_pts": risk_pts,
                    "pnl_usd": pnl, "exit_reason": exit_reason}]}


def test_veto_replay_report_math(tmp_path):
    jdir = tmp_path
    # A: long in upper_third range -> zone_long_upper veto; realized LOSER -30
    (jdir / "2024-01-02.json").write_text(json.dumps(
        _journal("2024-01-02", "long", 5000.0, 5010.0, 2.0, -30.0, "stop",
                 "upper_third")))
    # B: short in upper_third range -> fade, PASSES; realized WINNER +50
    (jdir / "2024-01-08.json").write_text(json.dumps(
        _journal("2024-01-08", "short", 5010.0, 5000.0, 2.0, 50.0, "target",
                 "upper_third")))

    files = sorted(jdir.glob("2024-*.json"))
    rep = veto_replay.analyze_cohort("oos_2024_25", files, knob_off=[])
    s = veto_replay.cohort_summary(rep)
    assert s["realized"] == 2 and s["proposals"] == 2
    assert s["blocked"] == 1 and s["losers"] == 1 and s["winners"] == 0
    assert s["losers_sum"] == -30.0
    assert s["pf_before"] == round(50 / 30, 2)          # 1.67
    assert s["pf_after"] is None                        # only the winner remains
    assert s["total_before"] == 20.0 and s["total_after"] == 50.0
    assert s["by_tag"]["zone_long_upper"]["loss"] == 1

    # ablation: turning the zone knob off unblocks A
    rep2 = veto_replay.analyze_cohort("oos_2024_25", files, knob_off=["zone_veto"])
    assert veto_replay.cohort_summary(rep2)["blocked"] == 0


def test_veto_replay_cooldown_across_two_trades(tmp_path):
    """A stop-out then a same-side re-entry 1 bar later -> cooldown_same blocks the 2nd."""
    jdir = tmp_path
    sc1 = _full_sidecar("lower_third", bar_index=6)     # regime range but zone won't veto long
    sc2 = _full_sidecar("lower_third", bar_index=8)
    journal = {
        "meta": {"session": "2024-02-01"},
        "decisions": [
            {"bar": 6, "time": "09:55", "decision":
                {"action": "long", "setup": "x", "stop": 5000.0, "target": 5010.0,
                 "reason": "bar 6 long setup long enough", "entry_type": "stop"},
             "sidecar": sc1},
            {"bar": 8, "time": "10:05", "decision":
                {"action": "long", "setup": "x", "stop": 5001.0, "target": 5011.0,
                 "reason": "bar 8 long re-entry long enough", "entry_type": "stop"},
             "sidecar": sc2}],
        "trades": [
            {"session": "2024-02-01", "side": "long", "entry_ts": "09:55",
             "exit_ts": "10:00", "entry": 5002.0, "stop": 5000.0, "target": 5010.0,
             "risk_pts": 2.0, "pnl_usd": -30.0, "exit_reason": "stop"}]}
    (jdir / "2024-02-01.json").write_text(json.dumps(journal))
    rep = veto_replay.analyze_cohort("oos_2024_25",
                                     [jdir / "2024-02-01.json"], knob_off=[])
    # the 2nd proposal (bar 8) is a same-side re-entry within 2 bars of the bar-6 stop-out
    tags = {r.bar: r.tag for r in rep.results}
    assert tags[8] == "cooldown_same"


def test_oos_sessions_from_manifest(tmp_path):
    # manifest points at an (absolute) sessions_file; 2026 dates are filtered out of OOS
    (tmp_path / "oos_manifest.json").write_text(json.dumps(
        {"sessions_file": str(tmp_path / "s.txt")}))
    (tmp_path / "s.txt").write_text("2024-01-02\n2025-06-01\n2026-01-01\n")
    got = veto_replay.oos_sessions(tmp_path)
    assert got == {"2024-01-02", "2025-06-01"}


# ============================================================== C. calibration Brier

def test_calibration_brier_on_synthetic():
    journal = {
        "meta": {"session": "2099-01-01"},
        "decisions": [
            {"bar": 5, "time": "10:10", "context": {"bar_type": "doji"},
             "decision": {"action": "no_trade", "reason": "x",
                          "next_bar": {"p_up": 70, "p_down": 20, "p_neutral": 10}}},
            {"bar": 6, "time": "10:15", "context": {"bar_type": "trend_bull"},
             "decision": {"action": "no_trade", "reason": "x"}},   # realized: up
            {"bar": 7, "time": "10:20", "context": {"bar_type": "doji"},
             "decision": {"action": "no_trade", "reason": "x",
                          "next_bar": {"p_up": 10, "p_down": 80, "p_neutral": 10}}},
            {"bar": 8, "time": "10:25", "context": {"bar_type": "trend_bear"},
             "decision": {"action": "no_trade", "reason": "x"}}]}   # realized: down
    rep = calibration.CalReport("t", calibration.score_journal(journal))
    m = rep.metrics()
    assert m["n"] == 2
    assert m["brier"] == pytest.approx(0.10, abs=1e-9)
    assert m["base_rate_brier"] == pytest.approx(0.50, abs=1e-9)
    assert m["skill"] == pytest.approx(0.80, abs=1e-9)


def test_next_bar_probs_validity():
    assert calibration.next_bar_probs(
        {"next_bar": {"p_up": 50, "p_down": 30, "p_neutral": 20}}) == [0.5, 0.3, 0.2]
    assert calibration.next_bar_probs(
        {"next_bar": {"p_up": 50, "p_down": 30, "p_neutral": 5}}) is None   # sum 85
    assert calibration.next_bar_probs(
        {"next_bar": {"p_up": 50.5, "p_down": 30, "p_neutral": 20}}) is None  # non-int
    assert calibration.next_bar_probs({"action": "no_trade"}) is None


def test_calibration_score_pairs_by_following_bar():
    j = {"meta": {"session": "x"}, "decisions": [
        {"bar": 3, "context": {"bar_type": "x"},
         "decision": {"next_bar": {"p_up": 40, "p_down": 40, "p_neutral": 20}}},
        {"bar": 4, "context": {"bar_type": "trend_bear"}, "decision": {"action": "no_trade"}}]}
    s = calibration.score_journal(j)
    assert len(s) == 1 and s[0].realized == "down" and s[0].bar == 3
