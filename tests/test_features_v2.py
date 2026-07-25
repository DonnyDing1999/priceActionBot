"""Sidecar v2 (features.build_sidecar) — additive Brooks context gated by PA_SIDECAR_V2.

Locks the WP1 contract: (a) flag off -> byte-identical to pre-v2 HEAD; (b) the flag on
only APPENDS the six new keys; (c) the hl_count pullback machine (H1->H2, strong-bar
reset, bear mirror); (d) zone cuts at 1/3 & 2/3; (e) working_range source labels.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.bars import ET  # noqa: E402
from pab.features import build_sidecar  # noqa: E402

SESSION = "2026-07-06"

_V2_KEYS = ("avg_range_10", "working_range", "price_position", "zone",
            "hl_count", "consec_trend_bars")

# The exact HEAD build_sidecar(...) bytes for _GOLDEN_FRAME, captured before the v2
# change (verified equal to `git show HEAD:pab/features.py` output). With PA_SIDECAR_V2=0
# the sidecar must reproduce this string byte-for-byte.
_GOLDEN_FRAME = [(100, 101, 99.5, 100.8), (100.8, 102, 100.5, 101.8),
                 (101.8, 103, 101.5, 102.8), (102.8, 103.2, 101.0, 101.4),
                 (101.4, 102.0, 100.8, 101.8), (101.8, 104, 101.6, 103.8)]
_GOLDEN_V4_JSON = '{"symbol": "ES=F", "session_date": "2026-07-06", "bar_time_et": "2026-07-06 09:55 EDT", "bar_index_from_open": 6, "in_trade_window": true, "bar": {"o": 101.8, "h": 104.0, "l": 101.6, "c": 103.8, "type": "trend_bull", "close_pos_in_range": 0.92, "range_pts": 2.4, "volume": 100}, "ema20": 101.42, "close_vs_ema_pts": 2.38, "ema_slope_pts_3bar": 0.35, "prior_rth_close": null, "open_gap_pts": null, "session_high": 104.0, "session_low": 99.5, "pts_below_session_high": 0.2, "pts_above_session_low": 4.3, "always_in_hint": "ail", "magnets": [], "chop": {"overlap_session": 0.46, "overlap_last5": 0.46, "efficiency": 0.84, "small_body_share": 0.0}, "swings": ["SH@103.2(bar4)", "SL@100.8(bar5)"], "last_bars_pattern": "", "session_table": "i|time|open|high|low|close|type|cp|ema\\n1|09:30|100|101|99.5|100.8|trend_bull|0.87|100.80\\n2|09:35|100.8|102|100.5|101.8|trend_bull|0.87|100.90\\n3|09:40|101.8|103|101.5|102.8|trend_bull|0.87|101.08\\n4|09:45|102.8|103.2|101|101.4|trend_bear|0.18|101.11\\n5|09:50|101.4|102|100.8|101.8|trend_bull|0.83|101.17\\n6|09:55|101.8|104|101.6|103.8|trend_bull|0.92|101.42", "config": {"tick": 0.25, "point_value_usd": 5.0, "per_trade_risk_usd": 75.0, "per_trade_risk_pts": 15.0}}'


def mk5(bars, start="09:30"):
    """5m frame from (o,h,l,c) tuples starting at `start` ET (mirrors test_engine.mk5)."""
    idx = pd.date_range(f"{SESSION} {start}", periods=len(bars), freq="5min", tz=ET)
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 100}
         for o, h, l, c in bars], index=idx)


def mk_multiday(rows):
    """Prior-RTH(3) + overnight(2) + today(2) frame (same layout as test_sidecar_magnets);
    returns (frame, today's last ts)."""
    d1 = pd.date_range("2026-07-02 09:30", periods=3, freq="5min", tz=ET)
    on = pd.date_range("2026-07-02 18:00", periods=2, freq="5min", tz=ET)
    d2 = pd.date_range(f"{SESSION} 09:30", periods=2, freq="5min", tz=ET)
    cont = pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 50}
         for o, h, l, c in rows], index=d1.append(on).append(d2))
    return cont, d2[-1]


def v2(cont, ts, monkeypatch, **kw):
    """build_sidecar with the v2 flag explicitly ON (hermetic against a CI env default)."""
    monkeypatch.setenv("PA_SIDECAR_V2", "1")
    return build_sidecar(cont, ts, **kw)


# ---------- (a) legacy byte-identity + (b) additive-only ----------

def test_flag_off_byte_identical_to_head(monkeypatch):
    monkeypatch.setenv("PA_SIDECAR_V2", "0")
    cont = mk5(_GOLDEN_FRAME)
    sc = build_sidecar(cont, cont.index[-1])
    assert json.dumps(sc) == _GOLDEN_V4_JSON
    assert not any(k in sc for k in _V2_KEYS)


def test_v2_is_purely_additive(monkeypatch):
    cont = mk5(_GOLDEN_FRAME)
    sc = v2(cont, cont.index[-1], monkeypatch)
    # the six keys are appended, in contract order, after the legacy block
    assert [k for k in sc if k in _V2_KEYS] == list(_V2_KEYS)
    base = {k: v for k, v in sc.items() if k not in _V2_KEYS}
    assert json.dumps(base) == _GOLDEN_V4_JSON            # legacy bytes untouched
    # one fully-rendered v2 fragment for this frame
    assert sc["avg_range_10"] == 1.72                     # mean(h-l) of the 6 bars
    assert sc["working_range"] == {"hi": 104.0, "lo": 99.5,
                                   "src_hi": "session", "src_lo": "session"}
    assert sc["price_position"] == 0.96 and sc["zone"] == "upper_third"
    assert sc["hl_count"] == {"dir": "bull", "count": 1,
                              "last_label": "H1", "reset_bar": None}
    assert sc["consec_trend_bars"] == {"dir": "bull", "n": 2}


def test_v2_default_on_when_unset(monkeypatch):
    monkeypatch.delenv("PA_SIDECAR_V2", raising=False)   # default "1"
    cont = mk5(_GOLDEN_FRAME)
    sc = build_sidecar(cont, cont.index[-1])
    assert all(k in sc for k in _V2_KEYS)


# ---------- (c) hl_count pullback machine ----------

def test_hl_count_h1_then_h2(monkeypatch):
    # two separate bull pullbacks, each resuming to a higher high -> H1 then H2
    cont = mk5([(100, 101, 99.5, 100.8), (100.8, 102, 100.5, 101.8),
                (101.8, 103, 101.5, 102.8), (102.8, 103.2, 101.4, 101.6),
                (101.6, 104, 101.4, 103.8), (103.8, 104.2, 101.0, 101.5),
                (101.5, 105, 101.3, 104.5)])
    hl = v2(cont, cont.index[-1], monkeypatch)["hl_count"]
    assert hl == {"dir": "bull", "count": 2, "last_label": "H2", "reset_bar": None}


def test_hl_count_reset_on_strong_bear(monkeypatch):
    # H1 forms at bar 5, then bar 6 is a bear bar whose body clears 1.2*avg_range_10
    # -> count resets to 0 and reset_bar records bar 6
    cont = mk5([(100, 101, 99.5, 100.8), (100.8, 102, 100.5, 101.8),
                (101.8, 103, 101.5, 102.8), (102.8, 103.2, 101.4, 101.6),
                (101.6, 104, 101.4, 103.8), (103.8, 104, 100.0, 100.5),
                (101.0, 105, 100.8, 104.5)])
    sc = v2(cont, cont.index[-1], monkeypatch)
    assert sc["hl_count"] == {"dir": "bull", "count": 0,
                              "last_label": "", "reset_bar": 6}
    assert abs(100.5 - 103.8) >= 1.2 * sc["avg_range_10"]   # bar 6 body clears threshold


def test_hl_count_bear_mirror(monkeypatch):
    # mirror of the bull case: pullbacks are up-bars (higher high), resumptions undercut
    # the prior low -> L1 then L2
    cont = mk5([(100.0, 100.0, 99.5, 99.5), (99.5, 99.5, 99.0, 99.0),
                (99.0, 99.0, 98.5, 98.5), (98.5, 99.3, 98.5, 99.2),
                (99.2, 99.2, 98.0, 98.1), (98.1, 99.4, 98.0, 98.9),
                (99.3, 99.3, 97.5, 97.6)])
    hl = v2(cont, cont.index[-1], monkeypatch)["hl_count"]
    assert hl == {"dir": "bear", "count": 2, "last_label": "L2", "reset_bar": None}


# ---------- (d) zone cuts at exactly 1/3 and 2/3 ----------

def test_zone_boundaries_exact_thirds(monkeypatch):
    # single-day frame -> no magnets -> working_range == session extremes (100..103)
    low = mk5([(100, 103, 100, 102), (102, 102.5, 101, 101)])   # (101-100)/3 == 1/3
    sc = v2(low, low.index[-1], monkeypatch)
    assert sc["working_range"] == {"hi": 103.0, "lo": 100.0,
                                   "src_hi": "session", "src_lo": "session"}
    assert sc["price_position"] == 0.33 and sc["zone"] == "middle"      # 1/3 -> middle
    high = mk5([(100, 103, 100, 102), (102, 102.5, 101, 102)])  # (102-100)/3 == 2/3
    sc = v2(high, high.index[-1], monkeypatch)
    assert sc["price_position"] == 0.67 and sc["zone"] == "upper_third"  # 2/3 -> upper


# ---------- (e) working_range source labels ----------

def test_working_range_source_labels(monkeypatch):
    # today gaps above the magnets: session supplies the high, yesterday's low the floor
    above, ts = mk_multiday([(100, 105, 99, 104), (104, 106, 103, 105),
                             (105, 107, 104, 106), (106, 108, 105.5, 107),
                             (107, 109, 106, 108.5), (110, 111, 109, 110.5),
                             (110.5, 112, 110, 111)])
    assert v2(above, ts, monkeypatch)["working_range"] == {
        "hi": 112.0, "lo": 99.0, "src_hi": "session", "src_lo": "yday_low"}
    # today trades INSIDE yesterday's range: magnets supply both bounds
    inside, ts = mk_multiday([(110, 120, 109, 115), (115, 118, 110, 112),
                              (112, 116, 108, 110), (110, 113, 90, 95),
                              (95, 100, 88, 92), (100, 102, 99, 101),
                              (101, 103, 100, 102)])
    assert v2(inside, ts, monkeypatch)["working_range"] == {
        "hi": 120.0, "lo": 88.0, "src_hi": "yday_high", "src_lo": "overnight_low"}


# ---------- consec_trend_bars + avg_range_10 window ----------

def test_consec_trend_bars_run_and_doji_break(monkeypatch):
    # doji seed then three bull trend bars -> run length 3 (the doji breaks the walk-back)
    up = mk5([(100, 101, 99.5, 100.0), (100, 101, 99.8, 100.8),
              (100.8, 102, 100.5, 101.8), (101.8, 103, 101.5, 102.8)])
    assert v2(up, up.index[-1], monkeypatch)["consec_trend_bars"] == \
        {"dir": "bull", "n": 3}
    # a doji AT the current bar yields no run at all
    doji_now = mk5([(100, 101, 99.5, 100.8), (100.8, 102, 100.5, 101.8),
                    (101.8, 102.5, 101.0, 101.75)])            # |body| 0.05 < 0.1*range
    assert v2(doji_now, doji_now.index[-1], monkeypatch)["consec_trend_bars"] == \
        {"dir": "none", "n": 0}


def test_avg_range_10_uses_last_ten_bars(monkeypatch):
    # first two bars have huge ranges that the 10-bar cap must exclude
    bars = [(100, 130, 70, 100), (100, 130, 70, 100)] + [(100, 101, 99, 100.0)] * 10
    cont = mk5(bars)
    assert v2(cont, cont.index[-1], monkeypatch)["avg_range_10"] == 2.0
