"""RiskManager v2 structural-veto tests (WP2).

Pure unit tests on crafted ctx dicts (no engine needed) for every new veto tag, both
firing and not-firing; a ctx=None ⇒ v1-identical parametrization; a post-stop cooldown
sequence; and one engine-level integration proving a middle-zone signal is vetoed.

Each crafted trade passes the v1 gate (geometry / $ cap / R:R) on its own, so the tag
asserted is always the v2 veto — never a v1 rejection leaking through.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.backtest import Config, Signal, run_session  # noqa: E402
from pab.bars import ET  # noqa: E402
from pab.risk import RiskManager  # noqa: E402

SESSION = "2026-07-06"

# a valid long / short whose v1 geometry-$-R:R all pass (risk 5pt, RR 2.0, cost 25<=75)
LONG = ("long", 100.0, 95.0, 110.0)
SHORT = ("short", 100.0, 105.0, 90.0)


def base_ctx(**kw):
    """A neutral ctx (every v2 veto inert by default); override keys per test."""
    ctx = {
        "regime": "open", "always_in": "neutral", "zone": "middle",
        "price_position": 0.5, "working_range": {"hi": 110, "lo": 90},
        "avg_range_10": 2.0, "hl_count": {"dir": "none", "count": 0, "last_label": ""},
        "consec_trend_bars": {"dir": "none", "n": 0},
        "signal_bar_range": 2.0, "bar_index": 10, "entry_type": "market",
    }
    ctx.update(kw)
    return ctx


def mk5(bars):
    idx = pd.date_range(f"{SESSION} 09:30", periods=len(bars), freq="5min", tz=ET)
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 100}
         for o, h, l, c in bars], index=idx)


def fire_at(bar_index, signal):
    def strat(sidecar, bars):
        return signal if sidecar["bar_index_from_open"] == bar_index else None
    return strat


# ---------- zone veto (range regime only) ----------

def test_zone_middle_vetoes_any_side():
    rm = RiskManager(Config())
    ctx = base_ctx(regime="range", zone="middle")
    assert rm.validate_entry(*LONG, ctx=ctx).reason == "zone_middle"
    assert rm.validate_entry(*SHORT, ctx=ctx).reason == "zone_middle"


def test_zone_upper_blocks_long_allows_short_fade():
    rm = RiskManager(Config())
    ctx = base_ctx(regime="range", zone="upper_third")
    assert rm.validate_entry(*LONG, ctx=ctx).reason == "zone_long_upper"
    assert rm.validate_entry(*SHORT, ctx=ctx).ok          # fade toward the upper extreme


def test_zone_lower_blocks_short_allows_long_fade():
    rm = RiskManager(Config())
    ctx = base_ctx(regime="range", zone="lower_third")
    assert rm.validate_entry(*SHORT, ctx=ctx).reason == "zone_short_lower"
    assert rm.validate_entry(*LONG, ctx=ctx).ok           # fade toward the lower extreme


def test_zone_veto_inert_off_or_non_range():
    # knob off -> no zone veto even in the middle of a range
    off = RiskManager(Config(zone_veto=False))
    assert off.validate_entry(*LONG, ctx=base_ctx(regime="range", zone="middle")).ok
    # non-range regime -> zone veto never applies (open here keeps trend gates inert too)
    rm = RiskManager(Config())
    assert rm.validate_entry(*LONG, ctx=base_ctx(regime="open", zone="middle")).ok


# ---------- pullback gate + counter-trend (trend regime only) ----------

def test_trend_needs_second_entry_fires_on_first_leg():
    rm = RiskManager(Config())
    ctx = base_ctx(regime="trend", always_in="ail",
                   hl_count={"dir": "bull", "count": 1},
                   consec_trend_bars={"dir": "bull", "n": 1})
    assert rm.validate_entry(*LONG, ctx=ctx).reason == "trend_needs_second_entry"


def test_trend_second_entry_satisfied_passes():
    rm = RiskManager(Config())
    ctx = base_ctx(regime="trend", always_in="ail",
                   hl_count={"dir": "bull", "count": 2},
                   consec_trend_bars={"dir": "bull", "n": 1})
    assert rm.validate_entry(*LONG, ctx=ctx).ok


def test_trend_spike_exception_allows_first_entry():
    # >=3-bar spike lets an H1 through even without a second pullback leg
    rm = RiskManager(Config())
    ctx = base_ctx(regime="trend", always_in="ail",
                   hl_count={"dir": "bull", "count": 1},
                   consec_trend_bars={"dir": "bull", "n": 3})
    assert rm.validate_entry(*LONG, ctx=ctx).ok


def test_trend_gate_wrong_direction_hl_still_blocks():
    # a bull hl_count does NOT satisfy a short with-trend proposal (direction must match)
    rm = RiskManager(Config())
    ctx = base_ctx(regime="trend", always_in="ais",
                   hl_count={"dir": "bull", "count": 3},
                   consec_trend_bars={"dir": "bull", "n": 1})
    assert rm.validate_entry(*SHORT, ctx=ctx).reason == "trend_needs_second_entry"


def test_counter_trend_blocks_against_always_in():
    rm = RiskManager(Config())
    # market is always-in-short: a long is counter-trend
    assert rm.validate_entry(*LONG, ctx=base_ctx(regime="trend",
                             always_in="ais")).reason == "counter_trend"
    # mirror: always-in-long, a short is counter-trend
    assert rm.validate_entry(*SHORT, ctx=base_ctx(regime="trend",
                             always_in="ail")).reason == "counter_trend"


def test_pullback_gate_off_passes_both():
    off = RiskManager(Config(pullback_gate=False))
    # would-be trend_needs_second_entry
    assert off.validate_entry(*LONG, ctx=base_ctx(regime="trend", always_in="ail",
                              hl_count={"dir": "bull", "count": 0})).ok
    # would-be counter_trend
    assert off.validate_entry(*LONG, ctx=base_ctx(regime="trend", always_in="ais")).ok


# ---------- climax no-chase (all regimes, stop entries only) ----------

def test_climax_no_chase_fires_on_stop_beyond_climax():
    rm = RiskManager(Config())
    ctx = base_ctx(regime="open", entry_type="stop",
                   signal_bar_range=5.0, avg_range_10=2.0)   # 5 >= 2*2
    assert rm.validate_entry(*LONG, ctx=ctx).reason == "climax_no_chase"


def test_climax_inert_for_non_stop_entries():
    rm = RiskManager(Config())
    for et in ("market", "limit"):
        ctx = base_ctx(regime="open", entry_type=et,
                       signal_bar_range=5.0, avg_range_10=2.0)
        assert rm.validate_entry(*LONG, ctx=ctx).ok


def test_climax_below_threshold_and_missing_avg():
    rm = RiskManager(Config())
    # just under 2x -> allowed
    assert rm.validate_entry(*LONG, ctx=base_ctx(regime="open", entry_type="stop",
                             signal_bar_range=3.9, avg_range_10=2.0)).ok
    # avg_range_10 zero / missing -> skip (avoid div-by-zero-style noise)
    assert rm.validate_entry(*LONG, ctx=base_ctx(regime="open", entry_type="stop",
                             signal_bar_range=5.0, avg_range_10=0.0)).ok
    assert rm.validate_entry(*LONG, ctx=base_ctx(regime="open", entry_type="stop",
                             signal_bar_range=5.0, avg_range_10=None)).ok


def test_climax_knob_off():
    off = RiskManager(Config(no_chase=False))
    assert off.validate_entry(*LONG, ctx=base_ctx(regime="open", entry_type="stop",
                              signal_bar_range=5.0, avg_range_10=2.0)).ok


# ---------- cooldown across a two-trade sequence ----------

def test_cooldown_after_stop_out_sequence():
    rm = RiskManager(Config())            # cooldown_bars=3, cooldown_ticks=3, tick=0.25
    # a long stops out on bar 5 at entry 100.0
    rm.on_trade_closed(-40.0, side="long", entry=100.0, exit_bar_index=5,
                       exit_reason="stop")
    oc = lambda b: base_ctx(regime="open", bar_index=b)  # noqa: E731

    # SAME-side long is blocked for 2 bars regardless of price
    assert rm.validate_entry(*LONG, ctx=oc(6)).reason == "cooldown_same"
    assert rm.validate_entry("long", 200.0, 195.0, 210.0,
                             ctx=oc(7)).reason == "cooldown_same"   # far away, still blocked
    assert rm.validate_entry(*LONG, ctx=oc(8)).ok                  # bars_since 3 -> clear

    # OPPOSITE-side short near the stopped entry (|100.5-100|=0.5 <= 3*0.25) -> blocked
    assert rm.validate_entry("short", 100.5, 105.0, 90.0,
                             ctx=oc(6)).reason == "cooldown_opposite"
    # OPPOSITE-side short far from the stopped entry -> allowed
    assert rm.validate_entry("short", 105.0, 110.0, 95.0, ctx=oc(6)).ok
    # OPPOSITE-side beyond the cooldown_bars window -> allowed even if near
    assert rm.validate_entry("short", 100.5, 105.0, 90.0, ctx=oc(9)).ok  # bars_since 4 > 3


def test_cooldown_only_after_a_stop_exit():
    rm = RiskManager(Config())
    rm.on_trade_closed(+50.0, side="long", entry=100.0, exit_bar_index=5,
                       exit_reason="target")               # not a stop -> no memory
    assert rm.validate_entry(*LONG, ctx=base_ctx(regime="open", bar_index=6)).ok


def test_cooldown_knob_off():
    off = RiskManager(Config(cooldown=False))
    off.on_trade_closed(-40.0, side="long", entry=100.0, exit_bar_index=5,
                        exit_reason="stop")
    assert off.validate_entry(*LONG, ctx=base_ctx(regime="open", bar_index=6)).ok


def test_on_trade_closed_v1_kwargs_optional():
    # positional-only call (v1 callers) still updates the circuit breakers
    rm = RiskManager(Config())
    rm.on_trade_closed(-10.0)
    rm.on_trade_closed(-10.0)
    assert rm.session_halted() == (True, "consec_loss")
    assert rm._last_stop is None                           # no stop info without kwargs


# ---------- ctx=None ⇒ exactly v1 behavior ----------

V1_CASES = [
    ("long", 100.0, 101.0, 105.0, None, "geometry"),
    ("short", 100.0, 99.0, 95.0, None, "geometry"),
    ("long", 100.0, 98.0, 101.0, None, "rr_below_floor"),
    ("long", 100.0, 84.0, 120.0, None, "risk_gt_cap"),
    ("long", 100.0, 99.5, 101.0, None, "risk_too_small"),
    ("nope", 100.0, 95.0, 110.0, None, "bad_side"),
    ("long", 100.0, None, 110.0, None, "missing_stop_or_target"),
    ("long", 100.0, 95.0, 110.0, [103.0], "rr_to_magnet"),
    ("long", 100.0, 95.0, 106.0, None, None),              # None expected == ok
    ("long", 100.0, 98.0, 102.5, None, None),              # 2.0pt = at the floor
    ("long", 100.0, 95.0, 105.0, [105.0], None),           # magnet AT target -> ok
]


@pytest.mark.parametrize("side,entry,stop,target,obs,expected", V1_CASES)
def test_ctx_none_is_v1_identical(side, entry, stop, target, obs, expected):
    # ctx=None must reproduce the v1 result, and be identical to omitting ctx entirely
    a = RiskManager(Config()).validate_entry(side, entry, stop, target,
                                             obstacles=obs, ctx=None)
    b = RiskManager(Config()).validate_entry(side, entry, stop, target, obstacles=obs)
    assert (a.ok, a.reason) == (b.ok, b.reason)
    if expected is None:
        assert a.ok
    else:
        assert not a.ok and a.reason == expected


def test_v1_tag_wins_tie_over_v2():
    # a broken-geometry long carrying a would-be zone_middle ctx still reports geometry
    rm = RiskManager(Config())
    ctx = base_ctx(regime="range", zone="middle")
    assert rm.validate_entry("long", 100.0, 101.0, 105.0, ctx=ctx).reason == "geometry"
    # and the SAME ctx vetoes a v1-valid trade -> proves ctx is actually consulted
    assert rm.validate_entry("long", 100.0, 95.0, 110.0, ctx=ctx).reason == "zone_middle"


# ---------- engine-level integration: middle-zone signal vetoed ----------

def test_engine_range_middle_zone_veto():
    # a flat, two-sided chop keeps always_in neutral (=> classify_regime "range") and the
    # close pinned mid-range (=> zone "middle"). A market long fired on bar 7 (>6, so not
    # "open") passes v1 but is vetoed zone_middle. Bar 7 is not the last bar, so it is
    # actually evaluated (the entry loop needs a next bar to fill on).
    flat = (100, 102, 98, 100)
    cont = mk5([flat] * 8)                                 # bars 1..8, all balanced dojis
    sig = Signal("long", stop=95.0, target=110.0, entry_type="market")
    stats: dict = {}
    trades = run_session(cont, SESSION, fire_at(7, sig), Config(), stats=stats)
    assert trades == []
    assert stats["veto"]["zone_middle"] == 1
    assert stats["signals"] == 1
