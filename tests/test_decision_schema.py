"""WP4 — Decision JSON v2 validation (pure) + retry-direction guard + decide() wiring.

No LLM, no network: validate_decision is pure; the decide() tests monkeypatch the
provider. Covers the accept/downgrade matrix, the bar-citation regex, the next_bar
probability sum, echo/override, v1-sidecar graceful skip, and the retry_flip guard.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pab.decision_schema import validate_decision  # noqa: E402


# ---------- fixtures ----------

def sc_v2(bar=6, ai="ail", zone="lower_third"):
    """A sidecar-v2 stub (carries `zone`)."""
    return {"bar_index_from_open": bar, "always_in_hint": ai, "zone": zone,
            "session_date": "2026-07-06"}


def sc_v1(bar=6, ai="ail"):
    """A sidecar-v1 stub (no `zone` key)."""
    return {"bar_index_from_open": bar, "always_in_hint": ai,
            "session_date": "2026-07-06"}


def good_trade(**over):
    d = {"action": "long", "setup": "H2 pullback", "entry_type": "stop",
         "stop": 7000.0, "target": 7020.0, "confidence": 60,
         "reason": ("H2 pullback to the rising EMA at bar 5 held; trend bar 6 closed on "
                    "its high with room to the measured-move target"),
         "echo": {"regime": "trend", "always_in": "ail", "zone": "lower_third"},
         "override": None,
         "next_bar": {"p_up": 60, "p_down": 30, "p_neutral": 10},
         "_usage": {"provider": "zhipu", "model": "glm-4.5-flash"}}
    d.update(over)
    return d


# ---------- accept ----------

def test_valid_trade_passes_untouched():
    d, term = validate_decision(good_trade(), sc_v2())
    assert term is None
    assert d["action"] == "long"
    assert d["next_bar"] == {"p_up": 60, "p_down": 30, "p_neutral": 10}  # kept intact
    assert "terminal_reason" not in d


def test_no_trade_always_passes_untouched():
    nt = {"action": "no_trade", "setup": "", "entry_type": "market",
          "stop": None, "target": None, "reason": "two-sided; no edge",
          "next_bar": {"p_up": 1, "p_down": 1, "p_neutral": 1}}  # bad sum, but no_trade
    d, term = validate_decision(dict(nt), sc_v2())
    assert term is None and d == nt          # literally untouched


def test_limit_with_entry_px_passes():
    d, term = validate_decision(
        good_trade(entry_type="limit", entry_px=6998.0), sc_v2())
    assert term is None and d["action"] == "long" and d["entry_px"] == 6998.0


# ---------- downgrade matrix ----------

def test_echo_mismatch_without_override_downgrades():
    bad = good_trade(echo={"regime": "trend", "always_in": "ais", "zone": "lower_third"})
    d, term = validate_decision(bad, sc_v2(ai="ail"))
    assert term == "echo_mismatch"
    assert d["action"] == "no_trade" and d["terminal_reason"] == "echo_mismatch"
    assert d["original_decision"]["action"] == "long"          # proposal preserved
    assert "_usage" not in d["original_decision"]              # meta stripped from snapshot
    assert d["_usage"]["provider"] == "zhipu"                  # ...but kept at top level


def test_echo_mismatch_with_override_preserved():
    d, term = validate_decision(
        good_trade(echo={"regime": "trend", "always_in": "ais", "zone": "lower_third"},
                   override={"field": "always_in", "claim": "ais",
                             "reason": "bar 6 broke the prior swing low, flipping AIS"}),
        sc_v2(ai="ail"))
    assert term is None and d["action"] == "long"             # accountable disagreement kept


def test_zone_echo_mismatch_downgrades():
    bad = good_trade(echo={"regime": "range", "always_in": "ail", "zone": "upper_third"})
    d, term = validate_decision(bad, sc_v2(zone="lower_third"))
    assert term == "echo_mismatch" and d["action"] == "no_trade"


def test_bad_bar_citation_downgrades():
    # reason is long enough (passes thin) but cites bar 99 > current bar 6
    bad = good_trade(reason="clean setup off the pullback with strong follow-through at bar 99")
    d, term = validate_decision(bad, sc_v2(bar=6))
    assert term == "reason_uncited" and d["action"] == "no_trade"


def test_no_bar_citation_downgrades():
    bad = good_trade(reason="a long and confident sounding justification with no bar number")
    d, term = validate_decision(bad, sc_v2())
    assert term == "reason_uncited"


def test_thin_reason_downgrades():
    d, term = validate_decision(good_trade(reason="looks good bar 5"), sc_v2())
    assert term == "reason_too_thin" and d["action"] == "no_trade"


def test_limit_without_entry_px_downgrades():
    d, term = validate_decision(good_trade(entry_type="limit", entry_px=None), sc_v2())
    assert term == "limit_no_px" and d["action"] == "no_trade"


def test_non_numeric_stop_target_downgrades():
    d, term = validate_decision(good_trade(stop=None), sc_v2())
    assert term == "non_numeric" and d["action"] == "no_trade"


# ---------- next_bar (drop block, never kill trade) ----------

def test_next_bar_bad_sum_drops_block_keeps_trade():
    d, term = validate_decision(
        good_trade(next_bar={"p_up": 40, "p_down": 30, "p_neutral": 10}), sc_v2())  # sum 80
    assert term is None and d["action"] == "long"
    assert "next_bar" not in d                    # malformed block dropped


def test_next_bar_non_integer_drops_block_keeps_trade():
    d, term = validate_decision(
        good_trade(next_bar={"p_up": 33.3, "p_down": 33.3, "p_neutral": 33.4}), sc_v2())
    assert term is None and d["action"] == "long" and "next_bar" not in d


def test_next_bar_boundary_sums_kept():
    for total in (99, 100, 101):
        d, term = validate_decision(
            good_trade(next_bar={"p_up": total - 2, "p_down": 1, "p_neutral": 1}), sc_v2())
        assert term is None and "next_bar" in d, total


# ---------- v1 sidecar graceful degradation ----------

def test_v1_sidecar_skips_zone_echo_check():
    # echo.zone would mismatch, but a v1 sidecar has no `zone` -> the zone check is skipped;
    # always_in still matches, so the trade passes.
    d, term = validate_decision(
        good_trade(echo={"regime": "trend", "always_in": "ail", "zone": "upper_third"}),
        sc_v1(ai="ail"))
    assert term is None and d["action"] == "long"


def test_pre_v2_reply_without_echo_or_next_bar_passes():
    # an old cached reply: no echo, no next_bar, but a cited substantive reason
    legacy = {"action": "short", "setup": "double top", "entry_type": "stop",
              "stop": 7030.0, "target": 7005.0, "confidence": 55,
              "reason": "lower second top at bar 4 under the EMA; bar 5 sold off, MTR in play"}
    d, term = validate_decision(legacy, sc_v1())
    assert term is None and d["action"] == "short"


# ---------- bar-citation regex forms ----------

def test_bar_citation_tolerant_forms():
    from pab.decision_schema import _has_valid_citation
    sc = {"bar_index_from_open": 12}
    for r in ("entry at bar 5", "see bar#7", "bar #3 signal", "bars 3-5 pullback",
              "range bar 10-12 fade"):
        assert _has_valid_citation(r, sc), r
    assert not _has_valid_citation("no numbers here", sc)
    assert not _has_valid_citation("bar 13 is beyond the window", sc)  # 13 > 12


# ---------- retry-direction guard (mock provider re-ask) ----------

def test_retry_flip_guard_forces_no_trade():
    from pab.agent import _reask_toward_no_trade
    # mock provider: attempt 1 parsed a no_trade, a re-ask parsed a long
    first = {"action": "no_trade", "setup": "", "reason": "no setup yet"}
    reask = {"action": "long", "setup": "breakout", "stop": 100.0, "target": 110.0,
             "reason": "bar 5 broke out", "_usage": {"provider": "zhipu"}}
    out = _reask_toward_no_trade(first, reask)
    assert out["action"] == "no_trade" and out["terminal_reason"] == "retry_flip"
    assert out["original_decision"]["action"] == "long"
    assert out["_usage"] == {"provider": "zhipu"}


def test_retry_flip_guard_side_flip_clamped():
    from pab.agent import _reask_toward_no_trade
    out = _reask_toward_no_trade({"action": "long"}, {"action": "short", "stop": 1.0})
    assert out["terminal_reason"] == "retry_flip"


def test_retry_guard_allows_toward_no_trade_directions():
    from pab.agent import _reask_toward_no_trade
    trade = {"action": "long", "stop": 1.0, "target": 3.0}
    assert _reask_toward_no_trade(None, trade) is trade            # first parse failed -> ok
    nt = {"action": "no_trade"}
    assert _reask_toward_no_trade({"action": "long"}, nt) is nt    # trade -> no_trade ok
    same = {"action": "long", "stop": 2.0}
    assert _reask_toward_no_trade({"action": "long"}, same) is same  # same side unchanged


# ---------- decide() wiring: post-cache validation ----------

def test_decide_applies_validator(monkeypatch):
    import pab.agent as ag
    fake = {"action": "long", "setup": "x", "entry_type": "stop", "stop": 100.0,
            "target": 110.0, "confidence": 50, "reason": "go", "_usage": {"provider": "mock"}}
    monkeypatch.setattr(ag, "_decide_anthropic", lambda *a, **k: dict(fake))
    monkeypatch.setenv("PA_PROVIDER", "anthropic")
    monkeypatch.setenv("PA_CACHE", "0")
    monkeypatch.setenv("PA_DECISION_V2", "1")
    out = ag.decide(sc_v2(), cfg=ag.AgentConfig(), cases=[])
    assert out["action"] == "no_trade" and out["terminal_reason"] == "reason_too_thin"
    assert out["_usage"]["provider"] == "mock"


def test_decide_caches_raw_and_validates_each_call(monkeypatch, tmp_path):
    import pab.agent as ag
    monkeypatch.setattr(ag, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("PA_PROVIDER", "anthropic")
    monkeypatch.setenv("PA_CACHE", "1")
    monkeypatch.setenv("PA_DECISION_V2", "1")
    raw = {"action": "long", "setup": "x", "entry_type": "stop", "stop": 100.0,
           "target": 110.0, "confidence": 50, "reason": "go", "_usage": {"provider": "mock"}}
    calls = {"n": 0}

    def prov(*a, **k):
        calls["n"] += 1
        return dict(raw)
    monkeypatch.setattr(ag, "_decide_anthropic", prov)

    out1 = ag.decide(sc_v2(), cfg=ag.AgentConfig(), cases=[])
    assert out1["action"] == "no_trade" and out1["terminal_reason"] == "reason_too_thin"
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["action"] == "long"   # RAW model output cached

    out2 = ag.decide(sc_v2(), cfg=ag.AgentConfig(), cases=[])
    assert calls["n"] == 1                                          # served from cache
    assert out2["action"] == "no_trade" and out2["terminal_reason"] == "reason_too_thin"
    assert out2["_usage"].get("cached") is True


def test_decision_v2_off_is_raw_passthrough(monkeypatch):
    import pab.agent as ag
    fake = {"action": "long", "setup": "x", "entry_type": "stop", "stop": 100.0,
            "target": 110.0, "confidence": 50, "reason": "go", "_usage": {"provider": "mock"}}
    monkeypatch.setattr(ag, "_decide_anthropic", lambda *a, **k: dict(fake))
    monkeypatch.setenv("PA_PROVIDER", "anthropic")
    monkeypatch.setenv("PA_CACHE", "0")
    monkeypatch.setenv("PA_DECISION_V2", "0")           # legacy: no validation
    out = ag.decide(sc_v2(), cfg=ag.AgentConfig(), cases=[])
    assert out["action"] == "long"                       # thin reason NOT downgraded


def test_decision_v2_off_system_prompt_has_no_output_v2(monkeypatch):
    import pab.agent as ag
    monkeypatch.setenv("PA_DECISION_V2", "0")
    assert "OUTPUT v2" not in ag._system(False, "trend", "mes")
    monkeypatch.setenv("PA_DECISION_V2", "1")
    assert "OUTPUT v2" in ag._system(False, "trend", "mes")
